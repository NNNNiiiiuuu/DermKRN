import copy
import random
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
AUG = T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                 T.RandomRotation(30), T.ColorJitter(brightness=0.2, contrast=0.2), T.ToTensor(), NORM])
EVAL = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])
WORKERS = 8


class Images(Dataset):
    def __init__(self, paths, idx, tf):
        self.paths, self.idx, self.tf = paths, idx, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        return (self.tf(Image.open(self.paths[i]).convert("RGB")) if self.tf else torch.zeros(1)), i


def trunk(arch):
    m = getattr(torchvision.models, arch)(weights="DEFAULT")
    if arch.startswith("resnet"):
        return nn.Sequential(*list(m.children())[:-2]), m.fc.in_features
    if arch.startswith("densenet"):
        return nn.Sequential(m.features, nn.ReLU()), m.classifier.in_features
    if arch.startswith("efficientnet"):
        return m.features, m.classifier[1].in_features
    return m.features, 512


class Net(nn.Module):
    def __init__(self, arch, K, M, fusion=None, normed=False):
        super().__init__()
        self.arch, self.fusion, self.normed = arch, fusion, normed
        if arch == "mlp":
            self.head = nn.Sequential(nn.Linear(M, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, K))
            return
        if arch.startswith("crossvit"):
            self.f = timm.create_model(arch, pretrained=True, num_classes=K)
            return
        self.f, c = trunk(arch)
        if fusion == "metanet":
            self.g = nn.Sequential(nn.Linear(M, 512), nn.ReLU(), nn.Linear(512, c), nn.Sigmoid())
        if fusion == "metablock":
            self.t = nn.Sequential(nn.Linear(M, c), nn.BatchNorm1d(c))
            self.b = nn.Sequential(nn.Linear(M, c), nn.BatchNorm1d(c))
        self.head = nn.Linear(c + M * (fusion == "concat"), K, bias=not normed)

    def forward(self, x, m):
        if self.arch == "mlp":
            return self.head(m)
        if self.arch.startswith("crossvit"):
            return self.f(F.interpolate(x, size=240, mode="bilinear", align_corners=False))
        h = self.f(x)
        if self.fusion == "metanet":
            h = h * self.g(m)[..., None, None]
        if self.fusion == "metablock":
            h = torch.sigmoid(torch.tanh(h * self.t(m)[..., None, None]) + self.b(m)[..., None, None])
        h = h.mean((2, 3))
        if self.fusion == "concat":
            h = torch.cat([h, m], 1)
        return 30 * F.linear(F.normalize(h), F.normalize(self.head.weight)) if self.normed else self.head(h)


def objective(z, y, lv, w, o, ep, n):
    z, kind = z.float(), o["loss"]
    cb = (1 - 0.9999) / (1 - 0.9999 ** n)
    cb = cb / cb.sum() * len(n)
    if kind == "focal":
        lp = F.log_softmax(z, 1).gather(1, y[:, None])[:, 0]
        return (-(1 - lp.exp()) ** 2 * lp).mean()
    if kind == "cb":
        return F.cross_entropy(z, y, weight=cb)
    if kind == "ldam":
        m = n.pow(-0.25)
        m = m * 0.5 / m.max()
        return F.cross_entropy(z - 30 * m[y][:, None] * F.one_hot(y, len(n)), y, weight=cb if ep >= 80 else None)
    if kind == "la":
        return F.cross_entropy(z + (n / n.sum()).log(), y)
    if kind == "bs":
        return F.cross_entropy(z + n.log(), y)
    loss = F.cross_entropy(z + lv if o["cs"] else z, y)
    if o["lam"]:
        h = z if o.get("lk") == "raw" else z - z.mean(1, keepdim=True)
        loss = loss + o["lam"] * (w * F.relu(h)).sum() / ((w > 0).sum() + 1e-8)
    return loss


@torch.no_grad()
def predict(net, dl, MT):
    net.eval()
    out = []
    for x, i in dl:
        with torch.autocast(DEV, enabled=AMP):
            out.append(net(x.to(DEV), MT[i.to(DEV)]).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def latency(net, x, m, n=200):
    net.eval()
    ts = []
    for _ in range(n + 10):
        if AMP:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(DEV, enabled=AMP):
            net(x, m)
        if AMP:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts[10:])


def loader(d, idx, tf, shuffle=False, seed=0, keep=True):
    return DataLoader(Images(d["paths"], idx, tf), batch_size=64, shuffle=shuffle, num_workers=WORKERS,
                      pin_memory=AMP, persistent_workers=keep,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


def fit(d, o, seed, vc, meta):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ok = np.ones(len(d["y"]), bool) if o.get("src") is None else d["src"] == o["src"]
    tr = np.where((d["split"] == "train") & ok)[0]
    va = np.where((d["split"] == "val") & ok)[0]
    ev = np.where(d["split"] != "train")[0]
    lv = np.where(vc > 0, np.log(np.clip(vc, 1e-300, 1)), -100.0)
    LV = torch.tensor(lv, dtype=torch.float32, device=DEV)
    W = torch.tensor(1 - vc, dtype=torch.float32, device=DEV)
    MT = torch.tensor(meta, dtype=torch.float32, device=DEV)
    Y = torch.tensor(d["y"], device=DEV)
    n = torch.bincount(Y[tr], minlength=d["K"]).float()
    img = o["arch"] != "mlp"
    net = Net(o["arch"], d["K"], meta.shape[1], o.get("fusion"), o["loss"] == "ldam").to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    dl, dv = loader(d, tr, AUG if img else None, True, seed), loader(d, va, EVAL if img else None)
    if AMP:
        torch.cuda.reset_peak_memory_stats()
    best, bad, state, times = -1, 0, None, []
    for ep in range(100):
        net.train()
        t0 = time.time()
        for x, i in dl:
            x, i = x.to(DEV, non_blocking=True), i.to(DEV)
            with torch.autocast(DEV, enabled=AMP):
                z = net(x, MT[i])
            loss = objective(z, Y[i], LV[i], W[i], o, ep, n)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sch.step()
        times.append(time.time() - t0)
        b = balanced_accuracy_score(d["y"][va], (predict(net, dv, MT) + lv[va] * o["cs"]).argmax(1))
        if b > best:
            best, bad, state = b, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad == 10:
                break
    net.load_state_dict(state)
    x = EVAL(Image.open(d["paths"][ev[0]]).convert("RGB"))[None].to(DEV) if img else torch.zeros(1, 1, device=DEV)
    return dict(idx=ev, z=predict(net, loader(d, ev, EVAL if img else None, keep=False), MT), lv=lv[ev] * o["cs"],
                epoch_time=np.mean(times), mem=torch.cuda.max_memory_allocated() / 1e9 if AMP else 0.0,
                infer=latency(net, x, MT[ev[:1]]), best=best)
