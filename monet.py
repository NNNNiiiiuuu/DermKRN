import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DERMKRN_DATA", os.path.join(ROOT, "data"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 1 / np.exp(4.5944)
MONET = os.environ.get("MONET_REPO", os.path.join(ROOT, "MONET"))
PREFIXES = ["This is ", "This photo is ", "This lesion is ", "skin has become "]


def strip(p):
    for x in PREFIXES:
        p = p.replace(x, "")
    return p.lower()


def terms():
    spec = importlib.util.spec_from_file_location("monet_static", os.path.join(MONET, "src", "MONET", "utils", "static.py"))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    out = {}
    for col in st.skincon_cols:
        name = col[len("skincon_"):]
        prompts = [p for k, v in st.concept_to_prompt(name)[0].items() if k != "original" for p in v]
        out[re.sub("[^a-z]", "", re.split("[/(]", name.lower())[0])] = sorted({strip(p) for p in prompts})
    return out
CLINICAL = ["This is skin image of {}", "This is dermatology image of {}", "This is image of {}"]
DERMOSCOPY = ["This is dermatoscopy of {}", "This is dermoscopy of {}"]
TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TF(Image.open(self.paths[i]).convert("RGB"))


def prompts(ds, terms):
    if ds == "isic":
        return [[t.format(x) for t in DERMOSCOPY for x in terms]], [["This is dermatoscopy", "This is dermoscopy"]]
    return [[t.format(x) for x in terms] for t in CLINICAL], [["This is skin image"], ["This is dermatology image"], ["This is image"]]


@torch.no_grad()
def embed_text(model, tok, groups):
    out = []
    for g in groups:
        e = model.get_text_features(**tok(g, padding=True, truncation=True, return_tensors="pt").to(DEV)).float().cpu()
        out.append(e / e.norm(dim=1, keepdim=True))
    return out


def score(img, target, ref):
    t = np.stack([(g @ img.T).mean(0).numpy() for g in target])
    r = np.stack([(g @ img.T).mean(0).numpy() for g in ref])
    e = np.exp(np.stack([t, r]) / TEMP - np.maximum(t, r)[None] / TEMP)
    return (e[0] / e.sum(0)).mean(0)


@torch.no_grad()
def main(ds):
    path = os.path.join(DATA, ds + ".csv")
    df = pd.read_csv(path)
    proc = AutoProcessor.from_pretrained("chanwkim/monet")
    model = AutoModelForZeroShotImageClassification.from_pretrained("chanwkim/monet").to(DEV).eval()
    dl = DataLoader(Images([os.path.join(DATA, p) for p in df.image]), batch_size=64, num_workers=8)
    img = torch.cat([model.get_image_features(pixel_values=x.to(DEV)).float().cpu() for x in dl])
    img = img / img.norm(dim=1, keepdim=True)
    for c, words in terms().items():
        target, ref = prompts(ds, words)
        df["monet_" + c] = score(img, embed_text(model, proc.tokenizer, target), embed_text(model, proc.tokenizer, ref))
    df.to_csv(path, index=False)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["skincon", "isic"]:
        main(name)
