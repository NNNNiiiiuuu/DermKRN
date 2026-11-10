import gc
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
SEEDS = [2025, 2026, 2027, 2028, 2029]
ARCHS = ["vgg16", "resnet18", "resnet50", "efficientnet_b4", "densenet121"]
FUSION = {"concat": "resnet50", "metanet": "resnet50", "metablock": "efficientnet_b4"}
CROSSVIT = "crossvit_18_dagger_240"
LAM = {"skincon": 0.05, "isic": 0.1}
KC = {"skincon": dict(concepts="expert", meta=False, skb="graded", dkb=True, alpha="exp", m=0.01, rho=3.5, g0=1.0),
      "isic": dict(concepts="monet", meta=True, skb="graded", dkb=True, alpha="exp", m=0.01, rho=3.5, g0=1.0)}
GRID = dict(lam=[0, 0.01, 0.05, 0.1, 0.5, 1, 2], m=[0.005, 0.01, 0.02, 0.05, 0.1], rho=[1.5, 2, 2.5, 3, 3.5, 4],
            g0=[0.25, 0.5, 1, 2, 4])
ROBUST = {"skincon": dict(missing=[0.1, 0.2, 0.3, 0.4, 0.5], flip=[0.05, 0.1, 0.2, 0.3], delete=[4, 7, 11, 14, 18],
                          redirect=[4, 7, 11]),
          "isic": dict(missing=[0.1, 0.2, 0.3, 0.4, 0.5], flip=[0.05, 0.1, 0.2, 0.3], noise=[0.05, 0.1, 0.2, 0.3])}
DRAWS = 3


def R(name, kc=None, **o):
    return dict(name=name, kc=kc, arch="densenet121", fusion=None, loss="ce", lam=0, cs=False, meta="full",
                src=None) | o


def runs(ds):
    L, kc = LAM[ds], KC[ds]

    def K(name, k=None, **o):
        return R(name, kc | (k or {}), lam=L, cs=True) | o

    out = [R(f"image_{a}", arch=a) for a in ARCHS] + [K(f"dermkrn_{a}", arch=a) for a in ARCHS]
    out += [R(f, arch=a, fusion=f) for f, a in FUSION.items()] + [R("crossvit", arch=CROSSVIT)]
    out += [R(x, loss=x) for x in ("focal", "cb", "ldam", "la", "bs")] + [R("mlp", arch="mlp")]
    out += [K("cs_only", lam=0), K("lk_only", cs=False), K("original", dict(skb="hard", alpha="prod"), lk="raw"),
            K("hard", dict(skb="hard")), K("shuffle", dict(control="shuffle")), K("random", dict(control="random"))]
    out += [K(f"lam_{v}", lam=v) for v in GRID["lam"] if v not in (0, L)]
    if ds == "skincon":
        out += [K("no_rmel4", dict(drop=["R-MEL-4"])), K("dermkrn_monet", dict(concepts="monet"))]
        for s in ("ddi", "fitz"):
            out += [R(f"{s}_image", src=s), R(f"{s}_crossvit", arch=CROSSVIT, src=s), K(f"{s}_dermkrn", src=s)]
            out += [R(f"{s}_{f}", arch=a, fusion=f, src=s) for f, a in FUSION.items()]
    else:
        out += [K("no_dkb", dict(dkb=False)), K("no_skb", dict(skb=None)),
                R("metablock_meta", arch=FUSION["metablock"], fusion="metablock", meta="patient"),
                K("dermkrn_meta", dict(concepts=None)), K("dermkrn_concepts", dict(meta=False))]
        out += [K(f"{p}_{v}", {p: v}) for p in ("m", "rho", "g0") for v in GRID[p] if v != kc[p]]
    for kind, levels in ROBUST[ds].items():
        for v in levels:
            for skb in ("graded", "hard"):
                out += [K(f"rob_{kind}_{v}_{skb}_{r}", dict(skb=skb, pert=kind, level=v, draw=r)) for r in range(DRAWS)]
    return out


def main():
    from knowledge import knowledge, load, meta
    from train import fit
    for ds in ("skincon", "isic"):
        d, cache = load(ds), {}
        for r in runs(ds):
            for seed in SEEDS:
                f = os.path.join(OUT, ds, r["name"], f"{seed}.npz")
                if os.path.exists(f):
                    continue
                kc = r["kc"]
                if kc is None:
                    vc = np.ones((len(d["y"]), d["K"]))
                else:
                    key = (repr(sorted(kc.items())), kc.get("draw", seed if kc.get("control") else 0), r["src"])
                    if key not in cache:
                        cache = {key: knowledge(d, kc, key[1], r["src"])["vc"]}
                    vc = cache[key]
                os.makedirs(os.path.dirname(f), exist_ok=True)
                np.savez(f, **fit(d, r, seed, vc, meta(d, r["meta"])))
                gc.collect()
    import report
    report.main()


if __name__ == "__main__":
    main()
