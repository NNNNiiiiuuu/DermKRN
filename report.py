import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest, rankdata
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve

from knowledge import VALID, counterfactual, knowledge, load, lvof, qtime, sm
from run import ARCHS, DRAWS, GRID, KC, LAM, OUT, ROBUST, SEEDS
from style import C, FIG, FS, LEG, apply_style, calib, cbar, diag, grid, heat, hline, palette, panel, risk, roc, roc_macro, robust, save, size, vline

TAB, FIG_DIR = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
MAIN, IMG = "dermkrn_densenet121", "image_densenet121"
COLS = ["ACC", "BACC", "P", "F1", "AUC"]
DS = ("skincon", "isic")
TAUS = [-1, -2, -5, -10, -20, -50, -100]
TABLES, REL = {}, {}


@lru_cache(None)
def data(ds):
    return load(ds)


@lru_cache(None)
def kn(ds, ev):
    return knowledge(data(ds), KC[ds] | ({"skb": "hard"} if ev == "hard" else {}))


@lru_cache(16)
def run(ds, name):
    d, out = data(ds), []
    for s in SEEDS:
        f = np.load(os.path.join(OUT, ds, name, f"{s}.npz"))
        Z, LV = np.zeros((len(d["y"]), d["K"])), np.zeros((len(d["y"]), d["K"]))
        Z[f["idx"]], LV[f["idx"]] = f["z"], f["lv"]
        out.append(dict(Z=Z, LV=LV, et=float(f["epoch_time"]), mem=float(f["mem"]), inf=f["infer"]))
    return out


def probs(ds, name, ev=None):
    return [sm(r["Z"] + (r["LV"] if ev is None else 0 if ev == "plain" else lvof(kn(ds, ev)["vc"]))) for r in run(ds, name)]


def test(ds):
    return data(ds)["split"] == "test"


def met(y, p):
    K, yh, L = p.shape[1], p.argmax(1), np.unique(y)
    cm = np.bincount(y * K + yh, minlength=K * K).reshape(K, K)
    tp, pc = cm.diagonal()[L], cm.sum(0)[L]
    rec = tp / cm.sum(1)[L]
    pr = np.divide(tp, pc, out=np.zeros(len(L)), where=pc > 0)
    f1 = np.divide(2 * pr * rec, pr + rec, out=np.zeros(len(L)), where=pr + rec > 0)
    T, n1 = y[:, None] == L, np.bincount(y)[L]
    auc = ((rankdata(p[:, L], axis=0) * T).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    return np.array([(yh == y).mean(), rec.mean(), pr.mean(), f1.mean(), auc.mean()])


def ms(a, k=3):
    a = np.asarray(a, float)
    return f"{a.mean():.{k}f}±{a.std():.{k}f}"


def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def tab(ds, rows, m, sel=(0, 1, 2, 3, 4)):
    y = data(ds)["y"][m]
    recs = [[lab] + [ms(c) for c in np.array([met(y, p[m]) for p in probs(ds, n, e)]).T[list(sel)]] for lab, n, e in rows]
    return pd.DataFrame(recs, columns=["Method"] + [COLS[i] for i in sel])


def comp_rows():
    return [("Image only", IMG, None), ("Concatenation", "concat", None), ("MetaNet", "metanet", None),
            ("CrossViT", "crossvit", None), ("MetaBlock", "metablock", None), ("Focal loss", "focal", None),
            ("Class-balanced loss", "cb", None), ("LDAM-DRW", "ldam", None), ("Logit-adjusted loss", "la", None),
            ("Balanced Softmax", "bs", None), ("DermKRN", MAIN, None)]


def abl_rows(ds):
    s, i = ds == "skincon", ds == "isic"
    return [("Image only", IMG, None), ("Metadata only (MLP)", "mlp", None)] + \
        [("MetaBlock, patient metadata only", "metablock_meta", None)] * i + \
        [("CS at inference only", IMG, "main"), ("CS only (no L_K)", "cs_only", None),
         ("L_K only (no CS)", "lk_only", None), ("L_K + CS at inference", "lk_only", "main"),
         ("Original formulation", "original", None), ("DermKRN with hard SKB", "hard", None)] + \
        [("DermKRN without R-MEL-4", "no_rmel4", None)] * s + \
        [("DermKRN without DKB", "no_dkb", None), ("DermKRN without SKB", "no_skb", None)] * i + \
        [("DermKRN", MAIN, None), ("Shuffled-rule control", "shuffle", None), ("Random-rule control", "random", None)] + \
        [("DermKRN, patient metadata only", "dermkrn_meta", None),
         ("DermKRN, MONET concepts only", "dermkrn_concepts", None)] * i


def src_sets():
    d = data("skincon")
    t, src, lab = test("skincon"), d["src"], d["df"].label
    c = pd.crosstab(lab, d["df"].source)
    keep = np.isin(lab, c.index[(c >= 10).all(1)])
    nm = dict(image="Image only", concat="Concatenation", metanet="MetaNet", crossvit="CrossViT", metablock="MetaBlock",
              dermkrn="DermKRN")
    w, x = list(nm), ["image", "metablock", "dermkrn"]
    sets = [("DDI within", t & (src == "ddi"), "ddi", w), ("Fitzpatrick17k within", t & (src == "fitz"), "fitz", w),
            ("Fitzpatrick17k -> DDI", t & (src == "ddi") & keep, "fitz", x),
            ("DDI -> Fitzpatrick17k", t & (src == "fitz") & keep, "ddi", x)]
    return [(f"{s} (n={m.sum()})", m, [(nm[k], f"{p}_{k}", None) for k in ks]) for s, m, p, ks in sets]


def split_counts():
    for ds in DS:
        df = data(ds)["df"]
        TABLES[f"S2_split_{ds}"] = pd.crosstab(df.label, df.split, margins=True, margins_name="Total").reset_index()


def monet_table():
    d = data("skincon")
    df, m = d["df"], d["split"] != "test"
    rows = []
    for c in VALID + ["excoriation", "vesicle"]:
        t, s = df.loc[m, c].fillna(0).values.astype(int), df.loc[m, "monet_" + c].values
        rows.append([c, t.sum(), round(t.mean(), 3), round(roc_auc_score(t, s), 3), round(average_precision_score(t, s), 3)])
    out = pd.DataFrame(rows, columns=["Concept", "Positives", "Prevalence", "AUROC", "AUPRC"]).sort_values("AUROC", ascending=False)
    a = out.AUROC.values
    TABLES["S3_monet"] = pd.concat([out, pd.DataFrame([["Mean±SD", "", "", ms(a), f"{(a >= 0.7).sum()}/{len(a)} ≥ 0.70"]], columns=out.columns)])


def sensitivity():
    recs = []
    for ds in DS:
        m, L, kc = data(ds)["split"] == "val", LAM[ds], KC[ds]
        y = data(ds)["y"][m]
        items = [("lambda", v, MAIN if v == L else "cs_only" if v == 0 else f"lam_{v}", {}) for v in GRID["lam"]]
        if ds == "isic":
            items += [(p, v, MAIN if v == kc[p] else f"{p}_{v}", {p: v} if p != "g0" else {})
                      for p in ("m", "rho", "g0") for v in GRID[p]]
        for p, v, n, k in items:
            M = np.array([met(y, q[m]) for q in probs(ds, n)])
            nd = len((knowledge(data(ds), kc | k) if k else kn(ds, "main"))["dk"])
            recs.append([ds, p, v, nd] + [ms(M[:, i]) for i in (0, 1, 4)])
    TABLES["sensitivity_val"] = pd.DataFrame(recs, columns=["Dataset", "Parameter", "Value", "DKB rules", "ACC", "BACC", "AUC"])


def main_tables():
    for ds in DS:
        rows = [(f"{a} {lab}", f"{p}_{a}", None) for a in ARCHS for lab, p in (("Baseline", "image"), ("+ DermKRN", "dermkrn"))]
        TABLES[f"backbones_{ds}"] = tab(ds, rows, test(ds))
        TABLES[f"comparison_{ds}"] = tab(ds, comp_rows(), test(ds))
        TABLES[f"ablation_{ds}"] = tab(ds, abl_rows(ds), test(ds))
    TABLES["S8_monet_skincon"] = tab("skincon", [("DermKRN, expert concepts", MAIN, None),
                                                 ("DermKRN, MONET concepts", "dermkrn_monet", None),
                                                 ("Image only", IMG, None)], test("skincon"))
    TABLES["sources_skincon"] = pd.concat([tab("skincon", rows, m, (0, 1, 3, 4)).assign(Setting=s)[["Setting", "Method", "ACC", "BACC", "F1", "AUC"]]
                                           for s, m, rows in src_sets()])


def dkb_table():
    k, d = kn("isic", "main"), data("isic")
    ti, recs = np.where(test("isic"))[0], []
    for i, (j, u, v, g, mt, md) in enumerate(k["dk"]):
        f = [t for t in ti if i in k["X"][t]]
        a = np.exp(-g * k["S"][f, u] * k["S"][f, v])
        recs.append([f"D{i + 1}", f"Penalty({d['classes'][j]}) <- {k['names'][u]} AND {k['names'][v]}", round(g, 3),
                     round(mt, 3), round(md, 4), len(f), ms(a) if len(f) else "-"])
    TABLES["dkb_rules_isic"] = pd.DataFrame(recs, columns=["ID", "Rule", "gamma", "M_target", "M_d", "Firings (test)", "alpha"]) \
        .sort_values("gamma", ascending=False, kind="stable")


def classwise():
    d, m = data("isic"), test("isic")
    y, K = d["y"][m], d["K"]
    R = np.mean([precision_recall_fscore_support(y, p[m].argmax(1), labels=range(K), zero_division=0)[:3] for p in probs("isic", MAIN)], 0)
    sup = np.bincount(y, minlength=K)
    rows = [[c, *np.round(R[:, j], 4), sup[j]] for j, c in enumerate(d["classes"])] + [["Macro average", *np.round(R.mean(1), 4), sup.sum()]]
    TABLES["classwise_isic"] = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1", "Support"])


def effect():
    out, audit = {}, []
    for ds in DS:
        d, k = data(ds), kn(ds, "main")
        y, ti = d["y"], np.where(test(ds))[0]
        n, yt = len(ti), d["y"][ti]
        fir = [(i, r) for i in ti for r in k["E"][i]]
        col = {"(a) Images with >=1 SKB rule": sum(bool(k["E"][i]) for i in ti),
               "(a) Images with >=1 DKB rule": sum(bool(k["X"][i]) for i in ti),
               "(a) SKB firings on true class / all SKB firings": f"{sum(k['rules'][r][0] == y[i] for i, r in fir)} / {len(fir)}",
               "(a) True class down-weighted by an SKB rule": sum(k["vS"][i, y[i]] < 1 for i in ti),
               "(a) SKB and DKB act on the same class": sum(((k["vS"][i] < 1) & (k["vD"][i] < 1)).any() for i in ti)}
        cred, B, C = {r: [0, 0] for r in k["rules"]}, [], []
        for s, r_ in zip(SEEDS, run(ds, MAIN)):
            p0, p1 = sm(r_["Z"][ti]), sm(r_["Z"][ti] + r_["LV"][ti])
            a, b = p0.argmax(1), p1.argmax(1)
            ch = a != b
            wc, cw, ww = ch & (a != yt) & (b == yt), ch & (a == yt) & (b != yt), ch & (a != yt) & (b != yt)
            sk = np.array([k["vS"][i, c] <= k["vD"][i, c] for i, c in zip(ti, a)])
            for q, sel in ((0, wc & sk), (1, cw & sk)):
                for i, c in zip(ti[sel], a[sel]):
                    cred[min([r for r in k["E"][i] if k["rules"][r][0] == c], key=lambda r: 1 - k["rules"][r][3])][q] += 1
            B.append([ch.sum(), wc.sum(), (wc & sk).sum(), (wc & ~sk).sum(), cw.sum(), (cw & sk).sum(), (cw & ~sk).sum(),
                      ww.sum(), (wc.sum() - cw.sum()) / n * 100])
            res = counterfactual(k, r_["Z"], ti, np.random.default_rng(s))
            cnt = {x: sum(o == x for o, _ in res) for x in ("prescribed", "stronger", "opposite")}
            C.append([n, n - sum(o == "same" for o, _ in res), cnt["prescribed"], cnt["stronger"], cnt["opposite"], sum(t for _, t in res)])
        B, C = np.array(B, float), np.array(C, float)
        pct = lambda v, base: f"{v.mean():.1f}±{v.std():.1f} ({100 * v.mean() / base:.1f}%)"
        names_b = ["Top-1 prediction changed", "wrong -> correct", "credited to SKB", "credited to DKB", "correct -> wrong",
                   "credited to SKB ", "credited to DKB ", "wrong -> wrong"]
        col |= {f"(b) {nb}": pct(B[:, j], n) for j, nb in enumerate(names_b)}
        col["(b) Net change in ACC (points)"] = f"{B[:, 8].mean():+.1f}"
        ch = C[:, 1].mean()
        col |= {"(c) Interventions": f"{C[:, 0].mean():.0f}",
                "(c) firing of the chosen rule changed": f"{ch:.1f} ({100 * ch / n:.1f}%)",
                "(c) probability moved as prescribed": f"{C[:, 2].mean():.1f} ({100 * C[:, 2].mean() / ch:.1f}%)",
                "(c) no change, stronger factor active": f"{C[:, 3].mean():.1f} ({100 * C[:, 3].mean() / ch:.1f}%)",
                "(c) opposite direction": f"{C[:, 4].mean():.1f} ({100 * C[:, 4].mean() / ch:.1f}%)",
                "(c) top-1 prediction changed": f"{C[:, 5].mean():.1f} ({100 * C[:, 5].mean() / n:.1f}%)"}
        out[f"{ds} (n={n})"] = col
        for r, (j, w0, pi, w) in k["rules"].items():
            f = [i for i, rr in fir if rr == r]
            audit.append([ds, r, d["classes"][j], w0, round(pi, 3), round(w, 3), len(f), sum(y[i] == j for i in f),
                          cred[r][0] / len(SEEDS), cred[r][1] / len(SEEDS)])
    TABLES["effect"] = pd.DataFrame(out).fillna("-").rename_axis("Quantity").reset_index()
    TABLES["S4_audit"] = pd.DataFrame(audit, columns=["Dataset", "Rule", "Class", "w0", "pi_hat", "w", "Firings", "On true class",
                                                      "Corrected", "Introduced"])


def tau():
    recs = []
    for ds in DS:
        vc, m = kn(ds, "hard")["vc"], test(ds)
        for t in TAUS:
            same, dp = [], []
            for r in run(ds, "hard"):
                p0, p1 = sm(r["Z"][m] + lvof(vc[m])), sm(r["Z"][m] + lvof(vc[m], t))
                same.append((p0.argmax(1) == p1.argmax(1)).mean())
                dp.append(np.abs(p1 - p0).max())
            recs.append([ds, t, f"{100 * np.mean(same):.2f}%", f"{np.mean(dp):.2g}"])
    TABLES["S6_tau"] = pd.DataFrame(recs, columns=["Dataset", "tau", "Same prediction as tau=-100", "max |dp| vs tau=-100"])


def nll(y, z):
    return -np.log(sm(z)[np.arange(len(y)), y] + 1e-12).mean()


def calib_stats(y, z):
    p = sm(z)
    conf, acc = p.max(1), p.argmax(1) == y
    b = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(acc[b == i].mean() - conf[b == i].mean()) * (b == i).mean() for i in range(15) if (b == i).any())
    o = np.argsort(-conf)
    risk = np.cumsum(~acc[o]) / np.arange(1, len(y) + 1)
    return [ece, ((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean(), nll(y, z), risk.mean(),
            acc[o][:int(np.ceil(0.9 * len(y)))].mean()], conf, acc


def reliability():
    recs = []
    for ds in DS:
        d = data(ds)
        mt, mv = test(ds), d["split"] == "val"
        for lab, n in (("Image only", IMG), ("MetaBlock", "metablock"), ("DermKRN", MAIN)):
            S, conf, acc = [], [], []
            for r in run(ds, n):
                zt, zv = r["Z"][mt] + r["LV"][mt], r["Z"][mv] + r["LV"][mv]
                T = minimize_scalar(lambda t: nll(d["y"][mv], zv / t), bounds=(0.05, 10), method="bounded").x
                s, c, a = calib_stats(d["y"][mt], zt)
                S.append(s[:3] + [T, calib_stats(d["y"][mt], zt / T)[0][0]] + s[3:])
                conf.append(c)
                acc.append(a)
            S = np.array(S)
            REL[ds, lab] = (S[:, 0].mean(), S[:, 5].mean(), np.concatenate(conf), np.concatenate(acc))
            recs.append([ds, lab] + [ms(S[:, j]) for j in range(7)])
    TABLES["reliability"] = pd.DataFrame(recs, columns=["Dataset", "Method", "ECE", "Brier", "NLL", "T", "ECE after TS", "AURC",
                                                        "Acc. at 90% cov."])


def cost():
    col = {}
    for ds in DS:
        k = kn(ds, "main")
        for lab, n in (("Image only", IMG), ("DermKRN", MAIN)):
            R = run(ds, n)
            c = {"Training time per epoch (s)": f"{np.mean([r['et'] for r in R]):.1f}",
                 "Inference per image (ms)": f"{R[0]['inf'].mean():.2f}±{R[0]['inf'].std():.2f}",
                 "Peak GPU memory, training (GB)": f"{np.mean([r['mem'] for r in R]):.2f}"}
            if n == MAIN:
                q = qtime(k)
                c |= {"Constraint rules (SKB + DKB)": f"{sum(any(r in e for e in k['E']) for r in k['rules'])} + {len(k['dk'])}",
                      "Prolog query per image (ms)": f"{q[0]:.3f}±{q[1]:.3f}",
                      "Constraint vectors, all images (s)": f"{k['tvec']:.1f}",
                      "DKB mining, training split (s)": f"{k['tmine']:.1f}" if k["dk"] else "-"}
            col[f"{ds} {lab}"] = c
    k = kn("skincon", "main")
    col["skincon DermKRN"] |= {f"Prolog query, {36 * x} rules (ms)": f"{qtime(k, dup=x)[0]:.3f}" for x in (1, 2, 4, 8)}
    TABLES["cost"] = pd.DataFrame(col).fillna("-").rename_axis("Quantity").reset_index()


def cases():
    d, k = data("skincon"), kn("skincon", "main")
    r, cl, rows = run("skincon", MAIN)[0], d["classes"], []
    for i in np.where(test("skincon"))[0]:
        v = k["vc"][i]
        p0, p1 = sm(r["Z"][i]), sm(r["Z"][i] + np.log(v))
        a, b, y = p0.argmax(), p1.argmax(), d["y"][i]
        top = list(dict.fromkeys(list(np.argsort(-p0)[:3]) + list(np.argsort(-p1)[:3])))
        rows.append([d["df"].image[i], cl[y], ", ".join(n for n, t in zip(k["names"], k["T"][i]) if t == 1),
                     "; ".join(f"{cl[c]} {p0[c]:.3f}x{v[c]:.3f}->{p1[c]:.3f}" for c in top),
                     "; ".join(f"{cl[k['rules'][q][0]]}: {q} ({1 - k['rules'][q][3]:.3f})" for q in k["E"][i] if k["rules"][q][0] in top),
                     round((p0 * v).sum(), 3), "C" if a != y == b else "U" if a == y == b else "W" if a != y != b else "R"])
    pd.DataFrame(rows, columns=["Image", "True", "Present concepts", "p_before x v -> p_after", "Rules (factor)", "Z", "Case"]) \
        .to_csv(os.path.join(TAB, "cases_skincon_seed2025.csv"), index=False)


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    out = np.empty(len(p))
    out[o] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    return out


def mcnemar(y, a, b):
    ca, cb = a.argmax(1) == y, b.argmax(1) == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0


def boot(y, PA, PB, B=2000):
    rng, n = np.random.default_rng(0), len(y)
    f = lambda ix: np.mean([met(y[ix], a[ix]) for a in PA], 0) - np.mean([met(y[ix], b[ix]) for b in PB], 0)
    bs = np.array([f(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    p = min(1.0, 2 * min((bs[:, 1] <= 0).mean(), (bs[:, 1] >= 0).mean()))
    return f(np.arange(n)), lo, hi, p, sum(mcnemar(y, a, b) < 0.05 for a, b in zip(PA, PB))


def groups():
    out = []
    for ds in DS:
        pairs = [(f"{a}: + DermKRN vs. baseline", (f"dermkrn_{a}", None), (f"image_{a}", None)) for a in ARCHS]
        seen = set()
        for lab, n, e in comp_rows()[:-1] + abl_rows(ds):
            if (n, e) != (MAIN, None) and lab not in seen:
                seen.add(lab)
                pairs.append((f"DermKRN vs. {lab}", (MAIN, None), (n, e)))
        out.append((ds, ds, test(ds), pairs))
    for s, m, rows in src_sets():
        out.append((s, "skincon", m, [(f"DermKRN vs. {lab}", rows[-1][1:], (n, e)) for lab, n, e in rows[:-1]]))
    return out


def stats():
    recs = []
    for g, ds, m, pairs in groups():
        y = data(ds)["y"][m]
        res = [(lab,) + boot(y, [p[m] for p in probs(ds, *A)], [p[m] for p in probs(ds, *B)]) for lab, A, B in pairs]
        for (lab, obs, lo, hi, p, mc), h in zip(res, holm([r[4] for r in res])):
            recs.append([g, lab] + [f"{obs[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}]" for j in (0, 1, 3, 4)] +
                        [pv(p), pv(h), f"{mc}/{len(SEEDS)}"])
    TABLES["S5_paired_tests"] = pd.DataFrame(recs, columns=["Test set", "Comparison", "dACC [95% CI]", "dBACC [95% CI]",
                                                            "dF1 [95% CI]", "dAUC [95% CI]", "p (BACC)", "p_Holm", "McNemar"])


NAMES = {"ACN": "Acne", "ACD": "Allergic Contact Dermatitis", "BCC": "Basal Cell Carcinoma", "DE": "Drug Eruption",
         "ECZ": "Eczema", "EM": "Erythema Multiforme", "FOL": "Folliculitis", "GA": "Granuloma Annulare",
         "LE": "Lupus Erythematosus", "LP": "Lichen Planus", "MEL": "Melanoma", "MN": "Melanocytic Nevi",
         "ND": "Neutrophilic Dermatoses", "NI": "Nematode Infection", "PHD": "Photodermatoses",
         "PRP": "Pityriasis Rubra Pilaris", "PSO": "Psoriasis", "SAR": "Sarcoidosis", "SCB": "Scabies",
         "SCC": "Squamous Cell Carcinoma", "SCL": "Scleroderma", "SK": "Seborrheic Keratosis", "VV": "Verruca Vulgaris",
         "AK": "Actinic keratosis (AK)", "BKL": "Benign keratosis (BKL)", "DF": "Dermatofibroma (DF)",
         "NV": "Melanocytic nevus (NV)", "VASC": "Vascular lesion (VASC)"}
ISIC_NAMES = {"BCC": "Basal cell carcinoma (BCC)", "MEL": "Melanoma (MEL)", "SCC": "Squamous cell carcinoma (SCC)"}


def class_names(ds):
    cl = data(ds)["classes"]
    names = [(ISIC_NAMES.get(c) if ds == "isic" else None) or NAMES[c] for c in cl]
    order = sorted(range(len(cl)), key=lambda k: names[k].lower()) if ds == "skincon" else list(range(len(cl)))
    return order, [names[k] for k in order]


def fig_confusion():
    for ds in DS:
        d, m = data(ds), test(ds)
        y, K = d["y"][m], d["K"]
        order, names = class_names(ds)
        tall = ds == "skincon"
        spec = FIG["heat_tall" if tall else "heat_wide"]
        fig, axes = plt.subplots(2 if tall else 1, 1 if tall else 2, figsize=size(spec))
        title = ["DenseNet121 image only", "DenseNet121 + DermKRN"] if tall else ["Image only", "DermKRN"]
        for k, (ax, n) in enumerate(zip(axes, (IMG, MAIN))):
            P = probs(ds, n)
            cm = sum(np.bincount(y * K + p[m].argmax(1), minlength=K * K).reshape(K, K) for p in P)[np.ix_(order, order)]
            b = np.mean([met(y, p[m])[1] for p in P])
            t = f"({'ab'[k]}) {title[k]} (BACC {b:.3f}, counts pooled over {len(SEEDS)} seeds)" if tall else \
                f"({'ab'[k]}) {title[k]}: counts pooled over {len(SEEDS)} seeds"
            im = heat(ax, cm, names, t, counts=not tall, fs=FS["heat_small" if tall else "heat_big"], ylab=tall or k == 0)
            if tall:
                cbar(fig, im, ax, **spec["cbar"])
        if not tall:
            cbar(fig, im, list(axes), **spec["cbar"])
        save(fig, os.path.join(FIG_DIR, f"confusion_{ds}.pdf"), rect=spec["rect"], tight=spec.get("tight", False))


def fig_roc():
    for ds in DS:
        d, m = data(ds), test(ds)
        y = d["y"][m]
        order, names = class_names(ds)
        colors = palette(len(order))
        fig, axes = plt.subplots(1, 2, figsize=size(FIG["roc"]))
        for k, (ax, n, lab) in enumerate(zip(axes, (IMG, MAIN), ("Image only", "DermKRN"))):
            p = probs(ds, n)[0][m]
            grid_x, tprs = np.linspace(0, 1, 1001), []
            for c, name, col in zip(order, names, colors):
                fpr, tpr, _ = roc_curve(y == c, p[:, c])
                roc(ax, fpr, tpr, col, f"{name} (AUC = {roc_auc_score(y == c, p[:, c]):.3f})")
                tprs.append(np.interp(grid_x, fpr, tpr))
            roc_macro(ax, grid_x, np.mean(tprs, 0), f"Macro average (AUC = {met(y, p)[4]:.3f})")
            diag(ax)
            grid(ax)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(**LEG["roc_small" if ds == "skincon" else "roc_big"])
            panel(ax, "ab"[k], f"{lab} (seed {SEEDS[0]})")
        save(fig, os.path.join(FIG_DIR, f"roc_{ds}.pdf"), rect=FIG["roc"]["rect"])


def fig_robust():
    xlab = dict(missing="Fraction of entries", flip="Fraction of entries", noise="σ", delete="Number of rules",
                redirect="Number of rules")
    title = {"skincon": dict(missing="Missing concepts / metadata", flip="Flipped concepts", delete="Deleted SKB rules",
                             redirect="Corrupted SKB rules"),
             "isic": dict(missing="Missing concepts / metadata", flip="Flipped concepts / corrupted metadata",
                          noise="Noisy MONET scores (σ)")}
    for ds in DS:
        m = test(ds)
        y = data(ds)["y"][m]
        bacc = lambda name: [met(y, p[m])[1] for p in probs(ds, name)]
        kinds = ROBUST[ds]
        fig, axes = plt.subplots(1, len(kinds), figsize=size(FIG["robust"], len(kinds)), sharey=True)
        for k, (ax, (kind, levels)) in enumerate(zip(axes, kinds.items())):
            for skb, base in (("graded", MAIN), ("hard", "hard")):
                v = [bacc(base)] + [sum((bacc(f"rob_{kind}_{x}_{skb}_{r}") for r in range(DRAWS)), []) for x in levels]
                mu, sd = np.array([np.mean(a) for a in v]), np.array([np.std(a) for a in v])
                robust(ax, [0] + levels, mu, mu - sd, mu + sd, skb, f"DermKRN ({skb} SKB)")
            hline(ax, np.mean(bacc(IMG)), "Image only")
            grid(ax)
            ax.set_xlabel(xlab[kind])
            ax.set_title(title[ds][kind], fontweight="bold")
            if k == 0:
                ax.set_ylabel("BACC (test)")
                ax.legend(**LEG[FIG["robust"]["leg"]])
        save(fig, os.path.join(FIG_DIR, f"robust_{ds}.pdf"), rect=FIG["robust"]["rect"])


def fig_calibration():
    fig, axes = plt.subplots(2, 2, figsize=size(FIG["calib"]))
    for col, ds in enumerate(DS):
        a1, a2 = axes[0, col], axes[1, col]
        name = "SKINCON" if ds == "skincon" else "ISIC 2019"
        for lab, color in (("Image only", C["image"]), ("MetaBlock", C["hard"]), ("DermKRN", C["ours"])):
            ece, aurc, conf, acc = REL[ds, lab]
            b = np.minimum((conf * 15).astype(int), 14)
            k = [i for i in range(15) if (b == i).any()]
            calib(a1, [conf[b == i].mean() for i in k], [acc[b == i].mean() for i in k], color,
                  f"{lab} (ECE {ece:.3f})", lab == "DermKRN")
            o = np.argsort(-conf)
            n = np.arange(1, len(o) + 1)
            risk(a2, n / len(o), np.cumsum(~acc[o]) / n, color, f"{lab} (AURC {aurc:.3f})", lab == "DermKRN")
        diag(a1)
        vline(a2, 0.9)
        a1.set_xlabel("Confidence")
        a1.set_ylabel("Accuracy")
        a2.set_xlabel("Coverage")
        a2.set_ylabel("Risk (error rate)")
        for ax, L, t in ((a1, "ab"[col], "reliability"), (a2, "cd"[col], "risk-coverage")):
            grid(ax)
            ax.legend(**LEG[FIG["calib"]["leg"]])
            panel(ax, L, f"{name}: {t} (pooled seeds)")
    save(fig, os.path.join(FIG_DIR, "calibration_selective.pdf"), rect=FIG["calib"]["rect"])


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    apply_style()
    for f in (split_counts, monet_table, sensitivity, main_tables, dkb_table, classwise, effect, tau, reliability, cost,
              cases, stats, fig_confusion, fig_roc, fig_robust, fig_calibration):
        f()
    for name, df in TABLES.items():
        df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
        print(f"\n[{name}]\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
