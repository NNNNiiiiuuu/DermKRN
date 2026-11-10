import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DERMKRN_DATA", os.path.join(ROOT, "data"))
SPLITS = os.path.join(ROOT, "splits")
FOLDER = {"fitz": "fitzpatrick17k", "ddi": "ddi"}


def monet(df):
    return [c for c in df.columns if c.lower().startswith("monet_")]


def export():
    os.makedirs(SPLITS, exist_ok=True)
    s = pd.read_csv(os.path.join(DATA, "skincon.csv"))
    s["image_id"] = s.image.map(os.path.basename)
    s[["image_id", "source", "label", "split"] + monet(s)].to_csv(os.path.join(SPLITS, "skincon_split.csv"), index=False)
    i = pd.read_csv(os.path.join(DATA, "isic.csv"))
    i["image"] = i.image.map(lambda p: os.path.splitext(os.path.basename(p))[0])
    i[["image", "label", "split"] + monet(i)].to_csv(os.path.join(SPLITS, "isic2019_split.csv"), index=False)


def build():
    s = pd.read_csv(os.path.join(SPLITS, "skincon_split.csv"))
    a = pd.concat([pd.read_csv(os.path.join(DATA, f"annotations_{f}.csv")) for f in ("fitzpatrick17k", "ddi")])
    s = s.merge(a.drop(columns="Do not consider this image", errors="ignore"), left_on="image_id", right_on="ImageID", how="left")
    s.insert(0, "image", s.source.map(FOLDER) + "/" + s.image_id)
    s.drop(columns=["image_id", "ImageID"]).to_csv(os.path.join(DATA, "skincon.csv"), index=False)
    m = pd.read_csv(os.path.join(DATA, "ISIC_2019_Training_Metadata.csv"))
    m = m.rename(columns={"age_approx": "age", "anatom_site_general": "site"})[["image", "age", "sex", "site"]]
    i = pd.read_csv(os.path.join(SPLITS, "isic2019_split.csv")).merge(m, on="image", how="left")
    i["image"] = "ISIC_2019_Training_Input/" + i.image + ".jpg"
    i.to_csv(os.path.join(DATA, "isic.csv"), index=False)


if __name__ == "__main__":
    {"export": export, "build": build}[sys.argv[1]]()
