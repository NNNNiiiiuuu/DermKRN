# DermKRN

Code for *DermKRN: A Knowledge-Guided Multimodal Reasoning Framework for Interpretable Dermatological Diagnosis*. Datasets are not included.

## Files

| File | Purpose |
|---|---|
| `prolog/skb.pl` | Static knowledge base (Table S1): 36 constraint rules (E/N), sufficient (S), differential (D) and global (G) rules, and the auxiliary predicates Elevated, Pigmented and MultiColor |
| `prolog/engine.pl` | Rule engine with Kleene's strong three-valued logic; returns the fired rules for an image and stores the mined DKB rules |
| `splits/` | Data splits used in the paper, with MONET concept scores |
| `splits.py` | Builds `data/skincon.csv` and `data/isic.csv` from the official dataset files and the split files |
| `monet.py` | Recomputes the MONET concept scores with the original MONET implementation and prompt templates (optional; the split files already contain the scores) |
| `knowledge.py` | Reasoning module: concept and metadata facts, PySwip queries, SKB weight calibration, DKB mining and constraint vectors |
| `train.py` | Backbones, fusion baselines, CrossViT, long-tail losses, knowledge-guided loss, Constrained Softmax and the training loop |
| `run.py` | All experiment configurations; trains every configuration with five seeds and then calls `report.py` |
| `report.py` | Metrics, statistical tests, tables and figures |
| `style.py` | Figure style |
| `requirements.txt` | Python dependencies |

## Usage

Requirements: Python 3.9, CUDA 11.8, SWI-Prolog 9.2.9.

```
pip install -r requirements.txt
```

Download ISIC 2019 (https://challenge.isic-archive.com/data/#2019), SKINCON (https://skincon-dataset.github.io), Fitzpatrick 17k (https://github.com/mattgroh/fitzpatrick17k) and DDI (https://ddi-dataset.github.io), and place them in `data/`:

```
data/
  ISIC_2019_Training_Input/ISIC_0000000.jpg ...
  ISIC_2019_Training_Metadata.csv
  fitzpatrick17k/<md5hash>.jpg ...
  ddi/<DDI_file> ...
  annotations_fitzpatrick17k.csv
  annotations_ddi.csv
```

Then run

```
python splits.py build
python run.py
```

To recompute the MONET concept scores, clone MONET into this folder (or set `MONET_REPO` to its path) and run `monet.py`; the concept term lists are read from the MONET repository:

```
git clone https://github.com/suinleelab/MONET.git
python monet.py skincon isic
```

Results are written to `results/`. Finished runs are skipped, so `run.py` can be resumed. `python report.py` regenerates the tables and figures from saved runs.

## License

The code is released under the MIT License. The files in `splits/` are derived from ISIC 2019, SKINCON, Fitzpatrick 17k and DDI and are subject to the terms of use of those datasets. MONET (https://github.com/suinleelab/MONET) is not included and is used under its own license.
