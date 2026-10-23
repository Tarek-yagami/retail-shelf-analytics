# Retail Shelf Analytics

Count beverage products on store shelf photos and measure a brand's shelf share against its competitors.

This started as a team project at a hackathon in February 2025, where the challenge came from Ramy, an Algerian juice company. It asked for a system that spots Ramy products on shelves, so merchandising teams can track shelf share and compare against competitors without counting by hand. This repository rebuilds the computer vision side properly. The hackathon version is kept in the history at the tag [`hackathon-2025`](../../tree/hackathon-2025).

**Status: work in progress.** The data has been audited and rebuilt. Model training comes next.

## What happened at the hackathon

We hand-labeled about 13,000 product boxes across 35 brand and packaging classes in Roboflow, then trained a YOLOv8 detector. It found roughly 1 product in 9 (mAP50 0.12, recall 0.11) and tended to call any orange bottle "Ramy". Auditing the data afterwards showed why:

- 84% of the boxes were Ramy products, so one detector learning 35 classes at once mostly learned "Ramy".
- Every photo was stretched to 640×640, which squashed tall shelf photos and left each product about 32×65 pixels.
- The 150 training images were 50 photos, each saved three times with a small rotation.
- Some competitor products were labeled as Ramy, and many products had no box at all.

## Approach

1. **Detector.** Find every product on the shelf without naming it, so the class imbalance doesn't matter at this stage. Start from a model pretrained on [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19), a public dataset of densely packed shelves.
2. **Classifier.** Name each cropped product at three levels: Ramy or not, brand, exact product. Product photos alone are enough to train this, so rare brands get data cheaply, and it can answer "unknown" instead of guessing.
3. **Shelf share.** Count products per brand on each photo.

Everything is scored on a fixed test set of 16 photos the hackathon model never trained on, so the before and after numbers are comparable.

## Data

The shelf photos and product images were provided for the hackathon and are not included in this repository. The code expects them in `data/raw/` (git-ignored). The class counts are in [`reports/data_audit.md`](reports/data_audit.md).

## Usage

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. PyTorch is installed with CUDA 12.8 support.

```
uv sync                        # install the environment
uv run pytest                  # run the tests
uv run shelf import-roboflow   # rebuild data/processed from data/raw
uv run shelf audit             # write reports/data_audit.md
```

## Repository layout

| Path | Contents |
| --- | --- |
| `configs/` | Product labels with brand and packaging, dataset paths and split settings |
| `src/retail_shelf_analytics/data/` | Dataset import from the Roboflow export, and the audit |
| `tests/` | Tests on small synthetic datasets shaped like the real export |
| `reports/` | Audit results and, later, evaluation results |

## Team

Built at the hackathon by Kao, Rayane, The-soulless12 and Tarek Benameur ([original repository](https://github.com/kaouthar-lefkir/AUP-2025)). Kao prepared the data augmentation, Rayane built the Flask backend and the dashboard, The-soulless12 added the data, Tarek labeled the dataset and trained the YOLO models, and a teammate built a Kotlin Android app around the model. Tarek is carrying the rebuild.

## License

[AGPL-3.0](LICENSE), the same license as [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which this project uses.
