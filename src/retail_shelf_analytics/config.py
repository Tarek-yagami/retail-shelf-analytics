"""Load the YAML files in configs/ into typed objects shared by every pipeline stage."""

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class ProductLabel:
    name: str
    brand: str
    packaging: str | None


@dataclass(frozen=True)
class ClassConfig:
    target_brand: str
    labels: tuple[ProductLabel, ...]

    def label_ids(self) -> dict[str, int]:
        return {label.name: i for i, label in enumerate(self.labels)}


@dataclass(frozen=True)
class DatasetConfig:
    roboflow_export_dir: Path
    shelf_photos_dir: Path
    output_dir: Path
    test_source_split: str
    val_fraction: float
    seed: int


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_config(path: Path = CONFIG_DIR / "classes.yaml") -> ClassConfig:
    raw = _read_yaml(path)
    labels = tuple(ProductLabel(**entry) for entry in raw["labels"])
    names = [label.name for label in labels]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: label names must be unique")
    if raw["target_brand"] not in {label.brand for label in labels}:
        raise ValueError(f"{path}: target_brand '{raw['target_brand']}' is not the brand of any label")
    return ClassConfig(target_brand=raw["target_brand"], labels=labels)


def load_dataset_config(path: Path = CONFIG_DIR / "dataset.yaml") -> DatasetConfig:
    raw = _read_yaml(path)
    return DatasetConfig(
        roboflow_export_dir=REPO_ROOT / raw["roboflow_export_dir"],
        shelf_photos_dir=REPO_ROOT / raw["shelf_photos_dir"],
        output_dir=REPO_ROOT / raw["output_dir"],
        test_source_split=raw["test_source_split"],
        val_fraction=float(raw["val_fraction"]),
        seed=int(raw["seed"]),
    )
