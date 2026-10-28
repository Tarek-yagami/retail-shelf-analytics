"""Rebuild the hand-annotated dataset at full resolution from the Roboflow export.

Roboflow stretched every photo to 640x640 and saved 3 augmented copies of each train photo.
Only one copy per photo is unaugmented; its normalized boxes also fit the full-resolution original.
"""

import csv
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit

from retail_shelf_analytics.config import ClassConfig, DatasetConfig

ROBOFLOW_SIZE = (640, 640)
# Mean absolute grey-level difference to the original: unaugmented copies score 2-5, rotated or sheared ones 35+.
MAX_COPY_DIFFERENCE = 15.0
STORE_PHOTO = re.compile(r"^pdv_pic_(\d+)_")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class AnnotatedPhoto:
    photo: Path
    label_file: Path
    group: str  # photos of the same store share a group and land in the same split

    @property
    def slug(self) -> str:
        return re.sub(r"[^0-9a-z]+", "-", self.photo.stem.lower()).strip("-")


def _key(name: str) -> str:
    # Roboflow keeps "à" in uploaded names ("2025-02-13 à 22.40") but writes "a" in export names.
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^0-9a-z]", "", ascii_name.lower())


def raw_photo_key(path: Path) -> str:
    return _key(path.stem)


def export_photo_key(path: Path) -> str:
    """Key of a Roboflow file such as 'WhatsApp-Image-..._jpg.rf.<hash>.jpg', matching raw_photo_key."""
    source_name = path.name.split(".rf.")[0]
    return _key(re.sub(r"_(jpe?g|png)$", "", source_name, flags=re.IGNORECASE))


def photo_group(path: Path) -> str:
    store = STORE_PHOTO.match(path.name)
    return f"store-{store.group(1)}" if store else f"photo-{raw_photo_key(path)}"


def load_upright(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path))


def _grey(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L").resize(ROBOFLOW_SIZE), dtype=np.float32)


def find_unaugmented_copy(copies: list[Path], photo: Path) -> Path:
    original = _grey(load_upright(photo))
    differences = {copy: float(np.abs(_grey(Image.open(copy)) - original).mean()) for copy in copies}
    best = min(differences, key=differences.get)
    if differences[best] > MAX_COPY_DIFFERENCE:
        raise ValueError(
            f"No unaugmented copy of {photo.name} in the export: the closest one differs by "
            f"{differences[best]:.1f} (limit {MAX_COPY_DIFFERENCE})."
        )
    return best


def collect_annotated_photos(export_dir: Path, photos_dir: Path) -> tuple[list[AnnotatedPhoto], list[str]]:
    """Return the annotated photos and the export names of those whose original is missing."""
    raw_photos = {raw_photo_key(p): p for p in photos_dir.iterdir() if p.is_file()}
    annotated, without_original = [], []
    for split_dir in sorted(d for d in export_dir.iterdir() if (d / "images").is_dir()):
        copies_by_key = defaultdict(list)
        for image in (split_dir / "images").iterdir():
            copies_by_key[export_photo_key(image)].append(image)
        for key, copies in sorted(copies_by_key.items()):
            if key not in raw_photos:
                without_original.append(copies[0].name.split(".rf.")[0])
                continue
            photo = raw_photos[key]
            copy = find_unaugmented_copy(copies, photo)
            annotated.append(
                AnnotatedPhoto(
                    photo=photo,
                    label_file=split_dir / "labels" / f"{copy.stem}.txt",
                    group=photo_group(photo),
                )
            )
    return annotated, without_original


def export_label_remap(export_dir: Path, classes: ClassConfig) -> dict[int, int]:
    """Map the export's class ids to configs/classes.yaml ids by label name."""
    export_names = yaml.safe_load((export_dir / "data.yaml").read_text(encoding="utf-8"))["names"]
    label_ids = classes.label_ids()
    unknown = sorted(set(export_names) - set(label_ids))
    if unknown:
        raise ValueError(f"Labels in the export are missing from configs/classes.yaml: {unknown}")
    return {i: label_ids[name] for i, name in enumerate(export_names)}


def assign_splits(
    photos: list[AnnotatedPhoto],
    test_photos: frozenset[str],
    val_fraction: float,
    seed: int,
) -> dict[AnnotatedPhoto, str]:
    test_keys = {raw_photo_key(Path(name)) for name in test_photos}
    splits = {p: "test" for p in photos if raw_photo_key(p.photo) in test_keys}
    if len(splits) != len(test_keys):
        found = {raw_photo_key(p.photo) for p in splits}
        missing = sorted(name for name in test_photos if raw_photo_key(Path(name)) not in found)
        raise ValueError(f"Test photos missing from the export: {missing}")
    rest = [p for p in photos if p not in splits]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(rest, groups=[p.group for p in rest]))
    splits.update({rest[i]: "train" for i in train_idx})
    splits.update({rest[i]: "val" for i in val_idx})
    return splits


def box_coordinates(values: list[str]) -> str:
    """Return 'x_center y_center width height' for a YOLO box, or the bounding box of a polygon's points."""
    if len(values) == 4:
        return " ".join(values)
    xs, ys = [float(v) for v in values[0::2]], [float(v) for v in values[1::2]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return f"{(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def write_dataset(
    splits: dict[AnnotatedPhoto, str],
    remap: dict[int, int],
    classes: ClassConfig,
    output_dir: Path,
) -> None:
    """Write upright full-resolution images, remapped labels, a manifest and a YOLO data.yaml."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True)
        (output_dir / "labels" / split).mkdir(parents=True)

    manifest = []
    for photo, split in sorted(splits.items(), key=lambda item: item[0].slug):
        rows = [line.split() for line in photo.label_file.read_text().splitlines() if line.strip()]
        labels = [f"{remap[int(row[0])]} {box_coordinates(row[1:])}" for row in rows]
        (output_dir / "labels" / split / f"{photo.slug}.txt").write_text("\n".join(labels) + "\n")
        image_name = f"{photo.slug}{photo.photo.suffix.lower()}"
        load_upright(photo.photo).save(output_dir / "images" / split / image_name, quality=95)
        manifest.append(
            {"image": image_name, "split": split, "group": photo.group, "boxes": len(labels), "source": photo.photo.name}
        )

    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    data_yaml = {
        "path": str(output_dir.resolve()),
        **{split: f"images/{split}" for split in SPLITS},
        "names": {i: label.name for i, label in enumerate(classes.labels)},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")


def import_roboflow_export(
    dataset: DatasetConfig,
    classes: ClassConfig,
) -> tuple[dict[AnnotatedPhoto, str], list[str]]:
    """Write the dataset and return each photo's split, plus the photos skipped for lack of an original."""
    remap = export_label_remap(dataset.roboflow_export_dir, classes)
    photos, without_original = collect_annotated_photos(dataset.roboflow_export_dir, dataset.shelf_photos_dir)
    splits = assign_splits(photos, dataset.test_photos, dataset.val_fraction, dataset.seed)
    write_dataset(splits, remap, classes, dataset.output_dir)
    return splits, without_original
