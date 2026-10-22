import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from retail_shelf_analytics.config import ClassConfig, DatasetConfig, ProductLabel
from retail_shelf_analytics.data.roboflow_import import (
    ROBOFLOW_SIZE,
    assign_splits,
    collect_annotated_photos,
    export_photo_key,
    import_roboflow_export,
    raw_photo_key,
)

CLASSES = ClassConfig(
    target_brand="ramy",
    labels=(
        ProductLabel("ramy_plastique", "ramy", "bottle"),
        ProductLabel("rouiba_carton", "rouiba", "carton"),
    ),
)
EXIF_ORIENTATION = 274
ROTATED_90_CW = 6


def make_photo(path: Path, seed: int, orientation: int = 1) -> Image.Image:
    """Save a portrait phone-like photo and return it as it looks once upright."""
    rng = np.random.default_rng(seed)
    upright = Image.fromarray(rng.integers(0, 255, (1280, 720, 3), dtype=np.uint8)).resize((72, 128)).resize((720, 1280))
    stored = upright.rotate(90, expand=True) if orientation == ROTATED_90_CW else upright
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = orientation
    stored.save(path, exif=exif)
    return upright


def export_copy(upright: Image.Image, augmented: bool) -> Image.Image:
    stretched = upright.resize(ROBOFLOW_SIZE)
    return stretched.rotate(15) if augmented else stretched


class FakeExport:
    """A Roboflow-style export: 3 copies per train photo (one unaugmented), 1 copy per valid photo."""

    def __init__(self, root: Path) -> None:
        self.export_dir = root / "export"
        self.photos_dir = root / "photos"
        self.photos_dir.mkdir(parents=True)
        # Export class order deliberately differs from CLASSES.
        self.export_dir.mkdir()
        (self.export_dir / "data.yaml").write_text(yaml.safe_dump({"names": ["rouiba_carton", "ramy_plastique"]}))

    def add(self, raw_name: str, split: str, seed: int, orientation: int = 1) -> None:
        upright = make_photo(self.photos_dir / raw_name, seed, orientation)
        stem = Path(raw_name).stem.replace(" ", "-").replace(".", "_") + "_jpg"
        for split_part in ("images", "labels"):
            (self.export_dir / split / split_part).mkdir(parents=True, exist_ok=True)
        for copy_index in range(3 if split == "train" else 1):
            augmented = copy_index != 1 if split == "train" else False
            name = f"{stem}.rf.copy{copy_index}"
            export_copy(upright, augmented).save(self.export_dir / split / "images" / f"{name}.jpg")
            # The label file records which copy it belongs to, through its box width.
            width = 0.1 if not augmented else 0.9
            (self.export_dir / split / "labels" / f"{name}.txt").write_text(f"0 0.5 0.5 {width} 0.2\n1 0.2 0.2 0.1 0.1\n")


class RoboflowImportTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.export = FakeExport(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_export_and_raw_names_share_a_key(self) -> None:
        raw = Path("WhatsApp Image 2025-02-04 a 14.09.26_1439cedb.jpg")
        exported = Path("WhatsApp-Image-2025-02-04-a-14_09_26_1439cedb_jpg.rf.ce1a51db7c.jpg")
        self.assertEqual(raw_photo_key(raw), export_photo_key(exported))

    def test_picks_the_unaugmented_copy_even_for_exif_rotated_photos(self) -> None:
        self.export.add("WhatsApp Image 2025-02-04 a 14.09.26_aaaa.jpg", "train", seed=1)
        self.export.add("WhatsApp Image 2025-02-05 a 10.00.00_bbbb.jpg", "train", seed=2, orientation=ROTATED_90_CW)
        photos = collect_annotated_photos(self.export.export_dir, self.export.photos_dir)
        self.assertEqual(len(photos), 2)
        for photo in photos:
            self.assertIn(".rf.copy1", photo.label_file.name)

    def test_missing_original_photo_is_reported(self) -> None:
        self.export.add("WhatsApp Image 2025-02-04 a 14.09.26_aaaa.jpg", "train", seed=1)
        (self.export.photos_dir / "WhatsApp Image 2025-02-04 a 14.09.26_aaaa.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no original photo"):
            collect_annotated_photos(self.export.export_dir, self.export.photos_dir)

    def test_photos_of_one_store_stay_in_one_split(self) -> None:
        for i in range(6):
            self.export.add(f"pdv_pic_1000_{i}_98552.jpg", "train", seed=10 + i)
        for i in range(6):
            self.export.add(f"WhatsApp Image 2025-02-0{i} a 10.00.00_{i}.jpg", "train", seed=20 + i)
        self.export.add("pdv_pic_2000_0_98552.jpg", "valid", seed=30)
        photos = collect_annotated_photos(self.export.export_dir, self.export.photos_dir)
        splits = assign_splits(photos, test_source_split="valid", val_fraction=0.3, seed=0)
        store_splits = {split for photo, split in splits.items() if photo.group == "store-1000"}
        self.assertEqual(len(store_splits), 1)
        self.assertEqual([p.group for p, s in splits.items() if s == "test"], ["store-2000"])
        self.assertEqual(set(splits.values()), {"train", "val", "test"})

    def test_import_writes_upright_images_and_remapped_labels(self) -> None:
        self.export.add("WhatsApp Image 2025-02-05 a 10.00.00_bbbb.jpg", "valid", seed=2, orientation=ROTATED_90_CW)
        for i in range(4):
            self.export.add(f"WhatsApp Image 2025-02-0{i} a 11.00.00_{i}.jpg", "train", seed=40 + i)
        output_dir = self.root / "out"
        dataset = DatasetConfig(self.export.export_dir, self.export.photos_dir, output_dir, "valid", 0.25, 0)
        import_roboflow_export(dataset, CLASSES)

        test_image = output_dir / "images" / "test" / "whatsapp-image-2025-02-05-a-10-00-00-bbbb.jpg"
        self.assertEqual(Image.open(test_image).size, (720, 1280))
        test_labels = (output_dir / "labels" / "test" / "whatsapp-image-2025-02-05-a-10-00-00-bbbb.txt").read_text()
        # Export id 0 is rouiba_carton, which is id 1 in CLASSES, and the reverse.
        self.assertEqual(test_labels, "1 0.5 0.5 0.1 0.2\n0 0.2 0.2 0.1 0.1\n")
        data_yaml = yaml.safe_load((output_dir / "data.yaml").read_text())
        self.assertEqual(data_yaml["names"], {0: "ramy_plastique", 1: "rouiba_carton"})
        self.assertEqual(len((output_dir / "manifest.csv").read_text().splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
