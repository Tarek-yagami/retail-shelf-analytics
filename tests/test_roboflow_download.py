import tempfile
import unittest
from pathlib import Path

import requests

from retail_shelf_analytics.config import RoboflowSource
from retail_shelf_analytics.data.roboflow_download import (
    RoboflowClient,
    RoboflowError,
    download_missing_originals,
)

API_KEY = "secret-key-123"


class FakeClient:
    """Answers like the Roboflow API: search results with ids and names, then image bytes."""

    def __init__(self, images: dict[str, str]) -> None:
        self._names = images  # image id -> uploaded name
        self.downloaded = []

    def list_images(self) -> list[dict]:
        return [{"id": image_id, "name": name} for image_id, name in self._names.items()]

    def original_url(self, image_id: str) -> str:
        return f"https://source.roboflow.com/ws/{image_id}/original.jpg"

    def download(self, step: str, url: str) -> bytes:
        self.downloaded.append(url)
        return b"jpeg bytes"


class RoboflowDownloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.export_dir = root / "export"
        self.photos_dir = root / "photos"
        (self.export_dir / "train" / "images").mkdir(parents=True)
        self.photos_dir.mkdir()
        for name in ("WhatsApp-Image-2025-02-13-a-22_40_29_0161e264_jpg.rf.1.jpg", "pdv_pic_1_2_98552_png.rf.2.jpg"):
            (self.export_dir / "train" / "images" / name).write_bytes(b"")
        (self.photos_dir / "pdv_pic_1_2_98552.png").write_bytes(b"already here")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_downloads_only_photos_missing_from_disk(self) -> None:
        client = FakeClient({"a1": "WhatsApp Image 2025-02-13 à 22.40.29_0161e264.jpg", "b2": "pdv_pic_1_2_98552.png"})
        saved = download_missing_originals(client, self.export_dir, self.photos_dir)
        self.assertEqual([p.name for p in saved], ["WhatsApp Image 2025-02-13 à 22.40.29_0161e264.jpg"])
        self.assertEqual(client.downloaded, ["https://source.roboflow.com/ws/a1/original.jpg"])
        self.assertEqual((self.photos_dir / "pdv_pic_1_2_98552.png").read_bytes(), b"already here")

    def test_photo_deleted_from_roboflow_is_left_out(self) -> None:
        client = FakeClient({"b2": "pdv_pic_1_2_98552.png"})
        self.assertEqual(download_missing_originals(client, self.export_dir, self.photos_dir), [])
        self.assertEqual(client.downloaded, [])

    def test_connection_error_does_not_reveal_the_api_key(self) -> None:
        client = RoboflowClient(API_KEY, RoboflowSource("ws", "project", 11))

        def time_out(*args, **kwargs) -> None:
            # requests puts the full URL, query string included, in its error messages.
            raise requests.ConnectTimeout(f"Max retries exceeded with url: /ws/project/11?api_key={API_KEY}")

        client._session.request = time_out
        with self.assertRaises(RoboflowError) as caught:
            client.list_images()
        self.assertNotIn(API_KEY, str(caught.exception))
        self.assertIn("Listing images failed", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
