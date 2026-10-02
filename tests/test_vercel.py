import json
import base64
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import tomllib

import app


BASE_DIR = Path(__file__).resolve().parents[1]


class VercelBundleTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_homepage_uses_vercel_cdn_assets_and_shows_samples(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        for expected in (
            b"/static/style.css",
            b"/static/app.js",
            b"/static/sample1.jpg",
            b"/static/sample2.jpg",
            b"/static/sample3.jpg",
        ):
            self.assertIn(expected, response.data)
        self.assertIsNone(app.app.static_folder)

    def test_upload_limit_fits_under_vercel_function_body_limit(self):
        self.assertEqual(app.MAX_IMAGE_BYTES, 3 * 1024 * 1024)
        self.assertLessEqual(app.app.config["MAX_CONTENT_LENGTH"], 4.5 * 1024 * 1024)

    def test_vercel_configuration_includes_runtime_models_and_templates(self):
        configuration = json.loads((BASE_DIR / "vercel.json").read_text("utf-8"))

        self.assertEqual(configuration["functions"]["app.py"]["maxDuration"], 300)
        self.assertIn("models/", configuration["functions"]["app.py"]["includeFiles"])
        self.assertIn(
            "templates/", configuration["functions"]["app.py"]["includeFiles"]
        )
        for sample_name in ("sample1.jpg", "sample2.jpg", "sample3.jpg"):
            self.assertTrue((BASE_DIR / "public" / "static" / sample_name).is_file())
        self.assertEqual(app.DETECTOR_PATH.name, "camtsd1_8n_int8.onnx")
        self.assertEqual(app.CLASSIFIER_PATH.name, "camtsr.onnx")
        self.assertTrue(app.DETECTOR_PATH.is_file())
        self.assertTrue(app.CLASSIFIER_PATH.is_file())

    def test_onnxruntime_replaces_the_large_torch_stack(self):
        project = tomllib.loads((BASE_DIR / "pyproject.toml").read_text("utf-8"))
        lock = tomllib.loads((BASE_DIR / "uv.lock").read_text("utf-8"))
        locked_packages = {
            package["name"]: package for package in lock["package"]
        }

        self.assertEqual(project["project"]["dependencies"][-1], "onnxruntime==1.30.0")
        self.assertIn("onnxruntime", locked_packages)
        self.assertNotIn("torch", locked_packages)
        self.assertNotIn("torchvision", locked_packages)
        self.assertNotIn("ultralytics", locked_packages)

    def test_sample_images_run_detection_and_recognition(self):
        expected_labels = {
            "sample1.jpg": "SLOW DOWN",
            "sample2.jpg": "CHILDREN CROSSING",
            "sample3.jpg": "PRIORITY ROAD",
        }

        for sample_name, expected_label in expected_labels.items():
            with self.subTest(sample=sample_name):
                sample_path = BASE_DIR / "public" / "static" / sample_name
                with sample_path.open("rb") as image_file:
                    response = self.client.post(
                        "/api/detect-image",
                        data={"image": (image_file, sample_name)},
                        content_type="multipart/form-data",
                    )

                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
                detections = response.get_json()["detections"]
                self.assertTrue(detections)
                self.assertTrue(
                    any(
                        detection["class_name"] == expected_label
                        and detection["recognition_confidence"] > 0.5
                        for detection in detections
                    )
                )

    def test_video_frames_do_not_load_the_recognizer(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        success, encoded_frame = cv2.imencode(".jpg", frame)
        self.assertTrue(success)
        frame_url = "data:image/jpeg;base64," + base64.b64encode(
            encoded_frame.tobytes()
        ).decode("ascii")

        with patch.object(app, "get_classifier", side_effect=AssertionError):
            response = self.client.post("/api/detect-frame", json={"image": frame_url})

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
