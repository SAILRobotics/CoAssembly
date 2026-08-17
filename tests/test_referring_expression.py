from __future__ import annotations

import base64
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import referring_expression_test_babylon as study
from referring_expression_resolver import (
    Candidate,
    ReferringExpressionResolver,
    Resolution,
    match_box_to_candidate,
)


ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeResolver(ReferringExpressionResolver):
    name = "fake"
    model_name = "fake/test"

    def resolve(self, *, image_bytes, expression, candidates):
        self.received = (image_bytes, expression, candidates)
        candidate = candidates[0]
        return Resolution(
            predicted_part_file=candidate.part_file,
            predicted_mesh_name=candidate.mesh_name,
            bbox=candidate.bbox,
            confidence=0.91,
            mapping_score=0.88,
            latency_ms=12.5,
            resolver=self.name,
            model=self.model_name,
            raw_output="test",
        )


class ResolverTests(unittest.TestCase):
    def test_box_is_matched_to_overlapping_candidate(self):
        candidates = [
            Candidate("left.stl", "left", (0, 0, 20, 20)),
            Candidate("right.stl", "right", (80, 0, 100, 20)),
        ]
        selected, score = match_box_to_candidate((78, 0, 98, 20), candidates)
        self.assertEqual(selected.part_file, "right.stl")
        self.assertGreater(score, 0.7)

    def test_candidate_rejects_inverted_box(self):
        with self.assertRaises(ValueError):
            Candidate.from_payload({
                "part_file": "part.stl", "mesh_name": "mesh", "bbox": [5, 5, 1, 1]
            })


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.previous_resolver = study.RESOLVER
        self.resolver = FakeResolver()
        study.RESOLVER = self.resolver
        study.app.config.update(TESTING=True)
        self.client = study.app.test_client()
        self.part_file = study.part_files()[0]

    def tearDown(self):
        study.RESOLVER = self.previous_resolver

    def test_resolve_does_not_receive_ground_truth(self):
        image = "data:image/png;base64," + base64.b64encode(ONE_PIXEL_PNG).decode()
        response = self.client.post("/api/resolve", json={
            "expression": "the small gear on the left",
            "image": image,
            "candidates": [{
                "part_file": self.part_file,
                "mesh_name": "mesh-1",
                "bbox": [0, 0, 20, 20],
            }],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["predicted_part_file"], self.part_file)
        self.assertEqual(self.resolver.received[1], "the small gear on the left")

    def test_every_babylon_mapping_uses_a_real_study_part(self):
        mapping = re.search(
            r"const assemblyParts = \{(?P<body>.*?)\n  \};",
            study.INDEX_HTML,
            re.DOTALL,
        )
        self.assertIsNotNone(mapping)
        mapped_files = set(re.findall(r"'([^']+\.stl)'\s*:", mapping.group("body")))
        self.assertTrue(mapped_files)
        self.assertEqual(mapped_files - set(study.part_files()), set())

    def test_response_and_prediction_are_logged_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(study, "CSV_PATH", root / "responses.csv"),
                patch.object(study, "PREDICTIONS_CSV_PATH", root / "predictions.csv"),
                patch.object(study, "append_xlsx", return_value=False),
            ):
                response = self.client.post("/api/responses", json={
                    "participant": "P01",
                    "part_file": self.part_file,
                    "description": "the target part",
                    "presentation_number": 1,
                    "target_presentations": 3,
                    "action": "response",
                    "prediction": {
                        "predicted_part_file": self.part_file,
                        "predicted_mesh_name": "mesh-1",
                        "confidence": 0.91,
                        "mapping_score": 0.88,
                        "latency_ms": 12.5,
                        "resolver": "forged-client-value",
                        "model": "forged-client-value",
                        "raw_output": "test",
                    },
                })
                self.assertEqual(response.status_code, 200)
                self.assertIn("description", (root / "responses.csv").read_text())
                prediction_text = (root / "predictions.csv").read_text()
                self.assertIn("is_correct", prediction_text)
                self.assertIn("fake/test", prediction_text)
                self.assertNotIn("forged-client-value", prediction_text)


if __name__ == "__main__":
    unittest.main()
