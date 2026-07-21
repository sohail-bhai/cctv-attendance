import json
import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.face_attendance.forensic_review import (
    ForensicReviewError,
    ForensicReviewItem,
    audit_forensic_review_package,
    export_forensic_review_package,
    validate_forensic_approvals,
    write_reviewer_launcher,
)


ENROLLMENT_ACTIONS = {
    "keep": "Keep",
    "duplicate_of_another_image": "Duplicate of another image",
}
CCTV_ACTIONS = {
    "approve_for_candidate_embedding_version": "Approve",
    "reject_duplicate": "Reject duplicate",
    "reject_quality": "Reject quality",
}


class ForensicReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_a = self.root / "student a.jpg"
        self.source_b = self.root / "student b.jpg"
        image = np.full((80, 120, 3), 160, dtype=np.uint8)
        cv2.circle(image, (60, 38), 22, (210, 210, 210), -1)
        self.assertTrue(cv2.imwrite(str(self.source_a), image))
        self.assertTrue(cv2.imwrite(str(self.source_b), image + 5))

    def tearDown(self):
        self.temp.cleanup()

    def export(self, *, actions=None, kind="enrollment_audit", count=2):
        sources = [self.source_a, self.source_b][:count]
        items = [
            ForensicReviewItem(
                item_id=f"ITEM-{index}",
                evidence_source=source,
                public={"Expected_Roll": f"ROLL{index}", "Quality_Warnings": "review"},
                private={"secret_provenance": f"private-{index}"},
            )
            for index, source in enumerate(sources, start=1)
        ]
        return export_forensic_review_package(
            output_dir=self.root / f"package-{kind}-{count}",
            package_id=f"pkg-{kind}-{count}",
            package_kind=kind,
            title="Forensic Review",
            instructions="Review every item.",
            actions=actions or ENROLLMENT_ACTIONS,
            items=items,
        )

    @staticmethod
    def complete_csv(package, actions):
        frame = pd.read_csv(
            package.reviewer_dir / "approvals_template.csv", dtype=str, keep_default_na=False
        )
        frame["Review_Action"] = actions
        path = package.root.parent / f"approvals-{package.package_id}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def test_package_hashes_validate_and_private_provenance_is_not_public(self):
        package = self.export()
        audit = audit_forensic_review_package(package.root)
        self.assertTrue(audit["privacy_passed"])
        reviewer_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in package.reviewer_dir.rglob("*")
            if path.is_file() and path.suffix in {".html", ".json", ".csv", ".txt"}
        )
        self.assertNotIn("secret_provenance", reviewer_text)
        self.assertNotIn("private-1", reviewer_text)

    def test_static_reviewer_uses_native_offline_controls_and_valid_json_payload(self):
        package = self.export()
        html = (package.reviewer_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn('<select class="action"', html)
        self.assertIn("localStorage", html)
        self.assertNotIn("https://", html)
        self.assertIn("join('\\r\\n')", html)
        self.assertNotIn("join('\r\n')", html)
        self.assertIn("Math.max(1, Math.round", html)
        self.assertIn("card.dataset.itemId", html)
        self.assertIn("Boolean(config.duplicate_action)", html)
        self.assertIn("[hidden] { display:none!important; }", html)
        payload_match = re.search(
            r'<script type="application/json" id="package-data">(.*?)</script>', html, re.S
        )
        self.assertIsNotNone(payload_match)
        payload = json.loads(payload_match.group(1))
        self.assertEqual(payload["package_id"], package.package_id)
        self.assertEqual(payload["duplicate_action"], "duplicate_of_another_image")

    def test_complete_approval_csv_validates(self):
        package = self.export()
        approvals = self.complete_csv(package, ["keep", "keep"])
        result = validate_forensic_approvals(package.root, approvals, expected_kind="enrollment_audit")
        self.assertTrue(result.summary["complete"])
        self.assertEqual(result.summary["reviewed_items"], 2)

    def test_incomplete_approval_csv_fails_closed(self):
        package = self.export()
        approvals = self.complete_csv(package, ["keep", "keep"])
        frame = pd.read_csv(approvals, dtype=str, keep_default_na=False).iloc[:1]
        frame.to_csv(approvals, index=False)
        with self.assertRaisesRegex(ForensicReviewError, "item set mismatch"):
            validate_forensic_approvals(package.root, approvals)

    def test_enrollment_duplicate_requires_a_different_valid_item(self):
        package = self.export()
        approvals = self.complete_csv(package, ["duplicate_of_another_image", "keep"])
        with self.assertRaisesRegex(ForensicReviewError, "must reference"):
            validate_forensic_approvals(package.root, approvals)
        frame = pd.read_csv(approvals, dtype=str, keep_default_na=False)
        frame.loc[0, "Duplicate_Of_Item_ID"] = "ITEM-2"
        frame.to_csv(approvals, index=False)
        result = validate_forensic_approvals(package.root, approvals)
        self.assertTrue(result.summary["complete"])

    def test_cctv_reject_duplicate_does_not_require_duplicate_target(self):
        package = self.export(actions=CCTV_ACTIONS, kind="verified_cctv_enrollment")
        approvals = self.complete_csv(package, ["reject_duplicate", "reject_quality"])
        result = validate_forensic_approvals(
            package.root, approvals, expected_kind="verified_cctv_enrollment"
        )
        self.assertEqual(result.summary["action_counts"]["reject_duplicate"], 1)
        html = (package.reviewer_dir / "index.html").read_text(encoding="utf-8")
        payload = json.loads(
            re.search(
                r'<script type="application/json" id="package-data">(.*?)</script>', html, re.S
            ).group(1)
        )
        self.assertEqual(payload["duplicate_action"], "")

    def test_source_mutation_after_export_fails_approval_validation(self):
        package = self.export(count=1)
        approvals = self.complete_csv(package, ["keep"])
        self.source_a.write_bytes(self.source_a.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ForensicReviewError, "source integrity"):
            validate_forensic_approvals(package.root, approvals)

    def test_long_path_launcher_quotes_paths_and_does_not_open_browser(self):
        long_root = self.root / (("long folder " * 7) + "long folder")
        long_root.mkdir(parents=True)
        launcher = write_reviewer_launcher(
            launcher_path=long_root / "open reviewer.ps1",
            python_executable=Path(r"F:\project path\venv\Scripts\python.exe"),
            cli_script=Path(r"F:\project path\scripts\validate_tracklet_ground_truth.py"),
            command="open-enrollment-audit-review",
            review_package=long_root / "review package",
        )
        content = launcher.read_text(encoding="utf-8")
        self.assertIn("open-enrollment-audit-review", content)
        self.assertIn("--review-package", content)
        self.assertIn("'", content)
        self.assertNotIn("Start-Process", content)


if __name__ == "__main__":
    unittest.main()
