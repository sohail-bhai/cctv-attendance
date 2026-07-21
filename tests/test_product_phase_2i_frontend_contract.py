from __future__ import annotations

import unittest
from pathlib import Path


class ProductPhase2IFrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.reports = (root / "frontend/src/pages/Reports.jsx").read_text(encoding="utf-8")
        cls.review = (root / "frontend/src/pages/ManualReview.jsx").read_text(encoding="utf-8")
        cls.consistency = (root / "frontend/src/utils/consistency.js").read_text(encoding="utf-8")
        cls.badge = (root / "frontend/src/components/StatusBadge.jsx").read_text(encoding="utf-8")
        cls.timetable = (root / "frontend/src/pages/Timetable.jsx").read_text(encoding="utf-8")
        cls.styles = (root / "frontend/src/styles.css").read_text(encoding="utf-8")

    def test_browser_requests_tracklet_authority_mode(self):
        self.assertIn("processing_mode: 'quality_aware_tracklet_authority_v1'", self.timetable)
        payload = self.timetable.split("const CONTROLLED_PROCESSING_PAYLOAD = {", 1)[1].split("};", 1)[0]
        self.assertNotIn("match_threshold", payload)
        self.assertNotIn("margin_threshold", payload)

    def test_report_and_review_expose_all_safe_statuses(self):
        for label in ("Unconfirmed", "Missing enrollment", "Absent", "Present", "Needs review"):
            self.assertIn(label, self.reports)
            self.assertIn(label, self.review)
        self.assertIn("Insufficient camera evidence; not proven absent", self.reports)
        self.assertIn("Multi-frame checkpoints", self.review)

    def test_unresolved_count_includes_uncertain_and_missing_rows(self):
        self.assertIn(
            "counts.review + counts.unconfirmed + counts.missingEnrollment + counts.unknown",
            self.consistency,
        )
        self.assertIn("if (mode === 'unconfirmed')", self.consistency)
        self.assertIn("if (mode === 'missing')", self.consistency)

    def test_status_badges_are_visually_distinct(self):
        self.assertIn("badge unconfirmed", self.badge)
        self.assertIn("badge missing-enrollment", self.badge)
        self.assertIn(".badge.unconfirmed", self.styles)
        self.assertIn(".badge.missing-enrollment", self.styles)


if __name__ == "__main__":
    unittest.main()
