import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProductPhase2JFrontendContractTests(unittest.TestCase):
    def test_reports_exposes_official_candidate_and_equivalent_revision_state(self):
        text = (ROOT / "frontend" / "src" / "pages" / "Reports.jsx").read_text(encoding="utf-8")
        self.assertIn("Official reviewed revision", text)
        self.assertIn("Latest automatic rerun archived", text)
        self.assertIn("pending_candidate_revision", text)
        self.assertIn("latest_candidate_matches_official", text)
        self.assertIn("No repeated review was required", text)
        self.assertIn("Mixed track rejected", text)
        self.assertIn("Strict:", text)
        self.assertIn("camera-unconfirmed case(s)", text)

    def test_manual_review_preserves_reviewed_authority_context(self):
        text = (ROOT / "frontend" / "src" / "pages" / "ManualReview.jsx").read_text(encoding="utf-8")
        self.assertIn("Automatic rerun archived without replacing this report", text)
        self.assertIn("Latest identical-source rerun matched this reviewed report", text)
        self.assertIn("Multi-frame checkpoints (reviewed)", text)
        self.assertIn("Mixed track rejected", text)
        self.assertIn("use verified classroom information", text)

    def test_backend_commits_reprocess_as_revision_candidate(self):
        text = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("preserve_official_or_commit_candidate", text)
        self.assertIn("apply_exact_review_carry_forward", text)
        self.assertIn("latest_candidate_matches_official", text)
        self.assertIn("Exact-source rerun matched the reviewed official report", text)


if __name__ == "__main__":
    unittest.main()
