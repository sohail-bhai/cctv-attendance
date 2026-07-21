from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.face_attendance.embedding_db import StudentEmbeddingDB
from src.face_attendance.mon_p3_source_ablation import (
    EXPECTED_BENCHMARK_ID,
    EXPECTED_BENCHMARK_ROWS,
    IMPLICATED_SOURCES,
    SourceAblationError,
    VectorizedTop3Index,
    _active_mask_for_config,
    _assert_score_reproduction,
    _comparison_metrics,
    _load_benchmark_features,
    _ordered_source_records,
    _pareto_frontier,
    _prediction_frame,
    _prediction_metrics,
    _required_metric,
    enumerate_source_subsets,
    rank_configurations,
    validate_implicated_sources,
    validate_record_alignment,
)
from src.face_attendance.recall_analysis import MATCH_THRESHOLD, MARGIN_THRESHOLD
from src.face_attendance.utils import normalize_embedding


def _unit(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _records() -> list[dict]:
    return [
        {"roll_no": "A", "image_path": "a0.png", "embedding": _unit(0)},
        {
            "roll_no": "A",
            "image_path": "a1.png",
            "embedding": normalize_embedding(_unit(0) + 0.2 * _unit(1)),
        },
        {
            "roll_no": "A",
            "image_path": "a2.png",
            "embedding": normalize_embedding(_unit(0) + 0.1 * _unit(2)),
        },
        {
            "roll_no": "A",
            "image_path": "a3.png",
            "embedding": normalize_embedding(_unit(0) + 0.3 * _unit(3)),
        },
        {"roll_no": "B", "image_path": "b0.png", "embedding": _unit(1)},
        {
            "roll_no": "B",
            "image_path": "b1.png",
            "embedding": normalize_embedding(_unit(1) + 0.1 * _unit(2)),
        },
        {"roll_no": "C", "image_path": "c0.png", "embedding": _unit(2)},
    ]


class RequiredMetricTests(unittest.TestCase):
    def test_zero_is_preserved(self) -> None:
        self.assertEqual(_required_metric({"unsafe": 0}, "unsafe"), 0)

    def test_missing_boolean_and_fractional_values_fail_closed(self) -> None:
        with self.assertRaises(SourceAblationError):
            _required_metric({}, "unsafe")
        with self.assertRaises(SourceAblationError):
            _required_metric({"unsafe": False}, "unsafe")
        with self.assertRaises(SourceAblationError):
            _required_metric({"unsafe": 0.5}, "unsafe")


class SubsetEnumerationTests(unittest.TestCase):
    def test_all_32_source_subsets_are_unique_and_complete(self) -> None:
        rows = enumerate_source_subsets()
        self.assertEqual(len(rows), 32)
        self.assertEqual({row["Mask"] for row in rows}, set(range(32)))
        self.assertEqual(len({row["Config_ID"] for row in rows}), 32)
        by_mask = {row["Mask"]: row for row in rows}
        self.assertEqual(by_mask[0]["Kept_Source_Count"], 0)
        self.assertEqual(by_mask[31]["Kept_Source_Count"], 5)
        self.assertEqual(
            set(by_mask[31]["Kept_Source_IDs"].split("|")),
            {source.source_id for source in IMPLICATED_SOURCES},
        )


class VectorizedTop3Tests(unittest.TestCase):
    def _assert_matches_reference(
        self, records: list[dict], queries: np.ndarray, active: np.ndarray | None = None
    ) -> None:
        index = VectorizedTop3Index.from_records(records)
        actual = index.score(
            queries,
            active_record_mask=active,
            match_threshold=MATCH_THRESHOLD,
            margin_threshold=MARGIN_THRESHOLD,
        )
        selected = records
        if active is not None:
            selected = [record for record, keep in zip(records, active) if bool(keep)]
        reference = StudentEmbeddingDB(selected)
        for row, query in zip(actual.to_dict("records"), queries):
            expected = reference.match(
                query,
                match_threshold=MATCH_THRESHOLD,
                margin_threshold=MARGIN_THRESHOLD,
                aggregate="top3",
            )
            self.assertEqual(bool(row["Accepted"]), expected.accepted)
            self.assertEqual(row["Best_Roll"], expected.roll_no)
            self.assertEqual(row["Second_Roll"], expected.second_roll_no or "")
            self.assertAlmostEqual(row["Best_Score"], expected.best_score, places=6)
            self.assertAlmostEqual(row["Second_Score"], expected.second_score, places=6)
            self.assertAlmostEqual(row["Margin"], expected.margin, places=6)
            self.assertEqual(row["Reason"], expected.reason)

    def test_vectorized_scores_match_operational_top3_matcher(self) -> None:
        queries = np.stack(
            [
                _unit(0),
                _unit(1),
                normalize_embedding(_unit(0) + _unit(1)),
                _unit(3),
            ]
        )
        self._assert_matches_reference(_records(), queries)

    def test_active_record_mask_matches_filtered_database(self) -> None:
        records = _records()
        active = np.asarray([True, False, True, False, True, True, True])
        queries = np.stack([_unit(0), _unit(1), _unit(2)])
        self._assert_matches_reference(records, queries, active)

    def test_ties_follow_sorted_roll_order(self) -> None:
        records = [
            {"roll_no": "B", "image_path": "b.png", "embedding": _unit(0)},
            {"roll_no": "A", "image_path": "a.png", "embedding": _unit(0)},
        ]
        row = VectorizedTop3Index.from_records(records).score(_unit(0)).iloc[0]
        self.assertEqual(row["Best_Roll"], "A")
        self.assertEqual(row["Second_Roll"], "B")
        self.assertFalse(bool(row["Accepted"]))
        self.assertEqual(row["Reason"], "margin_too_small")

    def test_bad_query_and_mask_shapes_fail_closed(self) -> None:
        index = VectorizedTop3Index.from_records(_records())
        with self.assertRaises(SourceAblationError):
            index.score(np.zeros((2, 64), dtype=np.float32))
        with self.assertRaises(SourceAblationError):
            index.score(_unit(0), active_record_mask=np.ones(2, dtype=bool))


class SourceInventoryTests(unittest.TestCase):
    def _inventory(self) -> pd.DataFrame:
        rows = []
        for order, spec in enumerate(IMPLICATED_SOURCES):
            rows.append(
                {
                    "Record_Order": str(order),
                    "Source_ID": spec.source_id,
                    "Canonical_Roll": spec.canonical_roll,
                    "Source_Kind": spec.source_kind,
                    "Source_Session": spec.source_session,
                    "Image_Path": "models/versions/example/" + spec.expected_image_suffix,
                }
            )
        return pd.DataFrame(rows)

    def test_exact_implicated_source_inventory_passes(self) -> None:
        result = validate_implicated_sources(self._inventory())
        self.assertEqual(len(result), 5)
        self.assertEqual(result["Bit_Index"].tolist(), list(range(5)))

    def test_changed_identity_kind_session_or_path_fails_closed(self) -> None:
        for column, value in (
            ("Canonical_Roll", "WRONG"),
            ("Source_Kind", "wrong_kind"),
            ("Source_Session", "wrong_session"),
            ("Image_Path", "wrong.png"),
        ):
            frame = self._inventory()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(SourceAblationError):
                validate_implicated_sources(frame)

    def test_source_record_order_must_be_complete_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source_records.csv"
            frame = self._inventory()
            frame.to_csv(path, index=False)
            ordered = _ordered_source_records(path)
            self.assertEqual(ordered["Record_Order"].astype(int).tolist(), list(range(5)))
            frame.loc[4, "Record_Order"] = "3"
            frame.to_csv(path, index=False)
            with self.assertRaises(SourceAblationError):
                _ordered_source_records(path)

    def test_embedding_and_source_records_must_align_one_to_one(self) -> None:
        sources = self._inventory()
        records = [
            {
                "roll_no": row["Canonical_Roll"],
                "image_path": row["Image_Path"],
                "embedding": _unit(index),
            }
            for index, row in enumerate(sources.to_dict("records"))
        ]
        validate_record_alignment(sources, records)
        records[0] = {**records[0], "roll_no": "WRONG"}
        with self.assertRaises(SourceAblationError):
            validate_record_alignment(sources, records)


class ActiveMaskTests(unittest.TestCase):
    def _source_frame(self) -> pd.DataFrame:
        rows = []
        for order, spec in enumerate(IMPLICATED_SOURCES):
            rows.append(
                {
                    "Source_ID": spec.source_id,
                    "Source_Kind": spec.source_kind,
                    "Source_Session": spec.source_session,
                }
            )
        rows.extend(
            [
                {
                    "Source_ID": "OTHER-P1",
                    "Source_Kind": "human_approved_cctv_crop",
                    "Source_Session": "2026-06-30__B51__P1__CVO",
                },
                {
                    "Source_ID": "OTHER-P2",
                    "Source_Kind": "same_track_recovered_cctv_medoid",
                    "Source_Session": "2026-06-30__B51__P2__CVO",
                },
                {
                    "Source_ID": "ENROLLMENT",
                    "Source_Kind": "cleaned_enrollment",
                    "Source_Session": "",
                },
            ]
        )
        return pd.DataFrame(rows)

    def test_full_composition_only_applies_requested_implicated_subset(self) -> None:
        frame = self._source_frame()
        kept = {source.source_id for source in IMPLICATED_SOURCES}
        self.assertTrue(
            _active_mask_for_config(
                frame, kept_source_ids=kept, evaluation_mode="full_composition"
            ).all()
        )
        mask = _active_mask_for_config(
            frame, kept_source_ids=set(), evaluation_mode="full_composition"
        )
        self.assertEqual(mask.tolist(), [False] * 5 + [True, True, True])

    def test_tue_p1_mode_excludes_all_same_session_p1_cctv_sources(self) -> None:
        frame = self._source_frame()
        kept = {source.source_id for source in IMPLICATED_SOURCES}
        mask = _active_mask_for_config(
            frame, kept_source_ids=kept, evaluation_mode="tue_p1_leakage_safe"
        )
        self.assertEqual(mask.tolist(), [False, False, False, True, True, False, True, True])

    def test_tue_p2_mode_excludes_all_same_session_p2_recovery_sources(self) -> None:
        frame = self._source_frame()
        kept = {source.source_id for source in IMPLICATED_SOURCES}
        mask = _active_mask_for_config(
            frame, kept_source_ids=kept, evaluation_mode="tue_p2_leakage_safe"
        )
        self.assertEqual(mask.tolist(), [True, True, True, False, False, True, False, True])

    def test_unknown_evaluation_mode_fails_closed(self) -> None:
        with self.assertRaises(SourceAblationError):
            _active_mask_for_config(
                self._source_frame(), kept_source_ids=set(), evaluation_mode="wrong"
            )


class PredictionAndComparisonTests(unittest.TestCase):
    def _metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Model_ID": "recorded",
                    "Row_ID": "R1",
                    "Actual_Class": "known_student",
                    "Actual_Roll": "A",
                    "Accepted": "False",
                    "Margin": "0.0",
                    "Accepted_Roll": "",
                    "Correct_Accepted": "False",
                    "Evidence_Mode": "old",
                },
                {
                    "Model_ID": "recorded",
                    "Row_ID": "R2",
                    "Actual_Class": "not_in_mapping",
                    "Actual_Roll": "",
                    "Accepted": "False",
                    "Margin": "0.0",
                    "Accepted_Roll": "",
                    "Correct_Accepted": "False",
                    "Evidence_Mode": "old",
                },
            ]
        )

    def _scores(self, accept_outsider: bool = True) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Row_ID": "R1",
                    "Accepted": True,
                    "Best_Roll": "A",
                    "Best_Score": 0.7,
                    "Second_Roll": "B",
                    "Second_Score": 0.5,
                    "Margin": 0.2,
                    "Reason": "accepted",
                },
                {
                    "Row_ID": "R2",
                    "Accepted": accept_outsider,
                    "Best_Roll": "B",
                    "Best_Score": 0.7,
                    "Second_Roll": "A",
                    "Second_Score": 0.5,
                    "Margin": 0.2,
                    "Reason": "accepted" if accept_outsider else "score_below_threshold",
                },
            ]
        )

    def test_recorded_model_columns_are_safely_replaced(self) -> None:
        prediction = _prediction_frame(
            self._metadata(),
            self._scores(),
            id_column="Row_ID",
            model_id="candidate",
            evidence_mode="test",
            promotion_evidence_eligible=False,
        )
        self.assertEqual(prediction["Model_ID"].unique().tolist(), ["candidate"])
        self.assertTrue(bool(prediction.loc[0, "Correct_Accepted"]))
        self.assertTrue(bool(prediction.loc[1, "Outsider_Absorption"]))
        self.assertEqual(_prediction_metrics(prediction)["unsafe_confirmed_acceptances"], 1)

    def test_comparison_counts_losses_recoveries_and_new_unsafe_accepts(self) -> None:
        production = _prediction_frame(
            self._metadata(),
            self._scores(accept_outsider=False),
            id_column="Row_ID",
            model_id="production",
            evidence_mode="test",
            promotion_evidence_eligible=True,
        )
        candidate = _prediction_frame(
            self._metadata(),
            self._scores(accept_outsider=True),
            id_column="Row_ID",
            model_id="candidate",
            evidence_mode="test",
            promotion_evidence_eligible=False,
        )
        metrics, joined = _comparison_metrics(
            production, candidate, id_column="Row_ID"
        )
        self.assertEqual(metrics["new_outsider_absorptions"], 1)
        self.assertEqual(metrics["lost_correct_production_accepts"], 0)
        self.assertEqual(len(joined), 2)


class ReproductionTests(unittest.TestCase):
    def _scored(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ID": "R1",
                    "Accepted": False,
                    "Best_Roll": "A",
                    "Best_Score": 0.5,
                    "Second_Roll": "B",
                    "Second_Score": 0.45,
                    "Margin": 0.05,
                    "Reason": "margin_too_small",
                }
            ]
        )

    def _recorded(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ID": "R1",
                    "Recorded_Accepted": "False",
                    "Recorded_Best_Roll": "A",
                    "Recorded_Best_Score": "0.5",
                    "Recorded_Second_Roll": "B",
                    "Recorded_Second_Score": "0.45",
                    "Recorded_Margin": "0.05",
                    "Recorded_Reason": "margin_too_small",
                }
            ]
        )

    def test_exact_reproduction_passes_and_score_drift_fails(self) -> None:
        _assert_score_reproduction(
            scored=self._scored(),
            recorded=self._recorded(),
            id_column="ID",
            recorded_prefix="Recorded_",
            label="test",
        )
        scored = self._scored()
        scored.loc[0, "Margin"] = 0.051
        with self.assertRaises(SourceAblationError):
            _assert_score_reproduction(
                scored=scored,
                recorded=self._recorded(),
                id_column="ID",
                recorded_prefix="Recorded_",
                label="test",
            )


class BenchmarkCacheTests(unittest.TestCase):
    def test_feature_cache_uses_generic_pickle_payload_and_validates_120_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "benchmark_id": EXPECTED_BENCHMARK_ID,
                "feature_rows": EXPECTED_BENCHMARK_ROWS,
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            payload = {
                "benchmark_id": EXPECTED_BENCHMARK_ID,
                "features": [
                    {
                        "benchmark_row_id": f"B{index:03d}",
                        "embedding": _unit(index % 128),
                    }
                    for index in range(EXPECTED_BENCHMARK_ROWS)
                ],
            }
            with (root / "cache.pkl").open("wb") as handle:
                pickle.dump(payload, handle)
            ids, features = _load_benchmark_features(
                root / "cache.pkl", root / "manifest.json"
            )
            self.assertEqual(len(ids), EXPECTED_BENCHMARK_ROWS)
            self.assertEqual(features.shape, (EXPECTED_BENCHMARK_ROWS, 128))


class RankingTests(unittest.TestCase):
    def _row(self, config: str, mask: int, hard: bool, recoveries: int, kept: int) -> dict:
        return {
            "Config_ID": config,
            "Mask": mask,
            "Hard_Gates_Passed": hard,
            "MON_P3_Preserved_Prior_Candidate_Recoveries": recoveries,
            "MON_P3_Recovered_Correct_vs_Production": recoveries,
            "Benchmark_Safe_Recovered_Correct_vs_Production": recoveries,
            "Benchmark_Safe_Correct_Accepted": 27 + recoveries,
            "Benchmark_Descriptive_Correct_Accepted": 30 + recoveries,
            "Kept_Source_Count": kept,
        }

    def test_ranking_prioritizes_hard_gates_then_recovery_then_fewer_sources(self) -> None:
        frame = pd.DataFrame(
            [
                self._row("unsafe", 0, False, 4, 0),
                self._row("safe-more", 2, True, 3, 3),
                self._row("safe-fewer", 1, True, 3, 2),
            ]
        )
        ranked = rank_configurations(frame)
        self.assertEqual(ranked["Config_ID"].tolist(), ["safe-fewer", "safe-more", "unsafe"])
        frontier = _pareto_frontier(ranked[ranked["Hard_Gates_Passed"]])
        self.assertEqual(frontier["Config_ID"].tolist(), ["safe-fewer"])

    def test_descriptive_same_session_gain_cannot_outrank_parsimony(self) -> None:
        fewer = self._row("safe-fewer", 16, True, 4, 1)
        fewer["Benchmark_Descriptive_Correct_Accepted"] = 32
        more = self._row("safe-more", 19, True, 4, 3)
        more["Benchmark_Descriptive_Correct_Accepted"] = 34
        ranked = rank_configurations(pd.DataFrame([more, fewer]))
        self.assertEqual(ranked.iloc[0]["Config_ID"], "safe-fewer")

    def test_descriptive_same_session_gain_cannot_break_equal_safety_tie(self) -> None:
        lower_mask = self._row("lower-mask", 1, True, 4, 1)
        lower_mask["Benchmark_Descriptive_Correct_Accepted"] = 30
        higher_mask = self._row("higher-mask", 2, True, 4, 1)
        higher_mask["Benchmark_Descriptive_Correct_Accepted"] = 40
        ranked = rank_configurations(pd.DataFrame([higher_mask, lower_mask]))
        self.assertEqual(ranked.iloc[0]["Config_ID"], "lower-mask")


if __name__ == "__main__":
    unittest.main()
