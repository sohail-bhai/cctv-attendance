from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.face_attendance.product_phase_2k_capture_intake as intake
from src.face_attendance.product_phase_2k_capture_intake import (
    CONFIRMATION_TOKEN,
    CaptureIntakeError,
    CaptureRequest,
    SourceSpec,
    build_source_specs,
    dry_run,
    materialize_capture_package,
    materialize_tooling_validation,
    parse_binding_values,
    plan_capture,
    read_video_header_metadata,
    verify_capture_package,
    verify_tooling_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def fake_metadata(_: Path) -> dict:
    return {
        "container": "mp4",
        "width": 640,
        "height": 360,
        "fps": 25.0,
        "frame_count": 500,
        "estimated_duration_seconds": 20.0,
        "metadata_method": "fixture_header_reader_no_decode",
        "metadata_confidence": "header_reported_complete",
        "frames_decoded": False,
    }


class CaptureIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dependency = intake._phase_2k_b_dependency(ROOT)
        cls.registry = intake._known_source_registry(cls.dependency)
        cls.baseline_hashes = {
            relative: intake.sha256_file(ROOT / relative)
            for relative in intake.PROTECTED_EXPECTED_HASHES
        }

    @classmethod
    def tearDownClass(cls) -> None:
        after = {
            relative: intake.sha256_file(ROOT / relative)
            for relative in intake.PROTECTED_EXPECTED_HASHES
        }
        if after != cls.baseline_hashes:
            raise AssertionError("Protected operational hashes changed during capture-intake tests")

    def _fixture(self, temp: str, *, suffix: str = ".mp4") -> tuple[Path, tuple[SourceSpec, ...]]:
        root = Path(temp) / "new_session_source"
        sources = []
        order = 0
        for checkpoint, camera in intake.EXPECTED_BINDINGS:
            path = root / checkpoint / f"{camera}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"safe-container-fixture-{checkpoint}-{camera}-{order}".encode("ascii"))
            sources.append(
                SourceSpec(
                    checkpoint,
                    camera,
                    path,
                    f"2026-08-03T10:{order:02d}:00+05:30",
                )
            )
            order += 1
        return root, tuple(sources)

    def _request(self, temp: str, **changes) -> CaptureRequest:
        _, sources = self._fixture(temp)
        values = {
            "repo_root": ROOT,
            "output_root": Path(temp) / "packages",
            "session_date": "2026-08-03",
            "section": "B51",
            "period": "P4",
            "subject": "CVO",
            "room": "B51",
            "sources": sources,
            "confirmation_token": CONFIRMATION_TOKEN,
        }
        values.update(changes)
        return CaptureRequest(**values)

    def _plan(self, request: CaptureRequest, **kwargs):
        return plan_capture(
            request,
            metadata_reader=kwargs.pop("metadata_reader", fake_metadata),
            dependency_loader=kwargs.pop("dependency_loader", lambda _: self.dependency),
            registry_loader=kwargs.pop("registry_loader", lambda _: self.registry),
            **kwargs,
        )

    @staticmethod
    def _rehash_manifest(package: Path, relative: str) -> None:
        manifest_path = package / "immutable_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = package / relative
        manifest["files"][relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        manifest_path.write_bytes(intake.canonical_json_bytes(manifest))

    def test_valid_dry_run_is_no_write(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            plan = self._plan(request)
            result = dry_run(plan)
            self.assertTrue(result["valid"])
            self.assertEqual(result["files_copied"], 0)
            self.assertFalse(request.output_root.exists())
            self.assertFalse(result["frames_decoded"])
            self.assertFalse(result["recognition_ran"])

    def test_valid_package_materializes_and_verifies(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))
            output, reused = materialize_capture_package(plan)
            self.assertFalse(reused)
            result = verify_capture_package(ROOT, output)
            self.assertTrue(result["valid"])
            self.assertEqual(result["source_count"], 10)
            self.assertEqual(result["overall_outcome"], "new_unique_source")

    def test_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            request = CaptureRequest(**{**request.__dict__, "sources": request.sources[:-2]})
            with self.assertRaisesRegex(CaptureIntakeError, "ten source|Exactly ten") as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "source_count_invalid")

    def test_missing_camera_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            replacement = SourceSpec("CP1", "side", request.sources[1].path, request.sources[1].capture_timestamp)
            request = CaptureRequest(**{**request.__dict__, "sources": (replacement,) + request.sources[1:]})
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "checkpoint_binding_missing")

    def test_extra_source_file_fails_discovery(self):
        with tempfile.TemporaryDirectory() as temp:
            source_root, _ = self._fixture(temp)
            (source_root / "notes.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(CaptureIntakeError) as caught:
                intake.discover_source_bindings(source_root)
            self.assertEqual(caught.exception.reason_code, "checkpoint_binding_extra")

    def test_duplicate_binding_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            request = CaptureRequest(
                **{**request.__dict__, "sources": (request.sources[0], request.sources[0]) + request.sources[2:]}
            )
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "checkpoint_binding_duplicate")

    def test_duplicate_bytes_inside_package_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            request.sources[1].path.write_bytes(request.sources[0].path.read_bytes())
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "duplicate_inside_package")
            self.assertEqual(caught.exception.outcome, "duplicate_inside_package")

    def test_known_contamination_hash_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            fixture_hash = hashlib.sha256(request.sources[0].path.read_bytes()).hexdigest()
            registry = json.loads(json.dumps(self.registry))
            registry["entries"].append(
                {
                    "sha256": fixture_hash,
                    "session_id": "historical",
                    "historical_relative_path": "historical/back.mp4",
                }
            )
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request, registry_loader=lambda _: registry)
            self.assertEqual(caught.exception.reason_code, "known_contaminated_source")
            self.assertEqual(caught.exception.outcome, "known_contaminated_source")

    def test_historical_duplicate_session_id_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp, session_date="2026-06-22", period="P4")
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "duplicate_session_id")

    def test_duplicate_session_id_in_other_package_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            other = request.output_root / "capture-package-existing"
            other.mkdir(parents=True)
            session = {"session_id": "2026-08-03__B51__P4__CVO"}
            (other / "session_metadata.json").write_bytes(intake.canonical_json_bytes(session))
            path = other / "session_metadata.json"
            manifest = {
                "schema_version": 1,
                "tool_policy_version": intake.TOOL_POLICY_VERSION,
                "files": {
                    "session_metadata.json": {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }
                },
            }
            (other / "immutable_manifest.json").write_bytes(intake.canonical_json_bytes(manifest))
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "duplicate_session_id")

    def test_invalid_session_metadata_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(self._request(temp, room=""))
            self.assertEqual(caught.exception.reason_code, "session_metadata_invalid")

    def test_invalid_capture_timestamp_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            bad = SourceSpec("CP1", "back", request.sources[0].path, "2026-08-03T10:00:00")
            request = CaptureRequest(**{**request.__dict__, "sources": (bad,) + request.sources[1:]})
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "capture_timestamp_invalid")

    def test_missing_capture_timestamp_binding_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            source_root, _ = self._fixture(temp)
            timestamps = [f"{cp}/{camera}=2026-08-03T10:00:00+05:30" for cp, camera in intake.EXPECTED_BINDINGS[:-1]]
            with self.assertRaises(CaptureIntakeError) as caught:
                build_source_specs(source_dir=source_root, source_values=[], capture_timestamp_values=timestamps)
            self.assertEqual(caught.exception.reason_code, "capture_timestamp_invalid")

    def test_source_changed_during_copy_fails_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))

            def changing_copy(source, target):
                shutil.copyfile(source, target)
                Path(source).write_bytes(Path(source).read_bytes() + b"changed")

            with self.assertRaises(CaptureIntakeError) as caught:
                materialize_capture_package(plan, copy_function=changing_copy)
            self.assertEqual(caught.exception.reason_code, "source_changed_during_copy")
            self.assertEqual(list(plan.output_root.glob(".*.staging-*")), [])

    def test_source_copy_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))

            def corrupt_copy(source, target):
                Path(target).write_bytes(Path(source).read_bytes() + b"corrupt")

            with self.assertRaises(CaptureIntakeError) as caught:
                materialize_capture_package(plan, copy_function=corrupt_copy)
            self.assertEqual(caught.exception.reason_code, "source_copy_mismatch")

    def test_copy_exception_cleans_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))

            def failed_copy(_source, _target):
                raise OSError("fixture copy failure")

            with self.assertRaises(OSError):
                materialize_capture_package(plan, copy_function=failed_copy)
            self.assertEqual(list(plan.output_root.glob(".*.staging-*")), [])

    def test_idempotent_reuse_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))
            output, reused = materialize_capture_package(plan)
            before = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}
            output_again, reused_again = materialize_capture_package(plan)
            after = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}
            self.assertFalse(reused)
            self.assertTrue(reused_again)
            self.assertEqual(output_again, output)
            self.assertEqual(after, before)

    def test_tampered_video_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))
            output, _ = materialize_capture_package(plan)
            (output / "videos" / "CP1" / "back.mp4").write_bytes(b"tampered")
            with self.assertRaises(CaptureIntakeError) as caught:
                verify_capture_package(ROOT, output)
            self.assertEqual(caught.exception.reason_code, "package_tampered")

    def test_existing_tampered_target_is_deterministic_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))
            output, _ = materialize_capture_package(plan)
            (output / "package_metadata.json").write_bytes(b"{}\n")
            with self.assertRaises(CaptureIntakeError) as caught:
                materialize_capture_package(plan)
            self.assertEqual(caught.exception.reason_code, "deterministic_id_collision")

    def test_changed_embedding_hash_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            changed = dict(intake.PROTECTED_EXPECTED_HASHES)
            changed["models/student_embeddings.pkl"] = "0" * 64
            with mock.patch.object(intake, "PROTECTED_EXPECTED_HASHES", changed):
                with self.assertRaises(CaptureIntakeError) as caught:
                    self._plan(self._request(temp))
            self.assertEqual(caught.exception.reason_code, "production_policy_changed")

    def test_changed_threshold_in_package_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))
            output, _ = materialize_capture_package(plan)
            path = output / "production_policy_freeze.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["fixed_production_configuration"]["match_threshold"] = 0.49
            path.write_bytes(intake.canonical_json_bytes(payload))
            self._rehash_manifest(output, "production_policy_freeze.json")
            with self.assertRaises(CaptureIntakeError) as caught:
                verify_capture_package(ROOT, output)
            self.assertEqual(caught.exception.reason_code, "production_policy_changed")

    def test_noncanonical_input_order_is_canonicalized(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            request = CaptureRequest(**{**request.__dict__, "sources": tuple(reversed(request.sources))})
            plan = self._plan(request)
            self.assertEqual([(row["checkpoint_id"], row["camera_id"]) for row in plan.sources], list(intake.EXPECTED_BINDINGS))
            self.assertEqual([row["canonical_order"] for row in plan.sources], list(range(10)))

    def test_windows_binding_paths_parse_without_drive_damage(self):
        parsed = parse_binding_values([r"CP1\back=C:\Capture Folder\clip.mp4"], value_label="source")
        self.assertEqual(parsed[("CP1", "back")], r"C:\Capture Folder\clip.mp4")

    def test_historical_prepared_path_fails_even_with_new_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            historical = Path(temp) / "cctv_videos" / "prepared_slots" / "new" / "back.mp4"
            historical.parent.mkdir(parents=True)
            historical.write_bytes(b"new-but-historical-path")
            replacement = SourceSpec("CP1", "back", historical, request.sources[0].capture_timestamp)
            request = CaptureRequest(**{**request.__dict__, "sources": (replacement,) + request.sources[1:]})
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "historical_prepared_path")

    def test_metadata_reader_failure_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            def failed_metadata(_):
                raise CaptureIntakeError("container_metadata_unavailable")

            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(self._request(temp), metadata_reader=failed_metadata)
            self.assertEqual(caught.exception.reason_code, "container_metadata_unavailable")

    def test_incomplete_header_metadata_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            incomplete = fake_metadata(Path("fixture"))
            incomplete["width"] = 0
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(self._request(temp), metadata_reader=lambda _: incomplete)
            self.assertEqual(caught.exception.reason_code, "container_metadata_unavailable")

    def test_unsupported_container_fails_before_metadata_read(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            bad_path = request.sources[0].path.with_suffix(".txt")
            bad_path.write_bytes(b"not-supported")
            replacement = SourceSpec("CP1", "back", bad_path, request.sources[0].capture_timestamp)
            request = CaptureRequest(**{**request.__dict__, "sources": (replacement,) + request.sources[1:]})
            with self.assertRaises(CaptureIntakeError) as caught:
                self._plan(request)
            self.assertEqual(caught.exception.reason_code, "unsupported_container")

    def test_source_symlink_is_rejected_without_os_symlink_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            target = request.sources[0].path.resolve()
            path_type = type(target)
            original = path_type.is_symlink
            with mock.patch.object(path_type, "is_symlink", lambda value: value.resolve() == target or original(value)):
                with self.assertRaises(CaptureIntakeError) as caught:
                    self._plan(request)
            self.assertEqual(caught.exception.reason_code, "source_symlink_rejected")

    def test_tiny_avi_header_fixture_is_read_without_frame_read(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tiny.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16))
            self.assertTrue(writer.isOpened())
            for value in (0, 64, 128):
                writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
            writer.release()
            metadata = read_video_header_metadata(path)
            self.assertEqual(metadata["container"], "avi")
            self.assertEqual(metadata["width"], 16)
            self.assertEqual(metadata["height"], 16)
            self.assertEqual(metadata["frame_count"], 3)
            self.assertFalse(metadata["frames_decoded"])

    def test_phase_2k_b_contract_compatibility_is_exact(self):
        self.assertEqual(self.dependency["immutable_manifest_sha256"], intake.EXPECTED_PHASE_2K_B_MANIFEST_SHA256)
        self.assertEqual(self.dependency["capture_contract"]["capture_contract_version"], intake.CAPTURE_CONTRACT_VERSION)
        self.assertEqual(self.registry["entry_count"], 40)
        self.assertEqual(len(self.registry["known_session_ids"]), 4)

    def test_package_calls_canonical_existing_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))
            original = intake.validate_capture_manifest
            with mock.patch.object(intake, "validate_capture_manifest", wraps=original) as validator:
                output, _ = materialize_capture_package(plan)
                self.assertTrue(output.is_dir())
                self.assertGreaterEqual(validator.call_count, 2)

    def test_verify_runner_command_succeeds(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))
            output, _ = materialize_capture_package(plan)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "create_product_phase_2k_c_capture_package.py"),
                    "--repo-root",
                    str(ROOT),
                    "verify",
                    "--package-dir",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue(json.loads(completed.stdout)["valid"])

    def test_canonical_phase_2k_b_validate_capture_runner_succeeds(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self._plan(self._request(temp))
            output, _ = materialize_capture_package(plan)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_product_phase_2k_carry_forward.py"),
                    "validate-capture",
                    "--package-dir",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertIn("CAPTURE_VALIDATION_STATUS=PASS", completed.stdout)

    def test_no_recognition_or_frame_decode_import_or_invocation(self):
        source = inspect.getsource(intake)
        self.assertNotIn("FaceDetectorYN", source)
        self.assertNotIn("FaceRecognizerSF", source)
        header_source = inspect.getsource(read_video_header_metadata)
        self.assertNotIn(".read(", header_source)
        self.assertNotIn("YuNet", header_source)
        self.assertNotIn("SFace", header_source)

    def test_protected_hashes_remain_exact(self):
        current = {relative: intake.sha256_file(ROOT / relative) for relative in intake.PROTECTED_EXPECTED_HASHES}
        self.assertEqual(current, self.baseline_hashes)

    def test_official_and_candidate_totals_are_independently_fixed(self):
        counts = intake.official_and_candidate_counts()
        self.assertEqual(counts["official"], {"Present": 6, "Needs Review": 4, "Unconfirmed": 16, "Missing Enrollment": 1, "Absent": 0, "Total": 27})
        self.assertEqual(counts["candidate"], {"Present": 2, "Needs Review": 8, "Unconfirmed": 16, "Missing Enrollment": 1, "Absent": 0, "Total": 27})


class ToolingValidationTests(unittest.TestCase):
    def test_no_video_tooling_artifact_is_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            output, reused = materialize_tooling_validation(ROOT, Path(temp))
            self.assertFalse(reused)
            manifest = verify_tooling_validation(output)
            self.assertTrue(manifest["immutable"])
            output_again, reused_again = materialize_tooling_validation(ROOT, Path(temp))
            self.assertEqual(output_again, output)
            self.assertTrue(reused_again)
            no_recognition = json.loads((output / "no_recognition_declaration.json").read_text(encoding="utf-8"))
            self.assertFalse(no_recognition["real_capture_package_created"])
            self.assertFalse(no_recognition["recognition_ran"])

    def test_tooling_artifact_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            output, _ = materialize_tooling_validation(ROOT, Path(temp))
            (output / "tooling_policy.json").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(CaptureIntakeError) as caught:
                verify_tooling_validation(output)
            self.assertEqual(caught.exception.reason_code, "package_tampered")


if __name__ == "__main__":
    unittest.main()
