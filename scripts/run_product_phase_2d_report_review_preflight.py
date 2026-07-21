from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as backend
from src.face_attendance.workflow_state import sanitize_status_store


def main() -> int:
    root = ROOT
    status_path = root / "data" / "attendance_status.json"
    overrides_path = root / "data" / "manual_overrides.json"

    if not status_path.is_file():
        raise SystemExit(f"Missing attendance status file: {status_path}")
    payload = json.loads(status_path.read_text(encoding="utf-8-sig"))
    clean, report = sanitize_status_store(payload)
    if report.removed_fields:
        raise SystemExit("attendance_status.json still contains transient runtime fields; rerun Product Phase 2A migration first.")
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8-sig"))
        if not isinstance(overrides, dict):
            raise SystemExit("manual_overrides.json must contain a JSON object.")

    admin = next((user for user in backend.load_role_users() if backend.is_admin_user(user)), None)
    if not admin:
        raise SystemExit("No HOD/Admin user is available for the read-only report preflight.")

    sessions = []
    for key, entry in clean.items():
        compact = backend.compact_attendance_session(str(key), entry, admin)
        if compact:
            if "attendance_data" in compact:
                raise SystemExit(f"Compact report session leaked attendance rows: {key}")
            if compact.get("review_state") not in {"needs_attention", "roster_mismatch", "ready_to_finalize", "finalized"}:
                raise SystemExit(f"Invalid review state for {key}: {compact.get('review_state')}")
            sessions.append(compact)

    unresolved = sum(int(item.get("unresolved_count") or 0) for item in sessions)
    finalized = sum(1 for item in sessions if item.get("attendance_finalized"))
    print("Product Phase 2D read-only report/review preflight")
    print(f"Repository: {root}")
    print(f"Saved attendance sessions visible to HOD/Admin: {len(sessions)}")
    print(f"Finalized sessions: {finalized}")
    print(f"Unresolved student review decisions: {unresolved}")
    print("Recognition executed: no")
    print("Official attendance changed: no")
    print("Production embeddings changed: no")
    print("PREFLIGHT_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
