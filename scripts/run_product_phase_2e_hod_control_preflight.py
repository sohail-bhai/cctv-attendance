from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as backend

EXPECTED_FAMILY = "embfam-274b5207b8b71294ff75"


def main() -> int:
    users = [backend.public_user(user) for user in backend.load_role_users()]
    if not users or users[0].get("name") != "HOD":
        raise SystemExit("HOD public identity normalization failed")
    if any("password" in user for user in users if user):
        raise SystemExit("A public role record exposed a password")

    payload = backend.hod_overview_payload()
    policy = payload["policy"]
    if (policy["checkpoint_count"], policy["present_checkpoints"], policy["checkpoint_min_detections"]) != (5, 3, 2):
        raise SystemExit("Current checkpoint policy does not match the protected operating contract")
    if (policy["match_threshold"], policy["margin_threshold"]) != (0.48, 0.08):
        raise SystemExit("Recognition thresholds changed")

    cvo = next((item for item in payload["subjects"] if item["abbr"] == "CVO"), None)
    if not cvo or cvo["roster_count"] != 27:
        raise SystemExit(f"Expected the authoritative 27-student CVO roster, got {cvo}")
    if not payload["coverage"]["embedding_source_available"]:
        raise SystemExit("Production embedding summary is unavailable")
    if cvo["embedding_available"] != 26:
        raise SystemExit(f"Expected 26/27 CVO embedding coverage after promotion, got {cvo['embedding_available']}/27")

    students = {row["roll"]: row for row in payload["students"]}
    if students.get("24011CSEAI0110", {}).get("embedding_available") is not True:
        raise SystemExit("AI0110 promoted embedding coverage is missing")
    if students.get("2401100CSE0268", {}).get("embedding_available") is not False:
        raise SystemExit("CSE0268 should remain explicitly missing from production embeddings")
    if "24011CSEAI0061" in {str(student.get("roll", "")).upper() for student in backend.load_student_map().get("subject_students", {}).get("CVO", [])}:
        raise SystemExit("AI0061 incorrectly reappeared in the CVO roster")

    production = payload["production"]
    if production.get("status") != "promoted" or production.get("family_id") != EXPECTED_FAMILY:
        raise SystemExit(f"Unexpected production pointer: {json.dumps(production, sort_keys=True)}")

    print("Product Phase 2E read-only HOD-control preflight")
    print(f"Faculty: {len(payload['faculty'])}")
    print(f"Subjects: {len(payload['subjects'])}")
    print(f"Unique roster students: {payload['coverage']['total_students']}")
    print(f"CVO embedding coverage: {cvo['embedding_available']}/{cvo['roster_count']}")
    print(f"Production family: {production['family_id']}")
    print("Authentication, roster, coverage, production pointer, and operating policy: PASS")
    print("Recognition executed: no")
    print("Official attendance changed: no")
    print("Role, roster, timetable, dataset, and embedding files changed: no")
    print("PREFLIGHT_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
