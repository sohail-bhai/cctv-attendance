from __future__ import annotations

import argparse
import json
import secrets
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as backend  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Serve an isolated synthetic authentication fixture for Phase 2L-C browser rehearsal.")
    result.add_argument("--credential-file", type=Path, required=True)
    result.add_argument("--port", type=int, default=5000)
    return result


def main() -> int:
    args = parser().parse_args()
    credential_path = args.credential_file.resolve()
    if credential_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing fixture credential file: {credential_path}")
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase2lc-auth-fixture-") as temp:
        fixture_root = Path(temp)
        password = secrets.token_urlsafe(24)
        hod_username = f"synthetic-hod-{secrets.token_hex(4)}"
        faculty_username = f"synthetic-faculty-{secrets.token_hex(4)}"
        users = [
            {"id": "admin", "username": hod_username, "password": password, "name": "Synthetic HOD", "role": "admin", "roleLabel": "Head of Department", "subjects": ["SWE", "CCM", "CVO"], "canManageSystem": True, "canSeeAll": True},
            {"id": "keerthi", "username": f"synthetic-swe-{secrets.token_hex(4)}", "password": secrets.token_urlsafe(24), "name": "Synthetic SWE Faculty", "role": "faculty", "roleLabel": "Faculty - SWE", "subjects": ["SWE"], "canManageSystem": False, "canSeeAll": False},
            {"id": "vikas", "username": faculty_username, "password": password, "name": "Synthetic CVO Faculty", "role": "faculty", "roleLabel": "Faculty - CVO", "subjects": ["CVO"], "canManageSystem": False, "canSeeAll": False},
            {"id": "pranay", "username": f"synthetic-ccm-{secrets.token_hex(4)}", "password": secrets.token_urlsafe(24), "name": "Synthetic CCM Faculty", "role": "faculty", "roleLabel": "Faculty - CCM", "subjects": ["CCM"], "canManageSystem": False, "canSeeAll": False},
        ]
        backend.USERS_PATH = fixture_root / "role_users.json"
        backend.write_json(backend.USERS_PATH, users)
        status_snapshot = json.loads((ROOT / "data/attendance_status.json").read_text(encoding="utf-8"))
        backend.load_statuses = lambda: deepcopy(status_snapshot)
        backend.repair_stale_processing_statuses = lambda value: value
        backend.ensure_jobs_hydrated = lambda: None
        backend.JOBS = {}
        credential_path.write_text(json.dumps({
            "hod": {"username": hod_username, "password": password},
            "faculty": {"username": faculty_username, "password": password},
        }), encoding="utf-8")
        try:
            backend.app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
        finally:
            credential_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
