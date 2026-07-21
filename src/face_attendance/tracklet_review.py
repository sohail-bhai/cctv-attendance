from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pandas as pd

from .diagnostics import finite_float, json_safe


LABEL_STATUSES = {
    "identified",
    "unidentifiable",
    "mixed_track",
    "not_in_mapping",
    "uncertain",
    "duplicate",
}
LABEL_COLUMNS = ["Package_ID", "Review_ID", "Review_Status", "Actual_Roll", "Reviewer_Notes"]
HIDDEN_COLUMNS = [
    "Package_ID",
    "Review_ID",
    "Tracklet_ID",
    "Session_ID",
    "Subject_Abbr",
    "Checkpoint_ID",
    "Camera_ID",
    "Video",
    "Predicted_Identity_Raw",
    "Predicted_Roll",
    "Best_Score",
    "Second_Identity_Raw",
    "Second_Roll",
    "Second_Score",
    "Margin",
    "Match_Threshold",
    "Margin_Threshold",
    "Aggregate_Mode",
    "Observation_Count",
    "Selected_Observation_Count",
    "Selected_Observation_IDs",
    "Evidence_SHA256",
]


class ReviewExportError(ValueError):
    pass


class LabelValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewPackageResult:
    root: Path
    reviewer_dir: Path
    private_dir: Path
    package_id: str
    accepted_tracklets: int
    evidence_images: int


@dataclass(frozen=True)
class ValidationResult:
    joined: pd.DataFrame
    summary: dict[str, Any]
    checkpoint_metrics: pd.DataFrame
    camera_metrics: pd.DataFrame
    score_metrics: pd.DataFrame
    margin_metrics: pd.DataFrame
    confusions: pd.DataFrame
    student_checkpoint_metrics: pd.DataFrame


@dataclass(frozen=True)
class CorrectionMergeResult:
    output_path: Path
    audit_path: Path
    merged: pd.DataFrame
    summary: dict[str, Any]


def normalize_roll(value: Any) -> str:
    return str(value or "").strip().upper()


def _canonical_known_roll(value: Any, known_rolls: Iterable[str]) -> str:
    raw = normalize_roll(value)
    known = {normalize_roll(roll): str(roll).strip() for roll in known_rolls if str(roll).strip()}
    if raw in known:
        return known[raw]
    for normalized in sorted(known, key=len, reverse=True):
        if raw.startswith(normalized) and raw[len(normalized):len(normalized) + 1] in {"", " ", "(", "-", "/"}:
            return known[normalized]
    token_match = re.match(r"^([A-Z0-9]+)", raw)
    if token_match:
        token = token_match.group(1)
        if any(character.isalpha() for character in token) and any(character.isdigit() for character in token):
            return token
    return str(value or "").strip()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return cleaned or "review"


def _split_ids(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _yes(value: Any) -> bool:
    return str(value or "").strip().lower() in {"yes", "true", "1"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_student_mapping(path: Path, subject_abbr: str) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    with Path(path).open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    subject = str(subject_abbr or "").strip().upper()
    subject_rows = payload.get("subject_students", {}).get(subject, [])
    subject_rolls = {normalize_roll(row.get("roll")) for row in subject_rows if row.get("roll")}
    by_roll: dict[str, dict[str, Any]] = {}
    for row in [*payload.get("all_students", []), *subject_rows]:
        roll = str(row.get("roll") or "").strip()
        if not roll:
            continue
        normalized = normalize_roll(roll)
        by_roll[normalized] = {
            "roll": roll,
            "name": str(row.get("name") or "").strip(),
            "in_subject_roster": normalized in subject_rolls,
        }
    roster = sorted(
        by_roll.values(),
        key=lambda row: (not row["in_subject_roster"], row["name"].upper(), row["roll"].upper()),
    )
    if not roster:
        raise ReviewExportError(
            f"Student mapping contains no students for reviewer export: {subject or 'unspecified subject'}"
        )
    return roster, set(by_roll), subject_rolls


def _video_index(video_root: Path) -> dict[str, list[Path]]:
    if not video_root.is_dir():
        raise ReviewExportError(f"Video root not found: {video_root}")
    index: dict[str, list[Path]] = {}
    for path in video_root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name.lower(), []).append(path)
    return index


def _resolve_video(index: dict[str, list[Path]], checkpoint_id: str, video_name: str) -> Path:
    matches = index.get(str(video_name).lower(), [])
    checkpoint = str(checkpoint_id).strip().upper()
    checkpoint_matches = [
        path for path in matches
        if any(part.upper() == checkpoint or part.upper().startswith(f"{checkpoint}_") for part in path.parts)
    ]
    if len(checkpoint_matches) == 1:
        return checkpoint_matches[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ReviewExportError(f"Source video not found for {checkpoint_id}/{video_name}")
    raise ReviewExportError(f"Ambiguous source video for {checkpoint_id}/{video_name}: {len(matches)} matches")


class _OpenCVFrameLoader:
    def __init__(self) -> None:
        self._captures: dict[Path, cv2.VideoCapture] = {}

    def _open(self, video_path: Path) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ReviewExportError(f"Could not open source video: {video_path}")
        self._captures[video_path] = capture
        return capture

    def __call__(self, video_path: Path, frame_index: int) -> np.ndarray:
        capture = self._captures.get(video_path)
        if capture is None:
            capture = self._open(video_path)
        # Checkpoint diagnostics increment the frame counter immediately after
        # cap.read(), so stored frame numbers are one-based while OpenCV seeks
        # with zero-based indices.
        target = max(0, int(frame_index) - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            return frame

        # Some MP4 decoders cannot seek directly to tail frames even though
        # those frames decode correctly in sequence. Retry from a nearby
        # keyframe window, then from frame zero as the bounded final fallback.
        starts = list(dict.fromkeys([max(0, target - 72), 0]))
        for start in starts:
            capture.release()
            capture = self._open(video_path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            decoded = None
            for _ in range(start, target + 1):
                ok, decoded = capture.read()
                if not ok or decoded is None or decoded.size == 0:
                    decoded = None
                    break
            if decoded is not None:
                return decoded
        raise ReviewExportError(f"Could not read frame {frame_index} from {video_path}")

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


def _parse_bbox(value: Any) -> tuple[float, float, float, float]:
    try:
        parts = [float(part.strip()) for part in str(value).split(",")]
    except (TypeError, ValueError) as exc:
        raise ReviewExportError(f"Invalid diagnostic bounding box: {value}") from exc
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ReviewExportError(f"Invalid diagnostic bounding box: {value}")
    return parts[0], parts[1], parts[2], parts[3]


def _face_crop(frame: np.ndarray, bbox: tuple[float, float, float, float], padding: float) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ReviewExportError("Empty evidence frame")
    x, y, width, height = bbox
    pad_x = width * padding
    pad_y = height * padding
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, int(math.floor(x - pad_x)))
    y1 = max(0, int(math.floor(y - pad_y)))
    x2 = min(frame_width, int(math.ceil(x + width + pad_x)))
    y2 = min(frame_height, int(math.ceil(y + height + pad_y)))
    if x2 <= x1 or y2 <= y1:
        raise ReviewExportError("Diagnostic bounding box falls outside the source frame")
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ReviewExportError("Evidence crop is empty")
    return crop


def _fit_crop(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    crop_height, crop_width = crop.shape[:2]
    scale = min(width / max(crop_width, 1), height / max(crop_height, 1))
    target = (max(1, int(crop_width * scale)), max(1, int(crop_height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(crop, target, interpolation=interpolation)
    x = (width - target[0]) // 2
    y = (height - target[1]) // 2
    canvas[y:y + target[1], x:x + target[0]] = resized
    return canvas


def _write_contact_sheet(
    path: Path,
    review_id: str,
    crops: list[np.ndarray],
    labels: list[str] | None = None,
) -> None:
    if not crops:
        raise ReviewExportError(f"No reviewable evidence for {review_id}")
    if labels is not None and len(labels) != len(crops):
        raise ReviewExportError("Evidence labels must match the number of crops")
    tile_width, tile_height, header_height = 260, 260, 48
    sheet = np.full((tile_height + header_height, tile_width * len(crops), 3), 245, dtype=np.uint8)
    cv2.putText(sheet, review_id, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (24, 32, 42), 2, cv2.LINE_AA)
    for index, crop in enumerate(crops):
        x1 = index * tile_width
        sheet[header_height:, x1:x1 + tile_width] = _fit_crop(crop, tile_width, tile_height)
        cv2.putText(
            sheet,
            labels[index] if labels is not None else f"View {index + 1}",
            (x1 + 12, header_height + tile_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise ReviewExportError(f"Could not write evidence image: {path}")


def _review_order(rows: list[dict[str, Any]], package_id: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{package_id}|{row.get('Tracklet_ID', '')}".encode("utf-8")
        ).hexdigest(),
    )


def _selected_observations(
    tracklet_row: dict[str, Any],
    observation_df: pd.DataFrame,
    evidence_count: int,
) -> list[dict[str, Any]]:
    tracklet_id = str(tracklet_row.get("Tracklet_ID") or "")
    subset = observation_df[observation_df["Tracklet_ID"].astype(str).eq(tracklet_id)].copy()
    if subset.empty:
        raise ReviewExportError(f"No observation diagnostics for review tracklet {tracklet_id}")
    by_id = {str(row["Observation_ID"]): row for row in subset.to_dict("records")}
    selected_ids = _split_ids(tracklet_row.get("Selected_Observation_IDs"))
    selected = [by_id[observation_id] for observation_id in selected_ids if observation_id in by_id]
    if not selected:
        if "Selected_For_Aggregation" in subset:
            subset = subset[subset["Selected_For_Aggregation"].map(_yes)]
        if subset.empty:
            raise ReviewExportError(f"No selected observations for review tracklet {tracklet_id}")
        subset["_quality"] = pd.to_numeric(subset.get("Quality_Weight", 0), errors="coerce").fillna(0)
        selected = subset.sort_values(["_quality", "Frame"], ascending=[False, True]).to_dict("records")
    return selected[:evidence_count]


def _roster_options(roster: list[dict[str, Any]]) -> str:
    course = [row for row in roster if row["in_subject_roster"]]
    other = [row for row in roster if not row["in_subject_roster"]]

    def options(rows: list[dict[str, Any]]) -> str:
        return "".join(
            f'<option value="{html.escape(row["roll"], quote=True)}">'
            f'{html.escape(row["roll"])} | {html.escape(row["name"])}</option>'
            for row in rows
        )

    groups = ['<option value="">Select a student</option>']
    if course:
        groups.append(f'<optgroup label="Course roster">{options(course)}</optgroup>')
    if other:
        groups.append(f'<optgroup label="Other mapped students">{options(other)}</optgroup>')
    return "".join(groups)


def _reviewer_html(manifest: dict[str, Any], roster: list[dict[str, Any]]) -> str:
    option_markup = _roster_options(roster)
    review_title = html.escape(str(manifest.get("review_title") or "Blind tracklet review"))
    review_subtitle = html.escape(
        str(manifest.get("review_subtitle") or f"{manifest.get('session_id', '')} / {manifest.get('subject_abbr', '')}")
    )
    export_prefix_js = json.dumps(
        _safe_component(str(manifest.get("export_filename_prefix") or "tracklet_review_labels"))
    )
    duplicate_option = (
        '<option value="duplicate">Duplicate evidence</option>'
        if manifest.get("allow_duplicate_status")
        else ""
    )
    cards: list[str] = []
    for item in manifest["review_items"]:
        review_id = html.escape(item["review_id"], quote=True)
        status_id = f"status-{review_id}"
        search_id = f"student-search-{review_id}"
        roll_id = f"student-{review_id}"
        notes_id = f"notes-{review_id}"
        details = " / ".join(
            html.escape(str(value))
            for value in (item["checkpoint_id"], item["camera_id"], f'{item["evidence_views"]} views')
            if value
        )
        cards.append(f"""
        <article class="review-card" data-review-id="{review_id}">
          <div class="evidence-panel">
            <div class="item-heading"><strong>{review_id}</strong><span>{details}</span></div>
            <img src="{html.escape(item['evidence_file'], quote=True)}" alt="Evidence views for {review_id}" loading="lazy">
          </div>
          <div class="label-panel">
            <div class="field">
              <label for="{status_id}">Status</label>
              <select id="{status_id}" class="status-input">
                <option value="">Not reviewed</option>
                <option value="identified">Identified</option>
                <option value="unidentifiable">Unidentifiable</option>
                <option value="mixed_track">Mixed people</option>
                <option value="not_in_mapping">Not in student mapping</option>
                <option value="uncertain">Unsure</option>
                {duplicate_option}
              </select>
            </div>
            <div class="field student-field">
              <label for="{search_id}">Search students</label>
              <input id="{search_id}" class="student-search" type="search" placeholder="Search by roll or name" autocomplete="off" spellcheck="false" aria-controls="{roll_id}" disabled>
              <label class="student-select-label" for="{roll_id}">Actual student</label>
              <select id="{roll_id}" class="roll-input" disabled>{option_markup}</select>
              <div class="student-match-count" aria-live="polite"></div>
            </div>
            <div class="field">
              <label for="{notes_id}">Reviewer notes</label>
              <textarea id="{notes_id}" class="notes-input" rows="2" maxlength="500" placeholder="Optional"></textarea>
            </div>
          </div>
        </article>""")

    package_id = html.escape(manifest["package_id"], quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{review_title}</title>
  <style>
    :root {{ color-scheme: light; --ink:#18222d; --muted:#617080; --line:#d8dee5; --paper:#f4f6f8; --white:#fff; --teal:#087f78; --teal-dark:#05655f; --amber:#a85d00; }}
    * {{ box-sizing:border-box; }}
    [hidden] {{ display:none !important; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; letter-spacing:0; }}
    header {{ position:sticky; top:0; z-index:5; border-bottom:1px solid var(--line); background:rgba(255,255,255,.97); }}
    .header-inner {{ max-width:1440px; margin:auto; padding:18px 24px; display:grid; grid-template-columns:1fr auto; gap:18px; align-items:center; }}
    h1 {{ margin:0 0 5px; font-size:24px; line-height:1.2; }}
    .session {{ color:var(--muted); font-size:14px; }}
    .actions {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }}
    button {{ min-height:40px; border:1px solid var(--line); border-radius:6px; background:var(--white); color:var(--ink); padding:0 14px; font:600 14px inherit; cursor:pointer; }}
    button:hover {{ border-color:#9ca8b4; }}
    button.primary {{ border-color:var(--teal); background:var(--teal); color:white; }}
    button.primary:hover {{ background:var(--teal-dark); }}
    .summary-bar {{ max-width:1440px; margin:auto; padding:0 24px 14px; display:grid; grid-template-columns:minmax(180px,1fr) auto; gap:18px; align-items:center; }}
    .progress-track {{ height:8px; border-radius:4px; background:#e2e7eb; overflow:hidden; }}
    .progress-fill {{ width:0; height:100%; background:var(--teal); transition:width .2s ease; }}
    .progress-text {{ font-size:13px; color:var(--muted); white-space:nowrap; }}
    main {{ max-width:1440px; margin:auto; padding:22px 24px 56px; }}
    .filters {{ display:flex; gap:6px; margin-bottom:16px; }}
    .filters button[aria-pressed="true"] {{ border-color:var(--ink); background:var(--ink); color:white; }}
    .review-list {{ display:grid; gap:12px; }}
    .review-card {{ display:grid; grid-template-columns:minmax(0, 1.7fr) minmax(300px, .8fr); border:1px solid var(--line); border-radius:8px; background:var(--white); overflow:hidden; box-shadow:0 3px 14px rgba(28,39,49,.06); }}
    .review-card.complete {{ border-left:4px solid var(--teal); }}
    .evidence-panel {{ min-width:0; overflow:hidden; border-right:1px solid var(--line); }}
    .item-heading {{ height:46px; padding:0 14px; display:flex; align-items:center; justify-content:space-between; gap:12px; border-bottom:1px solid var(--line); }}
    .item-heading span {{ color:var(--muted); font-size:13px; text-align:right; }}
    .evidence-panel img {{ display:block; width:100%; min-height:180px; aspect-ratio:4.2/1; object-fit:contain; background:#161616; pointer-events:none; }}
    .label-panel {{ position:relative; z-index:1; min-width:0; padding:16px; display:grid; gap:13px; align-content:start; background:var(--white); }}
    .field {{ min-width:0; display:grid; gap:6px; }}
    .field > label {{ color:#465462; font-size:12px; font-weight:700; text-transform:uppercase; }}
    .student-field {{ gap:7px; }}
    .student-select-label {{ margin-top:4px; }}
    select, input, textarea {{ width:100%; min-width:0; max-width:100%; border:1px solid #b9c3cc; border-radius:6px; background:white; color:var(--ink); padding:10px 11px; font:14px/1.35 Segoe UI,Arial,sans-serif; letter-spacing:0; text-transform:none; }}
    select, input {{ min-height:42px; }}
    select {{ text-overflow:ellipsis; }}
    select:focus, input:focus, textarea:focus, button:focus-visible {{ outline:3px solid rgba(8,127,120,.22); outline-offset:1px; border-color:var(--teal); }}
    select:disabled, input:disabled {{ background:#edf0f2; color:#89939d; cursor:not-allowed; }}
    .student-match-count {{ min-height:16px; color:var(--muted); font-size:12px; line-height:16px; }}
    textarea {{ resize:vertical; min-height:64px; }}
    .empty {{ padding:40px; text-align:center; color:var(--muted); }}
    @media (max-width:900px) {{ .header-inner {{ grid-template-columns:1fr; }} .actions {{ justify-content:flex-start; }} .review-card {{ grid-template-columns:1fr; }} .evidence-panel {{ border-right:0; border-bottom:1px solid var(--line); }} }}
    @media (max-width:560px) {{ .header-inner, .summary-bar, main {{ padding-left:14px; padding-right:14px; }} h1 {{ font-size:21px; }} .summary-bar {{ grid-template-columns:1fr; gap:7px; }} .actions {{ display:grid; grid-template-columns:1fr 1fr; }} .filters {{ display:grid; grid-template-columns:repeat(3,1fr); }} button {{ padding:0 9px; }} .item-heading {{ align-items:flex-start; height:auto; padding:11px 12px; flex-direction:column; gap:3px; }} .item-heading span {{ text-align:left; }} }}
    @media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
  </style>
</head>
<body data-package-id="{package_id}">
  <header>
    <div class="header-inner">
      <div><h1>{review_title}</h1><div class="session">{review_subtitle}</div></div>
      <div class="actions"><button id="reset" type="button">Clear labels</button><button id="download" class="primary" type="button">Export labels CSV</button></div>
    </div>
    <div class="summary-bar"><div class="progress-track" aria-hidden="true"><div class="progress-fill" id="progress-fill"></div></div><div class="progress-text" id="progress-text" aria-live="polite"></div></div>
  </header>
  <main>
    <div class="filters" aria-label="Review filters"><button type="button" data-filter="all" aria-pressed="true">All</button><button type="button" data-filter="pending" aria-pressed="false">Pending</button><button type="button" data-filter="complete" aria-pressed="false">Completed</button></div>
    <div class="review-list" id="review-list">{''.join(cards)}</div>
    <div class="empty" id="empty" hidden>No items in this view.</div>
  </main>
  <script>
    (() => {{
      const packageId = document.body.dataset.packageId;
      const exportPrefix = {export_prefix_js};
      const storageKey = `tracklet-review:${{packageId}}`;
      const cards = [...document.querySelectorAll('.review-card')];
      let filter = 'all';
      let state = {{}};
      try {{ state = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch (_) {{ state = {{}}; }}
      const clean = value => String(value || '').trim();
      const complete = value => value && value.status && (value.status !== 'identified' || value.roll);
      const identityRequired = status => status === 'identified';
      const save = () => {{ try {{ localStorage.setItem(storageKey, JSON.stringify(state)); }} catch (_) {{}} }};
      const csvCell = value => `"${{String(value ?? '').replace(/"/g, '""')}}"`;
      const update = () => {{
        let completed = 0;
        let visible = 0;
        cards.forEach(card => {{
          const id = card.dataset.reviewId;
          const value = state[id] || {{}};
          const isComplete = complete(value);
          if (isComplete) completed += 1;
          card.classList.toggle('complete', Boolean(isComplete));
          const show = filter === 'all' || (filter === 'complete' && isComplete) || (filter === 'pending' && !isComplete);
          card.hidden = !show;
          if (show) visible += 1;
        }});
        const percent = cards.length ? Math.round(completed / cards.length * 100) : 0;
        document.getElementById('progress-fill').style.width = `${{percent}}%`;
        document.getElementById('progress-text').textContent = `${{completed}} of ${{cards.length}} reviewed (${{percent}}%)`;
        document.getElementById('empty').hidden = visible !== 0;
      }};
      cards.forEach(card => {{
        const id = card.dataset.reviewId;
        const status = card.querySelector('.status-input');
        const search = card.querySelector('.student-search');
        const roll = card.querySelector('.roll-input');
        const matchCount = card.querySelector('.student-match-count');
        const notes = card.querySelector('.notes-input');
        const roster = [...roll.querySelectorAll('optgroup')].flatMap(group =>
          [...group.querySelectorAll('option')].map(option => ({{
            value: option.value,
            label: option.textContent,
            group: group.label,
            searchText: `${{option.value}} ${{option.textContent}}`.toLocaleLowerCase(),
          }}))
        );
        const rosterGroups = [...new Set(roster.map(student => student.group))];
        const renderStudents = query => {{
          const selected = roll.value;
          const term = clean(query).toLocaleLowerCase();
          const matches = roster.filter(student => !term || student.searchText.includes(term) || student.value === selected);
          const fragment = document.createDocumentFragment();
          fragment.append(new Option('Select a student', ''));
          rosterGroups.forEach(groupName => {{
            const groupMatches = matches.filter(student => student.group === groupName);
            if (!groupMatches.length) return;
            const group = document.createElement('optgroup');
            group.label = groupName;
            groupMatches.forEach(student => group.append(new Option(student.label, student.value)));
            fragment.append(group);
          }});
          if (!matches.length) {{
            const empty = new Option('No matching students', '');
            empty.disabled = true;
            fragment.append(empty);
          }}
          roll.replaceChildren(fragment);
          roll.value = selected;
          matchCount.textContent = term ? `${{matches.length}} matching student${{matches.length === 1 ? '' : 's'}}` : '';
        }};
        const setIdentityState = () => {{
          const enabled = identityRequired(status.value);
          search.disabled = !enabled;
          roll.disabled = !enabled;
          if (!enabled) {{
            search.value = '';
            roll.value = '';
            renderStudents('');
          }}
        }};
        const saved = state[id] || {{}};
        status.value = saved.status || '';
        renderStudents('');
        roll.value = saved.roll || '';
        notes.value = saved.notes || '';
        setIdentityState();
        const persist = () => {{
          setIdentityState();
          state[id] = {{status: clean(status.value), roll: clean(roll.value), notes: clean(notes.value)}};
          save(); update();
        }};
        status.addEventListener('change', persist);
        search.addEventListener('input', () => renderStudents(search.value));
        search.addEventListener('keydown', event => {{
          if (event.key === 'ArrowDown' && !roll.disabled) {{ event.preventDefault(); roll.focus(); }}
          if (event.key === 'Escape') {{ search.value = ''; renderStudents(''); }}
        }});
        roll.addEventListener('change', persist);
        notes.addEventListener('input', persist);
      }});
      document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
        filter = button.dataset.filter;
        document.querySelectorAll('[data-filter]').forEach(candidate => candidate.setAttribute('aria-pressed', String(candidate === button)));
        update();
      }}));
      document.getElementById('download').addEventListener('click', () => {{
        const rows = [['Package_ID','Review_ID','Review_Status','Actual_Roll','Reviewer_Notes']];
        cards.forEach(card => {{
          const value = state[card.dataset.reviewId] || {{}};
          rows.push([packageId, card.dataset.reviewId, value.status || '', value.roll || '', value.notes || '']);
        }});
        const csv = rows.map(row => row.map(csvCell).join(',')).join('\\r\\n');
        const url = URL.createObjectURL(new Blob([csv], {{type:'text/csv;charset=utf-8'}}));
        const link = Object.assign(document.createElement('a'), {{href:url, download:`${{exportPrefix}}_${{packageId}}.csv`}});
        link.click(); URL.revokeObjectURL(url);
      }});
      document.getElementById('reset').addEventListener('click', () => {{
        if (!confirm('Clear all saved labels for this package?')) return;
        localStorage.removeItem(storageKey); location.reload();
      }});
      update();
    }})();
  </script>
</body>
</html>
"""


def export_review_package(
    tracklet_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    video_root: Path,
    output_root: Path,
    student_map_path: Path,
    subject_abbr: str,
    diagnostic_run_id: str,
    *,
    evidence_count: int = 5,
    crop_padding: float = 0.45,
    package_id: str | None = None,
    frame_loader: Callable[[Path, int], np.ndarray] | None = None,
    source_files: dict[str, Path] | None = None,
    selected_tracklet_ids: Iterable[str] | None = None,
    review_kind: str = "accepted",
    hidden_extra_columns: dict[str, str] | None = None,
    include_context: bool = False,
    context_padding: float = 2.0,
    package_prefix: str = "tracklet_review",
    subject_roster_only: bool = False,
    allow_empty: bool = False,
    session_id_override: str = "",
    private_metadata: dict[str, Any] | None = None,
) -> ReviewPackageResult:
    if evidence_count < 1 or evidence_count > 10:
        raise ReviewExportError("evidence_count must be between 1 and 10")
    if crop_padding < 0 or crop_padding > 2:
        raise ReviewExportError("crop_padding must be between 0 and 2")
    if context_padding < crop_padding or context_padding > 4:
        raise ReviewExportError("context_padding must be between crop_padding and 4")
    if review_kind not in {"accepted", "unresolved", "shadow_recovery"}:
        raise ReviewExportError("review_kind must be accepted, unresolved, or shadow_recovery")
    required_tracklet = {"Tracklet_ID", "Tracklet_Accepted", "Tracklet_Best_Roll", "Checkpoint_ID", "Camera_ID", "Video"}
    required_observation = {"Tracklet_ID", "Observation_ID", "Frame", "BBox_Original_Coordinates"}
    missing_tracklet = sorted(required_tracklet.difference(tracklet_df.columns))
    missing_observation = sorted(required_observation.difference(observation_df.columns))
    if missing_tracklet:
        raise ReviewExportError(f"Tracklet diagnostics missing columns: {', '.join(missing_tracklet)}")
    if missing_observation:
        raise ReviewExportError(f"Tracklet observations missing columns: {', '.join(missing_observation)}")

    if selected_tracklet_ids is None:
        review_rows = tracklet_df[tracklet_df["Tracklet_Accepted"].map(_yes)].copy()
    else:
        selected_ids = {str(tracklet_id) for tracklet_id in selected_tracklet_ids}
        review_rows = tracklet_df[tracklet_df["Tracklet_ID"].astype(str).isin(selected_ids)].copy()
        missing_ids = sorted(selected_ids.difference(set(review_rows["Tracklet_ID"].astype(str))))
        if missing_ids:
            raise ReviewExportError(f"Selected tracklet ids not found: {', '.join(missing_ids)}")
    if review_rows.empty and not allow_empty:
        raise ReviewExportError(f"No {review_kind} tracklets are available for review export")
    if review_rows["Tracklet_ID"].astype(str).duplicated().any():
        raise ReviewExportError("Review tracklet diagnostics contain duplicate Tracklet_ID values")

    roster, known_rolls, subject_rolls = load_student_mapping(Path(student_map_path), subject_abbr)
    if subject_roster_only:
        roster = [row for row in roster if row["in_subject_roster"]]
        known_rolls = set(subject_rolls)
        if not roster:
            raise ReviewExportError(
                f"Authoritative subject roster is empty for {str(subject_abbr).strip().upper()}"
            )
    package_id = package_id or f"{_safe_component(diagnostic_run_id)}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"
    package_dir = Path(output_root) / f"{_safe_component(package_prefix)}_{_safe_component(package_id)}"
    reviewer_dir = package_dir / "reviewer"
    private_dir = package_dir / "private"
    evidence_dir = reviewer_dir / "evidence"
    package_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)

    ordered = _review_order(review_rows.to_dict("records"), package_id)
    index = _video_index(Path(video_root))
    loader = frame_loader or _OpenCVFrameLoader()
    owns_loader = frame_loader is None
    public_items: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    try:
        for position, tracklet in enumerate(ordered, start=1):
            review_id = f"R{position:04d}"
            observations = _selected_observations(tracklet, observation_df, evidence_count)
            crops: list[np.ndarray] = []
            crop_labels: list[str] = []
            for observation_index, observation in enumerate(observations):
                checkpoint_id = str(observation.get("Checkpoint_ID") or tracklet.get("Checkpoint_ID") or "")
                video_name = str(observation.get("Video") or tracklet.get("Video") or "")
                video_path = _resolve_video(index, checkpoint_id, video_name)
                frame = loader(video_path, int(finite_float(observation.get("Frame"))))
                if include_context and observation_index == 0:
                    crops.append(_face_crop(
                        frame,
                        _parse_bbox(observation.get("BBox_Original_Coordinates")),
                        context_padding,
                    ))
                    crop_labels.append("Context")
                crops.append(_face_crop(frame, _parse_bbox(observation.get("BBox_Original_Coordinates")), crop_padding))
                crop_labels.append(f"Face {observation_index + 1}" if include_context else f"View {observation_index + 1}")

            evidence_name = f"{review_id}.jpg"
            evidence_path = evidence_dir / evidence_name
            _write_contact_sheet(evidence_path, review_id, crops, crop_labels)
            raw_prediction = str(tracklet.get("Tracklet_Best_Roll") or "").strip()
            raw_second = str(tracklet.get("Tracklet_Second_Roll") or "").strip()
            hidden_row = {
                "Package_ID": package_id,
                "Review_ID": review_id,
                "Tracklet_ID": str(tracklet.get("Tracklet_ID") or ""),
                "Session_ID": str(tracklet.get("Session_ID") or ""),
                "Subject_Abbr": str(tracklet.get("Subject_Abbr") or subject_abbr),
                "Checkpoint_ID": str(tracklet.get("Checkpoint_ID") or ""),
                "Camera_ID": str(tracklet.get("Camera_ID") or ""),
                "Video": str(tracklet.get("Video") or ""),
                "Predicted_Identity_Raw": raw_prediction,
                "Predicted_Roll": _canonical_known_roll(raw_prediction, known_rolls),
                "Best_Score": finite_float(tracklet.get("Tracklet_Best_Score")),
                "Second_Identity_Raw": raw_second,
                "Second_Roll": _canonical_known_roll(raw_second, known_rolls),
                "Second_Score": finite_float(tracklet.get("Tracklet_Second_Score")),
                "Margin": finite_float(tracklet.get("Tracklet_Margin")),
                "Match_Threshold": finite_float(tracklet.get("Match_Threshold"), 0.48),
                "Margin_Threshold": finite_float(tracklet.get("Margin_Threshold"), 0.08),
                "Aggregate_Mode": str(tracklet.get("Aggregate_Mode") or ""),
                "Observation_Count": int(finite_float(tracklet.get("Observation_Count"))),
                "Selected_Observation_Count": len(observations),
                "Selected_Observation_IDs": "; ".join(str(row.get("Observation_ID") or "") for row in observations),
                "Evidence_SHA256": _sha256_file(evidence_path),
            }
            for output_column, source_column in (hidden_extra_columns or {}).items():
                hidden_row[output_column] = tracklet.get(source_column, "")
            hidden_rows.append(hidden_row)
            public_items.append({
                "review_id": review_id,
                "checkpoint_id": str(tracklet.get("Checkpoint_ID") or ""),
                "camera_id": str(tracklet.get("Camera_ID") or ""),
                "evidence_file": f"evidence/{evidence_name}",
                "evidence_views": len(crops),
                "evidence_labels": crop_labels,
            })
    finally:
        if owns_loader:
            loader.close()  # type: ignore[attr-defined]

    session_id = str(session_id_override or (ordered[0].get("Session_ID") if ordered else "") or "")
    manifest = {
        "schema_version": 1,
        "package_id": package_id,
        "diagnostic_run_id": diagnostic_run_id,
        "session_id": session_id,
        "subject_abbr": str(subject_abbr).upper(),
        "review_kind": review_kind,
        "review_tracklets": len(public_items),
        "accepted_tracklets": len(public_items) if review_kind == "accepted" else 0,
        "blind_review": True,
        "allow_duplicate_status": review_kind in {"unresolved", "shadow_recovery"},
        "review_items": public_items,
    }
    (reviewer_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    _write_csv(
        reviewer_dir / "labels_template.csv",
        [{"Package_ID": package_id, "Review_ID": item["review_id"], "Review_Status": "", "Actual_Roll": "", "Reviewer_Notes": ""} for item in public_items],
        LABEL_COLUMNS,
    )
    (reviewer_dir / "index.html").write_text(_reviewer_html(manifest, roster), encoding="utf-8")
    (reviewer_dir / "README.txt").write_text(
        "Open index.html, review every evidence strip, and export the completed labels CSV.\n"
        "This directory intentionally contains no recognition predictions.\n",
        encoding="utf-8",
    )
    hidden_columns = [
        *HIDDEN_COLUMNS,
        *[column for column in (hidden_extra_columns or {}) if column not in HIDDEN_COLUMNS],
    ]
    _write_csv(private_dir / "hidden_predictions.csv", hidden_rows, hidden_columns)
    metadata = {
        "schema_version": 1,
        "package_id": package_id,
        "diagnostic_run_id": diagnostic_run_id,
        "session_id": session_id,
        "subject_abbr": str(subject_abbr).upper(),
        "review_kind": review_kind,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "review_tracklets": len(hidden_rows),
        "accepted_tracklets": len(hidden_rows) if review_kind == "accepted" else 0,
        "evidence_count_limit": evidence_count,
        "crop_padding": crop_padding,
        "include_context": include_context,
        "context_padding": context_padding if include_context else None,
        "video_root": str(Path(video_root).resolve()),
        "student_map_path": str(Path(student_map_path).resolve()),
        "known_rolls": sorted(known_rolls),
        "subject_rolls": sorted(subject_rolls),
        "blindness_boundary": {
            "share_with_reviewer": "reviewer",
            "withhold_until_labels_are_final": "private",
            "reviewer_files_contain_predictions": False,
        },
        "source_sha256": {
            name: _sha256_file(Path(path))
            for name, path in (source_files or {}).items()
            if Path(path).is_file()
        },
    }
    reserved_metadata = sorted(set(private_metadata or {}).intersection(metadata))
    if reserved_metadata:
        raise ReviewExportError(
            "Private review metadata cannot override reserved fields: " + ", ".join(reserved_metadata)
        )
    metadata.update(json_safe(private_metadata or {}))
    (private_dir / "export_metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    evaluation_command = (
        "evaluate-shadow-session"
        if review_kind == "shadow_recovery"
        else "evaluate-unresolved"
        if review_kind == "unresolved"
        else "evaluate"
    )
    evaluation_scope = (
        "This shadow workflow evaluates identity precision only. It does not infer presence or calculate attendance recall.\n"
        if review_kind == "shadow_recovery"
        else "Supply an independently verified actual-present roster only when attendance metrics are required.\n"
    )
    (package_dir / "README.txt").write_text(
        "BLIND REVIEW BOUNDARY\n\n"
        "Give the reviewer only the reviewer directory. Keep private withheld until the labels CSV is final.\n"
        f"Run scripts/validate_tracklet_ground_truth.py {evaluation_command} afterward to join final labels "
        "with predictions and calculate metrics.\n"
        f"{evaluation_scope}",
        encoding="utf-8",
    )
    return ReviewPackageResult(
        root=package_dir,
        reviewer_dir=reviewer_dir,
        private_dir=private_dir,
        package_id=package_id,
        accepted_tracklets=len(hidden_rows) if review_kind == "accepted" else 0,
        evidence_images=len(public_items),
    )


def validate_label_dataframe(
    label_df: pd.DataFrame,
    expected_review_ids: set[str],
    package_id: str,
    known_rolls: set[str],
) -> pd.DataFrame:
    missing = sorted(set(LABEL_COLUMNS).difference(label_df.columns))
    if missing:
        raise LabelValidationError(f"Labels CSV missing columns: {', '.join(missing)}")
    labels = label_df[LABEL_COLUMNS].copy().fillna("")
    for column in LABEL_COLUMNS:
        labels[column] = labels[column].astype(str).str.strip()
    labels["Review_ID"] = labels["Review_ID"].str.upper()
    if labels["Review_ID"].eq("").any():
        raise LabelValidationError("Labels CSV contains a blank Review_ID")
    duplicates = sorted(labels.loc[labels["Review_ID"].duplicated(keep=False), "Review_ID"].unique())
    if duplicates:
        raise LabelValidationError(f"Labels CSV contains duplicate Review_ID values: {', '.join(duplicates)}")
    unexpected = sorted(set(labels["Review_ID"]).difference(expected_review_ids))
    if unexpected:
        raise LabelValidationError(f"Labels CSV contains Review_ID values from another package: {', '.join(unexpected)}")
    wrong_package = labels.loc[labels["Package_ID"].ne(package_id), "Package_ID"].unique().tolist()
    if wrong_package:
        raise LabelValidationError("Labels CSV Package_ID does not match the hidden prediction package")
    invalid_status = sorted(set(labels["Review_Status"]).difference(LABEL_STATUSES | {""}))
    if invalid_status:
        raise LabelValidationError(f"Unsupported review status: {', '.join(invalid_status)}")

    normalized_known = {normalize_roll(roll) for roll in known_rolls}
    for row in labels.to_dict("records"):
        status = row["Review_Status"]
        actual = normalize_roll(row["Actual_Roll"])
        if status == "identified" and not actual:
            raise LabelValidationError(f"{row['Review_ID']} is identified but has no Actual_Roll")
        if status == "identified" and actual not in normalized_known:
            raise LabelValidationError(
                f"{row['Review_ID']} Actual_Roll is not in the known student mapping: {row['Actual_Roll']}"
            )
        if status != "identified" and actual:
            raise LabelValidationError(f"{row['Review_ID']} has Actual_Roll but status is not identified")
    labels["Actual_Roll"] = labels["Actual_Roll"].map(normalize_roll)
    return labels


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _classification_metrics(predicted: set[str], actual: set[str]) -> dict[str, Any]:
    true_positive = predicted & actual
    false_positive = predicted - actual
    false_negative = actual - predicted
    precision = _ratio(len(true_positive), len(true_positive) + len(false_positive))
    recall = _ratio(len(true_positive), len(true_positive) + len(false_negative))
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    return {
        "predicted_present_rolls": sorted(predicted),
        "actual_present_rolls": sorted(actual),
        "true_positive_rolls": sorted(true_positive),
        "false_positive_rolls": sorted(false_positive),
        "false_negative_rolls": sorted(false_negative),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _present_rolls(rows: pd.DataFrame, roll_column: str, present_checkpoints: int) -> set[str]:
    checkpoints: dict[str, set[str]] = {}
    for row in rows.to_dict("records"):
        roll = normalize_roll(row.get(roll_column))
        checkpoint = str(row.get("Checkpoint_ID") or "").strip()
        if roll and checkpoint:
            checkpoints.setdefault(roll, set()).add(checkpoint)
    return {roll for roll, values in checkpoints.items() if len(values) >= present_checkpoints}


def _group_metrics(joined: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in joined.groupby(column, dropna=False):
        identified = group[group["Review_Status"].eq("identified")]
        reviewable = group[group["Review_Status"].isin({"identified", "mixed_track", "not_in_mapping"})]
        correct = int(identified["Identity_Correct"].eq("Yes").sum())
        rows.append({
            column: str(value),
            "Accepted_Tracklets": int(len(group)),
            "Reviewed_Tracklets": int(group["Reviewed"].eq("Yes").sum()),
            "Identified_Tracklets": int(len(identified)),
            "Correct_Identifications": correct,
            "Identified_Top1_Accuracy": _ratio(correct, len(identified)),
            "Reviewable_Accepted_Precision": _ratio(correct, len(reviewable)),
        })
    return pd.DataFrame(rows)


def _bucket_metrics(joined: pd.DataFrame, value_column: str, bucket_column: str, boundaries: list[float]) -> pd.DataFrame:
    labels: list[str] = []
    for index in range(len(boundaries) - 1):
        labels.append(f"{boundaries[index]:.2f}-{boundaries[index + 1]:.2f}")
    labels.append(f">={boundaries[-1]:.2f}")

    def bucket(value: Any) -> str:
        number = finite_float(value)
        for index in range(len(boundaries) - 1):
            if boundaries[index] <= number < boundaries[index + 1]:
                return labels[index]
        return labels[-1] if number >= boundaries[-1] else f"<{boundaries[0]:.2f}"

    copy = joined.copy()
    copy[bucket_column] = copy[value_column].map(bucket)
    return _group_metrics(copy, bucket_column)


def _student_checkpoint_rows(joined: pd.DataFrame) -> pd.DataFrame:
    predicted: dict[str, set[str]] = {}
    actual: dict[str, set[str]] = {}
    for row in joined.to_dict("records"):
        checkpoint = str(row.get("Checkpoint_ID") or "")
        predicted_roll = normalize_roll(row.get("Predicted_Roll"))
        if predicted_roll and checkpoint:
            predicted.setdefault(predicted_roll, set()).add(checkpoint)
        if row.get("Review_Status") == "identified":
            actual_roll = normalize_roll(row.get("Actual_Roll"))
            if actual_roll and checkpoint:
                actual.setdefault(actual_roll, set()).add(checkpoint)
    rows: list[dict[str, Any]] = []
    for roll in sorted(set(predicted) | set(actual)):
        predicted_values = predicted.get(roll, set())
        actual_values = actual.get(roll, set())
        rows.append({
            "Roll": roll,
            "Predicted_Checkpoints": "; ".join(sorted(predicted_values)),
            "Predicted_Checkpoint_Count": len(predicted_values),
            "Reviewed_Actual_Checkpoints": "; ".join(sorted(actual_values)),
            "Reviewed_Actual_Checkpoint_Count": len(actual_values),
            "Checkpoint_Intersection_Count": len(predicted_values & actual_values),
        })
    return pd.DataFrame(rows)


def _recommendation(identity: dict[str, Any], attendance: dict[str, Any]) -> dict[str, Any]:
    completion = identity["review_completion_rate"]
    mixed_rate = identity["mixed_tracklet_rate"]
    precision = identity["reviewable_accepted_precision"]
    reviewable_count = identity["reviewable_tracklets"]
    actual_metrics = attendance["actual_present_roster"]
    if completion < 1.0:
        next_step = "complete_blind_review"
        rationale = f"Only {identity['reviewed_tracklets']} of {identity['accepted_tracklets']} accepted tracklets have final labels."
    elif mixed_rate and mixed_rate > 0.05:
        next_step = "analyze_tracklet_association_errors"
        rationale = f"Mixed-person tracklets account for {mixed_rate:.1%} of reviewed evidence."
    elif reviewable_count < 20:
        next_step = "collect_more_reviewable_identity_labels"
        rationale = f"Only {reviewable_count} accepted tracklets are reviewable for precision estimation."
    elif precision is None or precision < 0.90:
        next_step = "analyze_identity_errors_before_recognition_changes"
        rationale = f"Reviewable accepted-track precision is {precision if precision is not None else 'unavailable'}."
    elif actual_metrics["status"] != "available":
        next_step = "collect_actual_present_roster"
        rationale = "Identity labels exist, but full attendance precision and recall require an actual-present roster."
    else:
        next_step = "validate_additional_sessions_and_cameras"
        rationale = "This session has identity and attendance labels; production promotion still requires broader validation."
    return {
        "production_decision": "hold",
        "next_step": next_step,
        "rationale": rationale,
        "production_defaults_should_change": False,
        "decision_gates": {
            "review_completion_required": 1.0,
            "minimum_reviewable_tracklets": 20,
            "target_reviewable_accepted_precision": 0.90,
            "maximum_mixed_tracklet_rate": 0.05,
            "actual_present_roster_required_for_full_attendance_metrics": True,
            "additional_sessions_required_before_production_promotion": True,
        },
    }


def evaluate_predictions(
    prediction_df: pd.DataFrame,
    label_df: pd.DataFrame,
    known_rolls: set[str],
    subject_rolls: set[str],
    actual_present_rolls: set[str] | None,
    present_checkpoints: int = 3,
) -> ValidationResult:
    if present_checkpoints < 1:
        raise LabelValidationError("present_checkpoints must be at least 1")
    required = {"Package_ID", "Review_ID", "Predicted_Roll", "Checkpoint_ID", "Camera_ID", "Best_Score", "Margin"}
    missing = sorted(required.difference(prediction_df.columns))
    if missing:
        raise LabelValidationError(f"Hidden predictions missing columns: {', '.join(missing)}")
    predictions = prediction_df.copy().fillna("")
    if predictions.empty:
        raise LabelValidationError("Hidden predictions are empty")
    package_ids = predictions["Package_ID"].astype(str).unique().tolist()
    if len(package_ids) != 1:
        raise LabelValidationError("Hidden predictions contain multiple Package_ID values")
    predictions["Review_ID"] = predictions["Review_ID"].astype(str).str.strip().str.upper()
    if predictions["Review_ID"].duplicated().any():
        raise LabelValidationError("Hidden predictions contain duplicate Review_ID values")
    labels = validate_label_dataframe(
        label_df,
        set(predictions["Review_ID"]),
        package_ids[0],
        known_rolls,
    )
    joined = predictions.merge(labels, on=["Package_ID", "Review_ID"], how="left", validate="one_to_one")
    for column in ("Review_Status", "Actual_Roll", "Reviewer_Notes"):
        joined[column] = joined[column].fillna("").astype(str)
    joined["Predicted_Roll"] = joined["Predicted_Roll"].map(lambda value: normalize_roll(_canonical_known_roll(value, known_rolls)))
    joined["Reviewed"] = joined["Review_Status"].map(lambda value: "Yes" if value else "No")
    joined["Identity_Correct"] = joined.apply(
        lambda row: (
            "Yes" if row["Review_Status"] == "identified" and normalize_roll(row["Predicted_Roll"]) == normalize_roll(row["Actual_Roll"])
            else "No" if row["Review_Status"] in {"identified", "mixed_track", "not_in_mapping"}
            else ""
        ),
        axis=1,
    )

    total = len(joined)
    reviewed = int(joined["Reviewed"].eq("Yes").sum())
    identified = joined[joined["Review_Status"].eq("identified")]
    reviewable = joined[joined["Review_Status"].isin({"identified", "mixed_track", "not_in_mapping"})]
    correct = int(identified["Identity_Correct"].eq("Yes").sum())
    incorrect = int(identified["Identity_Correct"].eq("No").sum())
    mixed = int(joined["Review_Status"].eq("mixed_track").sum())
    unidentifiable = int(joined["Review_Status"].eq("unidentifiable").sum())
    not_in_mapping = int(joined["Review_Status"].eq("not_in_mapping").sum())
    uncertain = int(joined["Review_Status"].eq("uncertain").sum())
    identity = {
        "accepted_tracklets": total,
        "reviewed_tracklets": reviewed,
        "unreviewed_tracklets": total - reviewed,
        "review_completion_rate": _ratio(reviewed, total),
        "identified_tracklets": len(identified),
        "correct_identifications": correct,
        "incorrect_identifications": incorrect,
        "mixed_tracklets": mixed,
        "unidentifiable_tracklets": unidentifiable,
        "not_in_mapping_tracklets": not_in_mapping,
        "uncertain_tracklets": uncertain,
        "reviewable_tracklets": len(reviewable),
        "identified_top1_accuracy": _ratio(correct, len(identified)),
        "reviewable_accepted_precision": _ratio(correct, len(reviewable)),
        "mixed_tracklet_rate": _ratio(mixed, reviewed),
        "unidentifiable_rate": _ratio(unidentifiable, reviewed),
    }

    identified_rows = joined[joined["Review_Status"].eq("identified")]
    reviewed_predicted_present = _present_rolls(identified_rows, "Predicted_Roll", present_checkpoints)
    reviewed_actual_present = _present_rolls(identified_rows, "Actual_Roll", present_checkpoints)
    review_evidence_metrics = _classification_metrics(reviewed_predicted_present, reviewed_actual_present)
    review_evidence_metrics.update({
        "status": "partial" if reviewed < total else "complete",
        "scope": "identified accepted-track evidence only; missed people cannot be measured",
    })

    predicted_present = _present_rolls(joined, "Predicted_Roll", present_checkpoints)
    if actual_present_rolls is None:
        actual_roster_metrics: dict[str, Any] = {
            "status": "unavailable",
            "reason": "No actual-present roster was supplied; full attendance precision, recall, and F1 were not calculated.",
            "predicted_present_rolls": sorted(predicted_present),
            "precision": None,
            "recall": None,
            "f1": None,
        }
    else:
        normalized_actual = {normalize_roll(roll) for roll in actual_present_rolls if normalize_roll(roll)}
        actual_roster_metrics = {
            "status": "available",
            **_classification_metrics(predicted_present, normalized_actual),
        }
    attendance = {
        "present_checkpoint_rule": present_checkpoints,
        "review_evidence": review_evidence_metrics,
        "actual_present_roster": actual_roster_metrics,
        "subject_roster_size": len(subject_rolls),
    }
    summary = {
        "schema_version": 1,
        "package_id": package_ids[0],
        "identity_metrics": identity,
        "attendance_metrics": attendance,
        "metric_scope": {
            "identity": "accepted tracklets with blind human labels",
            "attendance": "tracklet-derived candidate attendance; official frame-mode attendance is unchanged",
        },
    }
    summary["recommendation"] = _recommendation(identity, attendance)

    incorrect_rows = identified[identified["Identity_Correct"].eq("No")]
    confusions = (
        incorrect_rows.groupby(["Actual_Roll", "Predicted_Roll"], dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values(["Count", "Actual_Roll", "Predicted_Roll"], ascending=[False, True, True])
        if not incorrect_rows.empty
        else pd.DataFrame(columns=["Actual_Roll", "Predicted_Roll", "Count"])
    )
    return ValidationResult(
        joined=joined,
        summary=summary,
        checkpoint_metrics=_group_metrics(joined, "Checkpoint_ID"),
        camera_metrics=_group_metrics(joined, "Camera_ID"),
        score_metrics=_bucket_metrics(joined, "Best_Score", "Score_Band", [0.48, 0.51, 0.55, 0.60]),
        margin_metrics=_bucket_metrics(joined, "Margin", "Margin_Band", [0.08, 0.10, 0.15, 0.20]),
        confusions=confusions,
        student_checkpoint_metrics=_student_checkpoint_rows(joined),
    )


def load_actual_present(path: Path) -> set[str]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str).fillna("")
        if frame.empty or not len(frame.columns):
            return set()
        preferred = next((column for column in frame.columns if column.lower() in {"roll", "roll_no", "roll_number"}), frame.columns[0])
        return {normalize_roll(value) for value in frame[preferred] if normalize_roll(value)}
    rolls: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            rolls.add(normalize_roll(value))
    return rolls


def export_review_package_from_diagnostics(
    diagnostic_run_dir: Path,
    *,
    video_root: Path | None = None,
    output_root: Path | None = None,
    student_map_path: Path,
    subject_abbr: str = "",
    evidence_count: int = 5,
    crop_padding: float = 0.45,
) -> ReviewPackageResult:
    diagnostic_run_dir = Path(diagnostic_run_dir)
    tracklet_files = sorted(diagnostic_run_dir.glob("tracklet_diagnostics_*.csv"))
    observation_files = sorted(diagnostic_run_dir.glob("tracklet_observations_*.csv"))
    if len(tracklet_files) != 1 or len(observation_files) != 1:
        raise ReviewExportError(
            "Diagnostic run must contain exactly one tracklet_diagnostics CSV and one tracklet_observations CSV"
        )
    tracklets = pd.read_csv(tracklet_files[0], dtype=str, keep_default_na=False)
    observations = pd.read_csv(observation_files[0], dtype=str, keep_default_na=False)
    accepted = tracklets[tracklets["Tracklet_Accepted"].map(_yes)]
    if accepted.empty:
        raise ReviewExportError("Diagnostic run contains no accepted tracklets")
    first = accepted.iloc[0]
    inferred_subject = subject_abbr or first.get("Subject_Abbr", "") or first.get("Course_Abbr", "")
    if not inferred_subject:
        raise ReviewExportError("Subject abbreviation is missing; pass --subject-abbr")
    if video_root is None:
        source = str(first.get("Input_Source_Path", "")).strip()
        if not source:
            raise ReviewExportError("Video root is missing; pass --video-dir")
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / candidate
        video_root = candidate
    return export_review_package(
        tracklet_df=tracklets,
        observation_df=observations,
        video_root=video_root,
        output_root=output_root or diagnostic_run_dir,
        student_map_path=student_map_path,
        subject_abbr=inferred_subject,
        diagnostic_run_id=diagnostic_run_dir.name,
        evidence_count=evidence_count,
        crop_padding=crop_padding,
        source_files={"tracklet_diagnostics": tracklet_files[0], "tracklet_observations": observation_files[0]},
    )


def _report_markdown(summary: dict[str, Any]) -> str:
    identity = summary["identity_metrics"]
    attendance = summary["attendance_metrics"]["actual_present_roster"]
    recommendation = summary["recommendation"]
    attendance_text = (
        f"precision {attendance['precision']}, recall {attendance['recall']}, F1 {attendance['f1']}"
        if attendance["status"] == "available"
        else "unavailable (actual-present roster not supplied)"
    )
    return (
        "# Tracklet Ground-Truth Validation\n\n"
        f"- Review completion: {identity['reviewed_tracklets']}/{identity['accepted_tracklets']} "
        f"({identity['review_completion_rate']})\n"
        f"- Identified top-1 accuracy: {identity['identified_top1_accuracy']}\n"
        f"- Reviewable accepted precision: {identity['reviewable_accepted_precision']}\n"
        f"- Mixed tracklets: {identity['mixed_tracklets']}\n"
        f"- Attendance metrics: {attendance_text}\n"
        f"- Production decision: {recommendation['production_decision']}\n"
        f"- Next step: {recommendation['next_step']}\n"
        f"- Rationale: {recommendation['rationale']}\n\n"
        "Official frame-mode attendance and production defaults were not changed.\n"
    )


def load_review_package_data(package_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    package_dir = Path(package_dir)
    private_dir = package_dir / "private"
    predictions_path = private_dir / "hidden_predictions.csv"
    metadata_path = private_dir / "export_metadata.json"
    if not predictions_path.is_file() or not metadata_path.is_file():
        raise LabelValidationError("Review package is missing private hidden predictions or export metadata")
    predictions = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_path = package_dir / "reviewer" / "review_manifest.json"
    if not manifest_path.is_file():
        raise LabelValidationError("Review package is missing the blind review manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_id = str(metadata.get("package_id") or "")
    prediction_package_ids = set(predictions.get("Package_ID", pd.Series(dtype=str)).astype(str))
    if predictions.empty and int(metadata.get("review_tracklets") or 0) == 0:
        prediction_package_ids = {package_id}
    if manifest.get("package_id") != package_id or prediction_package_ids != {package_id}:
        raise LabelValidationError("Review package identifiers do not match")
    manifest_ids = {str(item.get("review_id") or "") for item in manifest.get("review_items", [])}
    prediction_ids = set(predictions.get("Review_ID", pd.Series(dtype=str)).astype(str))
    if manifest_ids != prediction_ids:
        raise LabelValidationError("Blind review manifest does not match the hidden prediction set")
    for row in predictions.to_dict("records"):
        evidence_path = package_dir / "reviewer" / "evidence" / f"{row['Review_ID']}.jpg"
        if not evidence_path.is_file():
            raise LabelValidationError(f"Evidence file is missing for {row['Review_ID']}: {evidence_path}")
        if _sha256_file(evidence_path) != str(row.get("Evidence_SHA256") or ""):
            raise LabelValidationError(f"Evidence integrity check failed for {row['Review_ID']}")
    return predictions, metadata, manifest


def export_label_correction_package(
    source_review_package: Path,
    source_labels_path: Path,
    student_map_path: Path,
    subject_abbr: str,
    *,
    output_root: Path | None = None,
    target_status: str = "not_in_mapping",
    package_id: str | None = None,
) -> ReviewPackageResult:
    """Create a blind, correction-only package from already reviewed evidence.

    The source package and labels remain untouched. Only rows with ``target_status``
    are copied, their original Review_ID values are preserved, and the reviewer is
    limited to the authoritative subject roster.
    """
    if target_status not in LABEL_STATUSES:
        raise ReviewExportError(f"Unsupported correction target status: {target_status}")

    source_review_package = Path(source_review_package)
    source_labels_path = Path(source_labels_path)
    predictions, source_metadata, source_manifest = load_review_package_data(source_review_package)
    source_package_id = str(source_metadata.get("package_id") or "")
    source_known_rolls = set(source_metadata.get("known_rolls", []))
    source_labels_raw = pd.read_csv(source_labels_path, dtype=str, keep_default_na=False)
    source_labels = validate_label_dataframe(
        source_labels_raw,
        set(predictions["Review_ID"].astype(str)),
        source_package_id,
        source_known_rolls,
    )
    missing_source_ids = sorted(set(predictions["Review_ID"].astype(str)).difference(source_labels["Review_ID"]))
    if missing_source_ids:
        raise LabelValidationError(
            f"Source labels are incomplete; missing Review_ID values: {', '.join(missing_source_ids)}"
        )

    target_labels = source_labels[source_labels["Review_Status"].eq(target_status)].copy()
    if target_labels.empty:
        raise ReviewExportError(f"Source labels contain no rows with status {target_status}")

    roster, _, subject_rolls = load_student_mapping(Path(student_map_path), subject_abbr)
    subject_roster = [row for row in roster if row["in_subject_roster"]]
    if not subject_roster:
        raise ReviewExportError(f"No authoritative {str(subject_abbr).upper()} roster is available")

    target_ids = target_labels["Review_ID"].astype(str).tolist()
    target_id_set = set(target_ids)
    manifest_by_id = {
        str(item.get("review_id") or ""): item
        for item in source_manifest.get("review_items", [])
    }
    missing_manifest_ids = sorted(target_id_set.difference(manifest_by_id))
    if missing_manifest_ids:
        raise ReviewExportError(
            f"Source review manifest is missing target evidence: {', '.join(missing_manifest_ids)}"
        )

    package_id = package_id or (
        f"{_safe_component(source_package_id)}-label-correction-"
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"
    )
    if output_root is None:
        output_root = source_review_package / "corrections"
    package_dir = Path(output_root) / f"label_correction_review_{_safe_component(package_id)}"
    reviewer_dir = package_dir / "reviewer"
    private_dir = package_dir / "private"
    evidence_dir = reviewer_dir / "evidence"
    package_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)

    public_items: list[dict[str, Any]] = []
    for review_id in target_ids:
        item = dict(manifest_by_id[review_id])
        relative_evidence = Path(str(item.get("evidence_file") or f"evidence/{review_id}.jpg"))
        source_evidence = source_review_package / "reviewer" / relative_evidence
        if not source_evidence.is_file():
            raise ReviewExportError(f"Source evidence is missing for {review_id}: {source_evidence}")
        destination = evidence_dir / f"{review_id}.jpg"
        shutil.copy2(source_evidence, destination)
        item["evidence_file"] = f"evidence/{review_id}.jpg"
        public_items.append(item)

    correction_manifest = {
        "schema_version": 1,
        "package_id": package_id,
        "diagnostic_run_id": str(source_metadata.get("diagnostic_run_id") or ""),
        "session_id": str(source_manifest.get("session_id") or source_metadata.get("session_id") or ""),
        "subject_abbr": str(subject_abbr).upper(),
        "review_kind": "label_correction",
        "review_tracklets": len(public_items),
        "accepted_tracklets": 0,
        "blind_review": True,
        "allow_duplicate_status": True,
        "review_title": "Roster label correction review",
        "review_subtitle": (
            f"{str(source_manifest.get('session_id') or source_metadata.get('session_id') or '')} / "
            f"{str(subject_abbr).upper()} / {len(public_items)} previously unmapped cards"
        ),
        "export_filename_prefix": "tracklet_label_corrections",
        "source_package_id": source_package_id,
        "target_status": target_status,
        "review_items": public_items,
    }
    (reviewer_dir / "review_manifest.json").write_text(
        json.dumps(correction_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    _write_csv(
        reviewer_dir / "labels_template.csv",
        [
            {
                "Package_ID": package_id,
                "Review_ID": review_id,
                "Review_Status": "",
                "Actual_Roll": "",
                "Reviewer_Notes": "",
            }
            for review_id in target_ids
        ],
        LABEL_COLUMNS,
    )
    (reviewer_dir / "index.html").write_text(
        _reviewer_html(correction_manifest, subject_roster), encoding="utf-8"
    )
    (reviewer_dir / "README.txt").write_text(
        "Open index.html and review only these previously unmapped evidence cards.\n"
        "The student dropdown contains only the authoritative subject roster.\n"
        "Use Not in student mapping for faculty, outsiders, or anyone outside the course roster.\n"
        "Export the completed corrections CSV; do not edit the original labels file.\n",
        encoding="utf-8",
    )

    selected_predictions = predictions[predictions["Review_ID"].astype(str).isin(target_id_set)].copy()
    selected_predictions["Source_Package_ID"] = selected_predictions["Package_ID"]
    selected_predictions["Source_Review_ID"] = selected_predictions["Review_ID"]
    selected_predictions["Package_ID"] = package_id
    selected_predictions.to_csv(private_dir / "hidden_predictions.csv", index=False, encoding="utf-8-sig")
    target_labels.to_csv(private_dir / "source_target_labels.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "schema_version": 1,
        "package_id": package_id,
        "diagnostic_run_id": str(source_metadata.get("diagnostic_run_id") or ""),
        "session_id": correction_manifest["session_id"],
        "subject_abbr": str(subject_abbr).upper(),
        "review_kind": "label_correction",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "review_tracklets": len(public_items),
        "accepted_tracklets": 0,
        "known_rolls": sorted(subject_rolls),
        "subject_rolls": sorted(subject_rolls),
        "source_review_package": str(source_review_package.resolve()),
        "source_package_id": source_package_id,
        "source_known_rolls": sorted(source_known_rolls),
        "source_review_ids": predictions["Review_ID"].astype(str).tolist(),
        "target_status": target_status,
        "target_review_ids": target_ids,
        "source_sha256": {
            "source_labels": _sha256_file(source_labels_path),
            "source_manifest": _sha256_file(source_review_package / "reviewer" / "review_manifest.json"),
            "source_hidden_predictions": _sha256_file(
                source_review_package / "private" / "hidden_predictions.csv"
            ),
            "student_map": _sha256_file(Path(student_map_path)),
        },
        "blindness_boundary": {
            "share_with_reviewer": "reviewer",
            "withhold_until_labels_are_final": "private",
            "reviewer_files_contain_predictions": False,
        },
    }
    (private_dir / "export_metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (package_dir / "README.txt").write_text(
        "CORRECTION-ONLY BLIND REVIEW\n\n"
        "Share only reviewer/ during manual review. Keep private/ withheld.\n"
        "The original Review_ID values and evidence images are preserved.\n"
        "After exporting all corrections, run the merge-corrections command.\n"
        "That command creates a new labels file and refuses to overwrite the original.\n",
        encoding="utf-8",
    )
    return ReviewPackageResult(
        root=package_dir,
        reviewer_dir=reviewer_dir,
        private_dir=private_dir,
        package_id=package_id,
        accepted_tracklets=0,
        evidence_images=len(public_items),
    )


def merge_label_corrections(
    correction_package: Path,
    corrections_path: Path,
    source_labels_path: Path,
    output_path: Path,
) -> CorrectionMergeResult:
    """Merge a completed correction package into a new copy of the source labels."""
    correction_package = Path(correction_package)
    corrections_path = Path(corrections_path)
    source_labels_path = Path(source_labels_path)
    output_path = Path(output_path)
    if output_path.resolve() == source_labels_path.resolve():
        raise LabelValidationError("Correction merge refuses to overwrite the original labels file")
    if output_path.exists():
        raise LabelValidationError(f"Correction merge output already exists: {output_path}")

    predictions, metadata, manifest = load_review_package_data(correction_package)
    if metadata.get("review_kind") != "label_correction" or manifest.get("review_kind") != "label_correction":
        raise LabelValidationError("The supplied package is not a label-correction review package")

    expected_source_hash = str(metadata.get("source_sha256", {}).get("source_labels") or "")
    if not expected_source_hash or _sha256_file(source_labels_path) != expected_source_hash:
        raise LabelValidationError(
            "Source labels no longer match the file used to create this correction package"
        )

    source_package_id = str(metadata.get("source_package_id") or "")
    source_review_ids = {str(value) for value in metadata.get("source_review_ids", [])}
    target_ids = {str(value) for value in metadata.get("target_review_ids", [])}
    target_status = str(metadata.get("target_status") or "not_in_mapping")
    if not source_package_id or not source_review_ids or not target_ids:
        raise LabelValidationError("Correction package provenance is incomplete")

    source_raw = pd.read_csv(source_labels_path, dtype=str, keep_default_na=False)
    source_labels = validate_label_dataframe(
        source_raw,
        source_review_ids,
        source_package_id,
        set(metadata.get("source_known_rolls", [])),
    )
    missing_source_ids = sorted(source_review_ids.difference(source_labels["Review_ID"]))
    if missing_source_ids:
        raise LabelValidationError(
            f"Source labels are incomplete; missing Review_ID values: {', '.join(missing_source_ids)}"
        )
    current_target_statuses = source_labels.set_index("Review_ID").loc[sorted(target_ids), "Review_Status"]
    changed_targets = current_target_statuses[current_target_statuses.ne(target_status)]
    if not changed_targets.empty:
        raise LabelValidationError(
            "One or more correction targets no longer have their original status: "
            + ", ".join(changed_targets.index.tolist())
        )

    corrections_raw = pd.read_csv(corrections_path, dtype=str, keep_default_na=False)
    corrections = validate_label_dataframe(
        corrections_raw,
        target_ids,
        str(metadata.get("package_id") or ""),
        set(metadata.get("subject_rolls", [])),
    )
    correction_ids = set(corrections["Review_ID"])
    missing_corrections = sorted(target_ids.difference(correction_ids))
    if missing_corrections:
        raise LabelValidationError(
            f"Correction labels are incomplete; missing Review_ID values: {', '.join(missing_corrections)}"
        )
    incomplete = corrections[corrections["Review_Status"].eq("")]["Review_ID"].tolist()
    if incomplete:
        raise LabelValidationError(
            f"Correction labels still contain unreviewed cards: {', '.join(incomplete)}"
        )

    correction_by_id = corrections.set_index("Review_ID")
    merged = source_labels.copy()
    before_status_counts = merged["Review_Status"].value_counts().sort_index().to_dict()
    for index, row in merged.iterrows():
        review_id = row["Review_ID"]
        if review_id not in target_ids:
            continue
        correction = correction_by_id.loc[review_id]
        merged.at[index, "Review_Status"] = correction["Review_Status"]
        merged.at[index, "Actual_Roll"] = correction["Actual_Roll"]
        correction_notes = str(correction["Reviewer_Notes"] or "").strip()
        if correction_notes:
            merged.at[index, "Reviewer_Notes"] = correction_notes

    if len(merged) != len(source_labels) or set(merged["Review_ID"]) != source_review_ids:
        raise LabelValidationError("Correction merge changed the source label row identity set")
    unchanged_ids = source_review_ids.difference(target_ids)
    source_by_id = source_labels.set_index("Review_ID")
    merged_by_id = merged.set_index("Review_ID")
    comparison_columns = [column for column in LABEL_COLUMNS if column != "Review_ID"]
    for review_id in unchanged_ids:
        if not source_by_id.loc[review_id, comparison_columns].equals(
            merged_by_id.loc[review_id, comparison_columns]
        ):
            raise LabelValidationError(f"Correction merge modified a non-target row: {review_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, merged.to_dict("records"), LABEL_COLUMNS)
    summary = {
        "schema_version": 1,
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "source_labels": str(source_labels_path.resolve()),
        "source_labels_sha256": _sha256_file(source_labels_path),
        "correction_package": str(correction_package.resolve()),
        "corrections": str(corrections_path.resolve()),
        "corrections_sha256": _sha256_file(corrections_path),
        "output": str(output_path.resolve()),
        "output_sha256": _sha256_file(output_path),
        "source_package_id": source_package_id,
        "correction_package_id": str(metadata.get("package_id") or ""),
        "source_rows": len(source_labels),
        "target_rows": len(target_ids),
        "unchanged_rows": len(unchanged_ids),
        "target_review_ids": sorted(target_ids),
        "status_counts_before": before_status_counts,
        "status_counts_after": merged["Review_Status"].value_counts().sort_index().to_dict(),
        "original_overwritten": False,
    }
    audit_path = output_path.with_suffix(output_path.suffix + ".merge_audit.json")
    audit_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    return CorrectionMergeResult(
        output_path=output_path,
        audit_path=audit_path,
        merged=merged,
        summary=summary,
    )


def evaluate_review_package(
    package_dir: Path,
    labels_path: Path,
    *,
    actual_present_path: Path | None = None,
    output_dir: Path | None = None,
    present_checkpoints: int = 3,
) -> tuple[ValidationResult, Path]:
    package_dir = Path(package_dir)
    predictions, metadata, _ = load_review_package_data(package_dir)

    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    actual_present = load_actual_present(actual_present_path) if actual_present_path else None
    result = evaluate_predictions(
        prediction_df=predictions,
        label_df=labels,
        known_rolls=set(metadata.get("known_rolls", [])),
        subject_rolls=set(metadata.get("subject_rolls", [])),
        actual_present_rolls=actual_present,
        present_checkpoints=present_checkpoints,
    )
    if output_dir is None:
        output_dir = package_dir / "evaluation" / datetime.now().strftime("validation_%Y%m%d_%H%M%S_%f")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result.joined.to_csv(output_dir / "joined_tracklet_review.csv", index=False)
    result.checkpoint_metrics.to_csv(output_dir / "checkpoint_identity_metrics.csv", index=False)
    result.camera_metrics.to_csv(output_dir / "camera_identity_metrics.csv", index=False)
    result.score_metrics.to_csv(output_dir / "score_band_metrics.csv", index=False)
    result.margin_metrics.to_csv(output_dir / "margin_band_metrics.csv", index=False)
    result.confusions.to_csv(output_dir / "identity_confusions.csv", index=False)
    result.student_checkpoint_metrics.to_csv(output_dir / "student_checkpoint_metrics.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(json_safe(result.summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "VALIDATION_REPORT.md").write_text(_report_markdown(result.summary), encoding="utf-8")
    return result, output_dir
