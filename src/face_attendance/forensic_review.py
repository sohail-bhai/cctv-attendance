from __future__ import annotations

import csv
import html
import json
import os
import shutil
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from .diagnostics import json_safe
from .shadow_validation import (
    ShadowValidationError,
    create_reviewer_server,
    verify_output_manifest,
    write_output_manifest,
)
from .tracklet_review import _sha256_file


FORENSIC_REVIEW_SCHEMA_VERSION = 1
APPROVAL_COLUMNS = [
    "Package_ID",
    "Item_ID",
    "Review_Action",
    "Duplicate_Of_Item_ID",
    "Reviewer_Notes",
]


class ForensicReviewError(ValueError):
    pass


@dataclass(frozen=True)
class ForensicReviewItem:
    item_id: str
    evidence_source: Path
    public: dict[str, Any]
    private: dict[str, Any]
    preserve_resolution: bool = False


@dataclass(frozen=True)
class ForensicReviewPackage:
    root: Path
    reviewer_dir: Path
    private_dir: Path
    package_id: str
    package_kind: str
    item_count: int
    manifest_path: Path


@dataclass(frozen=True)
class ApprovalValidationResult:
    approvals: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]


def _safe_component(value: Any) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(character if character in allowed else "_" for character in str(value or ""))
    return cleaned.strip("._-") or "review"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _preview_image(source: Path, target: Path, preserve_resolution: bool) -> None:
    source = Path(source)
    target = Path(target)
    image = cv2.imread(str(source))
    if image is None or image.size == 0:
        placeholder = np.full((360, 640, 3), 245, dtype=np.uint8)
        cv2.rectangle(placeholder, (18, 18), (621, 341), (170, 178, 188), 2)
        cv2.putText(
            placeholder,
            "Image could not be decoded",
            (95, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (39, 47, 58),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            placeholder,
            source.name[:55],
            (95, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (78, 88, 102),
            1,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(target), placeholder, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ForensicReviewError(f"Could not write evidence placeholder: {target}")
        return

    if preserve_resolution:
        suffix = source.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            shutil.copy2(source, target)
            return

    height, width = image.shape[:2]
    scale = min(1.0, 1280.0 / max(width, height, 1))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise ForensicReviewError(f"Could not write reviewer evidence: {target}")


def _display_pairs(public: dict[str, Any]) -> str:
    rows: list[str] = []
    for key, value in public.items():
        text = str(value or "").strip()
        if not text:
            continue
        label = str(key).replace("_", " ").strip().title()
        key_lower = str(key).lower()
        if "similarity_hint" in key_lower:
            extra_class = " hint"
        elif any(
            token in key_lower
            for token in ("primary_selection_reason", "priority_tier", "selected_issue", "confusion")
        ):
            extra_class = " priority"
        elif "warning" in key_lower:
            extra_class = " warning"
        else:
            extra_class = ""
        rows.append(
            f'<div class="fact{extra_class}"><dt>{html.escape(label)}</dt>'
            f'<dd>{html.escape(text)}</dd></div>'
        )
    return "".join(rows)


def _reviewer_html(
    *,
    package_id: str,
    package_kind: str,
    title: str,
    instructions: str,
    items: list[dict[str, Any]],
    actions: dict[str, str],
) -> str:
    action_options = "".join(
        f'<option value="{html.escape(code)}">{html.escape(label)}</option>'
        for code, label in actions.items()
    )
    cards = []
    for index, item in enumerate(items, start=1):
        item_id = str(item["item_id"])
        cards.append(
            f"""
            <article class="review-card" data-item-id="{html.escape(item_id)}">
              <div class="evidence-pane">
                <div class="item-index">Item {index} of {len(items)}</div>
                <img src="{html.escape(item['evidence_file'])}" alt="Review evidence for item {index}" loading="lazy">
              </div>
              <div class="decision-pane">
                <dl class="facts">{_display_pairs(dict(item.get('public') or {}))}</dl>
                <label class="field">
                  <span>Review action</span>
                  <select class="action" aria-label="Review action for item {index}" required>
                    <option value="">Choose an action</option>
                    {action_options}
                  </select>
                </label>
                <label class="field duplicate-field" hidden>
                  <span>Duplicate of item ID</span>
                  <input class="duplicate" type="text" inputmode="text" placeholder="Example: ENR-..." autocomplete="off">
                </label>
                <label class="field">
                  <span>Reviewer note <small>optional</small></span>
                  <textarea class="notes" rows="3" placeholder="Add only evidence-based context"></textarea>
                </label>
              </div>
            </article>
            """
        )
    empty_state = ""
    if not cards:
        empty_state = '<div class="empty">No review items were selected. Export remains available as an empty, complete package.</div>'
    payload = json.dumps(
        {
            "package_id": package_id,
            "package_kind": package_kind,
            "item_ids": [item["item_id"] for item in items],
            "duplicate_action": (
                "duplicate_of_another_image"
                if "duplicate_of_another_image" in actions
                else ""
            ),
        },
        ensure_ascii=True,
    ).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#17202b; --muted:#566274; --line:#cfd6df; --surface:#fff; --subtle:#f4f6f8; --accent:#0a5f8f; --accent-strong:#08496e; --focus:#c05621; --danger:#9b2c2c; }}
    * {{ box-sizing:border-box; }}
    [hidden] {{ display:none!important; }}
    body {{ margin:0; background:#e9edf1; color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.45; }}
    button,input,select,textarea {{ font:inherit; }}
    .topbar {{ position:sticky; top:0; z-index:20; border-bottom:1px solid #aeb8c5; background:rgba(255,255,255,.97); }}
    .topbar-inner {{ max-width:1480px; margin:auto; padding:14px 22px; display:flex; align-items:center; gap:18px; flex-wrap:wrap; }}
    h1 {{ margin:0; font-size:1.18rem; letter-spacing:0; }}
    .progress {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
    .toolbar {{ margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; }}
    button {{ min-height:40px; border:0; border-radius:6px; padding:8px 14px; background:var(--accent); color:#fff; font-weight:700; cursor:pointer; }}
    button:hover {{ background:var(--accent-strong); }}
    button.secondary {{ border:1px solid #9ca8b7; background:#fff; color:var(--ink); }}
    button.secondary:hover {{ background:var(--subtle); }}
    :focus-visible {{ outline:3px solid var(--focus); outline-offset:2px; }}
    main {{ max-width:1480px; margin:auto; padding:22px; }}
    .intro {{ max-width:78ch; margin:0 0 18px; color:#384657; }}
    .filter-row {{ display:flex; gap:10px; margin-bottom:18px; flex-wrap:wrap; }}
    .filter-row input {{ flex:1 1 360px; max-width:620px; min-height:42px; border:1px solid #8794a5; border-radius:6px; padding:9px 11px; color:var(--ink); background:#fff; }}
    .filter-row select {{ flex:0 1 220px; min-height:42px; }}
    .filter-row button {{ min-height:42px; }}
    .review-list {{ display:grid; gap:16px; }}
    .review-card {{ display:grid; grid-template-columns:minmax(360px,1.15fr) minmax(340px,.85fr); min-height:380px; border:1px solid #b8c2ce; border-radius:8px; overflow:hidden; background:var(--surface); }}
    .review-card.complete {{ border-color:#2f855a; }}
    .evidence-pane {{ min-width:0; padding:14px; background:#202731; display:flex; flex-direction:column; gap:10px; }}
    .item-index {{ color:#e5e9ee; font-size:.88rem; font-weight:700; }}
    .evidence-pane img {{ display:block; width:100%; height:100%; min-height:320px; max-height:680px; object-fit:contain; background:#11161d; }}
    .decision-pane {{ min-width:0; padding:20px; overflow-wrap:anywhere; }}
    .facts {{ margin:0 0 18px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px 14px; }}
    .fact {{ min-width:0; padding-bottom:8px; border-bottom:1px solid #e3e7ec; }}
    .fact.hint {{ grid-column:1/-1; padding:10px; border:1px solid #c78a2a; border-radius:6px; background:#fff8e8; }}
    .fact.priority {{ padding:10px; border:0; border-radius:6px; background:#eaf3f8; }}
    .fact.warning {{ padding:10px; border:0; border-radius:6px; background:#fff4d8; }}
    dt {{ color:var(--muted); font-size:.78rem; font-weight:700; }}
    dd {{ margin:2px 0 0; font-weight:650; }}
    .field {{ display:block; margin-top:13px; }}
    .field > span {{ display:block; margin-bottom:5px; font-weight:700; }}
    .field small {{ color:var(--muted); font-weight:500; }}
    select,input,textarea {{ width:100%; border:1px solid #8794a5; border-radius:6px; padding:9px 10px; background:#fff; color:var(--ink); }}
    select:disabled,input:disabled,textarea:disabled {{ color:#6f7885; background:#e9edf1; cursor:not-allowed; }}
    textarea {{ resize:vertical; }}
    .empty {{ padding:30px; border:1px solid var(--line); background:#fff; text-align:center; }}
    .notice {{ position:fixed; right:18px; bottom:18px; z-index:40; max-width:420px; padding:11px 14px; border-radius:6px; background:#17202b; color:#fff; opacity:0; pointer-events:none; transform:translateY(8px); transition:opacity .18s ease-out,transform .18s ease-out; }}
    .notice.show {{ opacity:1; transform:none; }}
    @media (max-width:850px) {{
      .topbar-inner,main {{ padding-left:12px; padding-right:12px; }}
      .toolbar {{ width:100%; margin-left:0; }}
      .review-card {{ grid-template-columns:1fr; }}
      .evidence-pane img {{ min-height:240px; max-height:520px; }}
      .facts {{ grid-template-columns:1fr; }}
      .fact.hint {{ grid-column:auto; }}
    }}
    @media (prefers-reduced-motion:reduce) {{ .notice {{ transition:none; }} }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <h1>{html.escape(title)}</h1>
      <div class="progress" id="progress" aria-live="polite">0 of {len(items)} reviewed</div>
      <div class="toolbar">
        <button class="secondary" id="show-pending" type="button">Show pending</button>
        <button id="export" type="button">Export approvals CSV</button>
      </div>
    </div>
  </header>
  <main>
    <p class="intro">{html.escape(instructions)}</p>
    <div class="filter-row">
      <input id="search" type="search" placeholder="Filter by roll, filename, warning, session, or item ID" aria-label="Filter review items">
      <select id="status-filter" aria-label="Filter by review status">
        <option value="all">All review statuses</option>
        <option value="pending">Pending only</option>
        <option value="reviewed">Reviewed only</option>
      </select>
      <button class="secondary" id="clear-filters" type="button">Clear filters</button>
    </div>
    <section class="review-list" id="review-list">{''.join(cards)}{empty_state}</section>
  </main>
  <div class="notice" id="notice" role="status"></div>
  <script type="application/json" id="package-data">{payload}</script>
  <script>
  (() => {{
    'use strict';
    const config = JSON.parse(document.getElementById('package-data').textContent);
    const storageKey = `forensic-review:${{config.package_id}}`;
    const cards = Array.from(document.querySelectorAll('.review-card'));
    const notice = document.getElementById('notice');
    const search = document.getElementById('search');
    const statusFilter = document.getElementById('status-filter');
    const state = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
    const csv = value => `"${{String(value ?? '').replaceAll('"','""')}}"`;
    const save = () => localStorage.setItem(storageKey, JSON.stringify(state));
    const flash = message => {{ notice.textContent = message; notice.classList.add('show'); setTimeout(() => notice.classList.remove('show'), 1600); }};
    const applyFilters = () => {{
      const query = search.value.trim().toLowerCase();
      const status = statusFilter.value;
      cards.forEach(card => {{
        const reviewed = Boolean((state[card.dataset.itemId] || {{}}).action);
        const statusMatches = status === 'all' || (status === 'pending' && !reviewed) || (status === 'reviewed' && reviewed);
        const haystack = `${{card.dataset.itemId}} ${{card.textContent}}`.toLowerCase();
        card.hidden = !statusMatches || (Boolean(query) && !haystack.includes(query));
      }});
    }};
    const updateProgress = () => {{
      const complete = cards.filter(card => (state[card.dataset.itemId] || {{}}).action).length;
      const percentage = cards.length
        ? (complete === 0 ? 0 : Math.max(1, Math.round(complete/cards.length*100)))
        : 100;
      document.getElementById('progress').textContent = `${{complete}} of ${{cards.length}} reviewed (${{percentage}}%)`;
      cards.forEach(card => card.classList.toggle('complete', Boolean((state[card.dataset.itemId] || {{}}).action)));
      applyFilters();
    }};
    cards.forEach(card => {{
      const id = card.dataset.itemId;
      const action = card.querySelector('.action');
      const duplicate = card.querySelector('.duplicate');
      const duplicateField = card.querySelector('.duplicate-field');
      const notes = card.querySelector('.notes');
      const saved = state[id] || {{}};
      action.value = saved.action || '';
      duplicate.value = saved.duplicate || '';
      notes.value = saved.notes || '';
      const syncDuplicate = () => {{
        const enabled = Boolean(config.duplicate_action) && action.value === config.duplicate_action;
        duplicateField.hidden = !enabled;
        duplicate.disabled = !enabled;
        if (!enabled) duplicate.value = '';
      }};
      const persist = () => {{
        syncDuplicate();
        state[id] = {{ action: action.value, duplicate: duplicate.value.trim(), notes: notes.value.trim() }};
        save(); updateProgress();
      }};
      action.addEventListener('change', persist);
      duplicate.addEventListener('input', persist);
      notes.addEventListener('input', persist);
      syncDuplicate();
    }});
    search.addEventListener('input', applyFilters);
    statusFilter.addEventListener('change', applyFilters);
    document.getElementById('clear-filters').addEventListener('click', () => {{
      search.value = '';
      statusFilter.value = 'all';
      applyFilters();
      search.focus();
    }});
    document.getElementById('show-pending').addEventListener('click', () => {{
      statusFilter.value = 'pending';
      applyFilters();
      const first = cards.find(card => !card.hidden);
      if (first) first.scrollIntoView({{behavior:'smooth',block:'start'}}); else flash('Every item has a review action.');
    }});
    document.getElementById('export').addEventListener('click', () => {{
      const pending = cards.filter(card => !(state[card.dataset.itemId] || {{}}).action);
      if (pending.length) {{ flash(`${{pending.length}} item(s) still need a review action.`); pending[0].scrollIntoView({{behavior:'smooth',block:'center'}}); return; }}
      const rows = [['Package_ID','Item_ID','Review_Action','Duplicate_Of_Item_ID','Reviewer_Notes']];
      config.item_ids.forEach(id => {{ const value = state[id] || {{}}; rows.push([config.package_id,id,value.action || '',value.duplicate || '',value.notes || '']); }});
      const body = '\ufeff' + rows.map(row => row.map(csv).join(',')).join('\\r\\n');
      const link = document.createElement('a');
      link.href = URL.createObjectURL(new Blob([body],{{type:'text/csv;charset=utf-8'}}));
      link.download = `forensic_review_approvals_${{config.package_id}}.csv`;
      document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      flash('Approval CSV exported. No source files were changed.');
    }});
    updateProgress();
  }})();
  </script>
</body>
</html>
"""


def export_forensic_review_package(
    *,
    output_dir: Path,
    package_id: str,
    package_kind: str,
    title: str,
    instructions: str,
    actions: dict[str, str],
    items: Iterable[ForensicReviewItem],
    metadata: dict[str, Any] | None = None,
) -> ForensicReviewPackage:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ForensicReviewError(f"Refusing to overwrite review package: {output_dir}")
    item_list = list(items)
    item_ids = [str(item.item_id) for item in item_list]
    if len(item_ids) != len(set(item_ids)) or any(not item_id for item_id in item_ids):
        raise ForensicReviewError("Review item IDs must be non-empty and unique")
    if not package_id or not package_kind or not actions:
        raise ForensicReviewError("Package ID, package kind, and review actions are required")

    reviewer_dir = output_dir / "reviewer"
    private_dir = output_dir / "private"
    evidence_dir = reviewer_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    private_dir.mkdir(parents=True, exist_ok=False)
    public_items: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for item in item_list:
        source = Path(item.evidence_source).resolve()
        if not source.is_file():
            raise ForensicReviewError(f"Review source image not found: {source}")
        suffix = source.suffix.lower() if item.preserve_resolution and source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
        evidence_name = f"{_safe_component(item.item_id)}{suffix}"
        target = evidence_dir / evidence_name
        _preview_image(source, target, item.preserve_resolution)
        public_items.append(
            {
                "item_id": item.item_id,
                "evidence_file": f"evidence/{evidence_name}",
                "evidence_sha256": _sha256_file(target),
                "public": json_safe(item.public),
            }
        )
        private_rows.append(
            {
                "Package_ID": package_id,
                "Item_ID": item.item_id,
                "Source_Path": str(source),
                "Source_SHA256": _sha256_file(source),
                "Reviewer_Evidence_File": f"reviewer/evidence/{evidence_name}",
                "Reviewer_Evidence_SHA256": _sha256_file(target),
                "Private_Provenance_JSON": json.dumps(json_safe(item.private), sort_keys=True, ensure_ascii=True),
            }
        )

    public_manifest = {
        "schema_version": FORENSIC_REVIEW_SCHEMA_VERSION,
        "package_id": package_id,
        "package_kind": package_kind,
        "title": title,
        "item_count": len(item_list),
        "actions": actions,
        "approval_columns": APPROVAL_COLUMNS,
        "review_items": public_items,
        "browser_private_mapping_loaded": False,
    }
    (reviewer_dir / "review_manifest.json").write_text(
        json.dumps(public_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    _write_csv(
        reviewer_dir / "approvals_template.csv",
        (
            {
                "Package_ID": package_id,
                "Item_ID": item_id,
                "Review_Action": "",
                "Duplicate_Of_Item_ID": "",
                "Reviewer_Notes": "",
            }
            for item_id in item_ids
        ),
        APPROVAL_COLUMNS,
    )
    (reviewer_dir / "index.html").write_text(
        _reviewer_html(
            package_id=package_id,
            package_kind=package_kind,
            title=title,
            instructions=instructions,
            items=public_items,
            actions=actions,
        ),
        encoding="utf-8",
    )
    (reviewer_dir / "README.txt").write_text(
        "Serve this reviewer through the generated localhost launcher. Complete every item, "
        "then export the approval CSV. Exporting does not modify source images or embeddings.\n",
        encoding="utf-8",
    )
    private_columns = [
        "Package_ID",
        "Item_ID",
        "Source_Path",
        "Source_SHA256",
        "Reviewer_Evidence_File",
        "Reviewer_Evidence_SHA256",
        "Private_Provenance_JSON",
    ]
    _write_csv(private_dir / "source_mapping.csv", private_rows, private_columns)
    export_metadata = {
        "schema_version": FORENSIC_REVIEW_SCHEMA_VERSION,
        "package_id": package_id,
        "package_kind": package_kind,
        "item_count": len(item_list),
        "actions": list(actions),
        "source_mapping_sha256": _sha256_file(private_dir / "source_mapping.csv"),
        "source_files_unchanged": True,
        "production_embeddings_changed": False,
        "browser_private_mapping_loaded": False,
        **json_safe(metadata or {}),
    }
    (private_dir / "export_metadata.json").write_text(
        json.dumps(export_metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    manifest_path = write_output_manifest(
        output_dir,
        {
            "package_id": package_id,
            "package_kind": package_kind,
            "item_count": len(item_list),
            "source_files_changed": False,
            "production_embeddings_changed": False,
        },
        filename="package_manifest.json",
    )
    verify_output_manifest(output_dir, manifest_path)
    audit = audit_forensic_review_package(output_dir)
    if not audit["privacy_passed"]:
        raise ForensicReviewError(
            "Reviewer package leaks private prediction/provenance fields: "
            + ", ".join(audit["forbidden_matches"][:10])
        )
    return ForensicReviewPackage(
        root=output_dir,
        reviewer_dir=reviewer_dir,
        private_dir=private_dir,
        package_id=package_id,
        package_kind=package_kind,
        item_count=len(item_list),
        manifest_path=manifest_path,
    )


def audit_forensic_review_package(package_dir: Path) -> dict[str, Any]:
    package_dir = Path(package_dir)
    reviewer_dir = package_dir / "reviewer"
    private_mapping = package_dir / "private" / "source_mapping.csv"
    if not reviewer_dir.is_dir() or not private_mapping.is_file():
        raise ForensicReviewError(f"Forensic review package is incomplete: {package_dir}")
    forbidden_tokens = {
        "Private_Provenance_JSON",
        "source_mapping.csv",
        "../private",
        "Predicted_Roll",
        "Predicted_Identity_Raw",
        "Second_Roll",
        "Best_Score",
        "Tracklet_ID",
        "Shadow_Decision_ID",
    }
    matches: list[str] = []
    for path in sorted(reviewer_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".html", ".json", ".csv", ".txt", ".js", ".css"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden_tokens:
            if token in content:
                matches.append(f"{path.relative_to(reviewer_dir).as_posix()}:{token}")
    manifest = json.loads((reviewer_dir / "review_manifest.json").read_text(encoding="utf-8"))
    mapping = pd.read_csv(private_mapping, dtype=str, keep_default_na=False)
    public_ids = [str(item.get("item_id") or "") for item in manifest.get("review_items", [])]
    private_ids = list(mapping.get("Item_ID", pd.Series(dtype=str)).astype(str))
    if len(public_ids) != len(set(public_ids)) or len(private_ids) != len(set(private_ids)):
        raise ForensicReviewError("Reviewer package contains duplicate item IDs")
    if set(public_ids) != set(private_ids):
        raise ForensicReviewError("Public reviewer and private provenance item IDs do not match")
    return {
        "review_items": len(public_ids),
        "private_items": len(private_ids),
        "forbidden_matches": matches,
        "privacy_passed": not matches,
        "private_mapping_loaded_by_browser": False,
    }


def load_forensic_review_package(package_dir: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    package_dir = Path(package_dir)
    manifest_path = package_dir / "package_manifest.json"
    try:
        verify_output_manifest(package_dir, manifest_path)
    except ShadowValidationError as exc:
        raise ForensicReviewError(str(exc)) from exc
    public_path = package_dir / "reviewer" / "review_manifest.json"
    metadata_path = package_dir / "private" / "export_metadata.json"
    mapping_path = package_dir / "private" / "source_mapping.csv"
    if not public_path.is_file() or not metadata_path.is_file() or not mapping_path.is_file():
        raise ForensicReviewError(f"Forensic review package is incomplete: {package_dir}")
    public = json.loads(public_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    package_ids = {str(public.get("package_id") or ""), str(metadata.get("package_id") or "")}
    if len(package_ids) != 1 or "" in package_ids:
        raise ForensicReviewError("Forensic review package IDs do not match")
    mapping_package_ids = set(mapping.get("Package_ID", pd.Series(dtype=str)).astype(str))
    if mapping_package_ids not in ({next(iter(package_ids))}, set()):
        raise ForensicReviewError("Private source mapping package ID does not match")
    audit = audit_forensic_review_package(package_dir)
    if not audit["privacy_passed"]:
        raise ForensicReviewError("Forensic reviewer privacy audit failed")
    return public, metadata, mapping


def validate_forensic_approvals(
    package_dir: Path,
    approvals_path: Path,
    *,
    expected_kind: str | None = None,
    verify_sources: bool = True,
) -> ApprovalValidationResult:
    public, metadata, mapping = load_forensic_review_package(package_dir)
    approvals_path = Path(approvals_path)
    if not approvals_path.is_file():
        raise ForensicReviewError(f"Approval CSV not found: {approvals_path}")
    approvals = pd.read_csv(approvals_path, dtype=str, keep_default_na=False)
    missing_columns = sorted(set(APPROVAL_COLUMNS).difference(approvals.columns))
    if missing_columns:
        raise ForensicReviewError("Approval CSV missing columns: " + ", ".join(missing_columns))
    approvals = approvals[APPROVAL_COLUMNS].copy()
    if approvals["Item_ID"].duplicated().any():
        duplicates = sorted(approvals.loc[approvals["Item_ID"].duplicated(False), "Item_ID"].unique())
        raise ForensicReviewError("Approval CSV contains duplicate item IDs: " + ", ".join(duplicates[:10]))
    package_id = str(public["package_id"])
    if set(approvals["Package_ID"].astype(str)) != ({package_id} if len(approvals) else set()):
        raise ForensicReviewError("Approval CSV package ID does not match the review package")
    expected_ids = {str(item.get("item_id") or "") for item in public.get("review_items", [])}
    actual_ids = set(approvals["Item_ID"].astype(str))
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ForensicReviewError(
            f"Approval item set mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    kind = str(public.get("package_kind") or "")
    if expected_kind and kind != expected_kind:
        raise ForensicReviewError(f"Expected {expected_kind} approvals, found {kind}")
    allowed_actions = set(dict(public.get("actions") or {}))
    invalid_actions = sorted(set(approvals["Review_Action"].astype(str)) - allowed_actions)
    if invalid_actions or approvals["Review_Action"].eq("").any():
        raise ForensicReviewError(
            "Approval CSV contains missing or invalid actions: " + ", ".join(invalid_actions or ["<blank>"])
        )
    duplicate_action = (
        "duplicate_of_another_image"
        if "duplicate_of_another_image" in allowed_actions
        else ""
    )
    for row in approvals.to_dict("records"):
        duplicate_target = str(row.get("Duplicate_Of_Item_ID") or "").strip()
        if duplicate_action and row["Review_Action"] == duplicate_action:
            if duplicate_target not in expected_ids or duplicate_target == row["Item_ID"]:
                raise ForensicReviewError(
                    f"{row['Item_ID']} must reference a different valid duplicate item ID"
                )
        elif duplicate_target:
            raise ForensicReviewError(
                f"{row['Item_ID']} supplies Duplicate_Of_Item_ID without the duplicate action"
            )
    if verify_sources:
        failures: list[str] = []
        for row in mapping.to_dict("records"):
            source = Path(str(row.get("Source_Path") or ""))
            if not source.is_file() or _sha256_file(source) != str(row.get("Source_SHA256") or ""):
                failures.append(str(row.get("Item_ID") or source))
        if failures:
            raise ForensicReviewError(
                "Approval source integrity check failed: " + ", ".join(failures[:10])
            )
    summary = {
        "schema_version": FORENSIC_REVIEW_SCHEMA_VERSION,
        "package_id": package_id,
        "package_kind": kind,
        "reviewed_items": len(approvals),
        "complete": len(approvals) == len(expected_ids),
        "action_counts": {
            str(key): int(value)
            for key, value in approvals["Review_Action"].value_counts().sort_index().items()
        },
        "approvals_path": str(approvals_path.resolve()),
        "approvals_sha256": _sha256_file(approvals_path),
        "sources_verified": bool(verify_sources),
        "source_files_changed": False,
        "production_embeddings_changed": False,
    }
    return ApprovalValidationResult(approvals=approvals, summary=summary, metadata=metadata)


def write_reviewer_launcher(
    *,
    launcher_path: Path,
    python_executable: Path,
    cli_script: Path,
    command: str,
    review_package: Path,
) -> Path:
    def quote(value: Any) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    launcher_path = Path(launcher_path)
    launcher_path.write_text(
        f"& {quote(Path(python_executable).resolve())} {quote(Path(cli_script).resolve())} "
        f"{command} --review-package {quote(Path(review_package).resolve())}\n",
        encoding="utf-8",
    )
    return launcher_path


def serve_forensic_review_package(
    review_package: Path,
    *,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    review_package = Path(review_package)
    load_forensic_review_package(review_package)
    try:
        server, url = create_reviewer_server(review_package, port=port)
    except ShadowValidationError as exc:
        raise ForensicReviewError(str(exc)) from exc
    print(f"Forensic reviewer URL: {url}")
    print(f"PID: {os.getpid()}")
    print("Server binding: 127.0.0.1 only")
    print("Clean shutdown: press Ctrl+C in this terminal")
    if open_browser and not webbrowser.open(url):
        print("Browser did not open automatically; open the URL shown above.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReviewer server stopped.")
    finally:
        server.server_close()


def timed_validate_forensic_approvals(
    package_dir: Path,
    approvals_path: Path,
    *,
    expected_kind: str | None = None,
) -> ApprovalValidationResult:
    started = time.perf_counter()
    result = validate_forensic_approvals(
        package_dir,
        approvals_path,
        expected_kind=expected_kind,
    )
    print(f"Approval validation elapsed: {time.perf_counter() - started:.2f} seconds")
    return result
