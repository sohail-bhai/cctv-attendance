# Product Phase 2K-C New Session Capture Guide

This procedure creates an immutable source freeze for a genuinely new CVO/B51 session. It validates provenance, bytes, layout, and container headers only. It does not authorize or run recognition.

## 1. Existing sessions are invalid

Do not use MON P3, MON P4, TUE P1, or TUE P2. The session IDs `2026-06-22__B51__P3__CVO`, `2026-06-22__B51__P4__CVO`, `2026-06-30__B51__P1__CVO`, and `2026-06-30__B51__P2__CVO` are verified contaminated evidence. Renaming or copying one of those clips does not make it untouched.

## 2. Untouched means never previously used

Every clip and the complete session must have had no recognition, human review, model selection, threshold tuning, calibration, source ablation, promotion evidence, retention testing, Phase 1.2N processing, Phase 2G/2I processing, or reviewed-benchmark use. There is no operator override for a matching historical hash, historical session ID, or unverifiable provenance.

## 3. Required source layout

Prepare exactly ten original video files. Use exactly one `back` and one `front` clip inside each `CP1`, `CP2`, `CP3`, `CP4`, and `CP5` directory. Supported extensions are `.mp4`, `.avi`, `.mov`, and `.mkv`. Do not add thumbnails, sidecars, preview files, or extra directories to the source root.

```text
new-session-source/
  CP1/back.mp4  CP1/front.mp4
  CP2/back.mp4  CP2/front.mp4
  CP3/back.mp4  CP3/front.mp4
  CP4/back.mp4  CP4/front.mp4
  CP5/back.mp4  CP5/front.mp4
```

## 4. Do not preview or process before freezing

Do not open the clips in the browser workflow, click **Process Attendance**, run a recognition script, export face crops, create tracklets, or use tools that decode frames before the source freeze is complete. The intake tool opens container headers and reads bytes for hashing; it never calls `read()` on video frames.

## 5. Preserve originals

Copy the camera originals into the source layout without transcoding, trimming, concatenating, recompressing, normalizing timestamps, or changing the original files. The intake tool copies into a private staging directory, hashes the source and staged copy independently, and rejects any change or mismatch.

## 6. Record exact metadata

Before running the tool, obtain the session date, section, period, subject, room, and one timezone-qualified ISO-8601 capture-start timestamp for every clip. Do not estimate missing capture timestamps. The operator timestamps are required even though filesystem creation and modification timestamps are recorded separately.

## 7. Run a no-write dry run first

From the repository root, supply all ten capture timestamps. This example uses the directory layout; replace every placeholder with exact values.

```powershell
venv\Scripts\python.exe scripts\create_product_phase_2k_c_capture_package.py --repo-root . create `
  --source-dir "D:\restricted-capture\new-session-source" `
  --output-root "D:\restricted-capture\phase-2k-c-freezes" `
  --session-date "YYYY-MM-DD" --section "B51" --period "P4" --subject "CVO" --room "B51" `
  --capture-timestamp "CP1/back=YYYY-MM-DDThh:mm:ss+05:30" --capture-timestamp "CP1/front=YYYY-MM-DDThh:mm:ss+05:30" `
  --capture-timestamp "CP2/back=YYYY-MM-DDThh:mm:ss+05:30" --capture-timestamp "CP2/front=YYYY-MM-DDThh:mm:ss+05:30" `
  --capture-timestamp "CP3/back=YYYY-MM-DDThh:mm:ss+05:30" --capture-timestamp "CP3/front=YYYY-MM-DDThh:mm:ss+05:30" `
  --capture-timestamp "CP4/back=YYYY-MM-DDThh:mm:ss+05:30" --capture-timestamp "CP4/front=YYYY-MM-DDThh:mm:ss+05:30" `
  --capture-timestamp "CP5/back=YYYY-MM-DDThh:mm:ss+05:30" --capture-timestamp "CP5/front=YYYY-MM-DDThh:mm:ss+05:30" `
  --confirm "CREATE_PHASE_2K_C_SOURCE_FREEZE" --dry-run
```

Dry-run validates paths, exact metadata, the proposed deterministic package ID, current production policy, and historical contamination. It copies no file and creates no output directory.

## 8. Create the immutable freeze

Run the same command without `--dry-run`. The output is created through a private staging directory and one atomic directory rename. Existing packages are never overwritten. Repeating an identical request verifies and reuses the byte-identical package; any deterministic-ID collision fails closed.

The PowerShell wrapper accepts the same values:

```powershell
& scripts\create_product_phase_2k_c_capture_package.ps1 -Command create `
  -SourceDir "D:\restricted-capture\new-session-source" -OutputRoot "D:\restricted-capture\phase-2k-c-freezes" `
  -SessionDate "YYYY-MM-DD" -Section "B51" -Period "P4" -Subject "CVO" -Room "B51" `
  -CaptureTimestamp @("CP1/back=...", "CP1/front=...", "CP2/back=...", "CP2/front=...", "CP3/back=...", "CP3/front=...", "CP4/back=...", "CP4/front=...", "CP5/back=...", "CP5/front=...") `
  -Confirm "CREATE_PHASE_2K_C_SOURCE_FREEZE"
```

## 9. Understand fail-closed reason codes

The tool emits one machine-readable `reason_code`. Important outcomes include `duplicate_inside_package`, `known_contaminated_source`, `duplicate_session_id`, and `unverifiable_provenance`. Other failures include missing or extra bindings, incomplete metadata, unsupported containers, source changes during copy, staged-copy mismatch, production-policy drift, tampering, and deterministic-ID collisions. Do not edit a package or bypass a failure; correct the source-side problem or acquire a new session.

## 10. Verify with both validators

Run the intake verifier and the canonical Phase 2K-B capture validator against the completed package:

```powershell
venv\Scripts\python.exe scripts\create_product_phase_2k_c_capture_package.py --repo-root . verify --package-dir "D:\restricted-capture\phase-2k-c-freezes\capture-package-..."
venv\Scripts\python.exe scripts\run_product_phase_2k_carry_forward.py validate-capture --package-dir "D:\restricted-capture\phase-2k-c-freezes\capture-package-..."
```

Both commands hash files and validate provenance/configuration. Neither command decodes frames or runs recognition.

## 11. Upload the complete package only

Transfer the complete `capture-package-*` directory, including all ten files under `videos`, every JSON/CSV contract file, and `immutable_manifest.json`. Do not upload loose clips, partial folders, edited manifests, regenerated CSV files, or a package missing the restricted operator record. Use authenticated encrypted transport and access-controlled encrypted storage outside public or web roots.

## 12. Validation is not recognition authorization

A passing package proves only that preparation and source integrity passed. It does not authorize YuNet, SFace, frame decoding, attendance processing, report creation, parameter tuning, authority changes, activation, or promotion. A separate explicit authorization is required before recognition.

## 13. Do not use browser Process Attendance

Do not upload the frozen clips into the normal browser processing flow and do not select **Process Attendance**. Phase 2K-C execution requires a separately controlled runner and a pre-frozen blind-review/evaluation plan. The C0 intake package deliberately exposes no execution path.

## 14. Privacy, storage, and lifecycle

Treat video, original absolute source paths, future identity evidence, and future appearance-similarity matrices as restricted. Keep them outside frontend/public roots. Do not store credentials in the package or guide. Raw observation embeddings are not part of intake and must not be added. Follow the Phase 2K-B retention contract for any later derived diagnostic evidence and record approved deletion evidence.

## 15. Staging recovery and immutable-package handling

The tool automatically removes its narrowly named `.capture-package-*.staging-*` directory after a failed create. If an operating-system interruption leaves one behind, first confirm no intake process is running, resolve the exact staging path under the selected output root, and remove only that incomplete staging directory. Never remove or edit a completed `capture-package-*` directory in place. If a completed package is invalid or no longer needed, preserve it for audit or move the entire directory through an approved restricted-storage lifecycle; create a new package from unchanged originals rather than repairing manifests.
