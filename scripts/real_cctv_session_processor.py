from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Iterable

from src.face_attendance.session_identity import build_attendance_session_id

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
TIMESTAMP_RE = re.compile(r'(20\d{12})')
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REAL_ROOT = ROOT_DIR / 'cctv_videos' / 'real_cctv'
DEFAULT_PREPARED_ROOT = ROOT_DIR / 'cctv_videos' / 'prepared_slots'
DEFAULT_CURRENT_SLOT_ROOT = ROOT_DIR / 'cctv_videos'
DEFAULT_DATA_DIR = ROOT_DIR / 'data'
DEFAULT_OUTPUT_DIR = ROOT_DIR / 'attendance_output'
DEFAULT_TIMETABLE = ROOT_DIR / 'timetable_b51_2026_2027.csv'


@dataclass
class VideoChunk:
    date: str
    camera: str
    file_path: str
    start_dt: datetime
    end_dt: datetime
    duration_seconds: float

    def to_row(self) -> dict:
        return {
            'date': self.date,
            'camera': self.camera,
            'file_path': self.file_path,
            'start_datetime': self.start_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'end_datetime': self.end_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'start_time': self.start_dt.strftime('%H:%M:%S'),
            'end_time': self.end_dt.strftime('%H:%M:%S'),
            'duration_seconds': round(self.duration_seconds, 2),
        }


@dataclass
class TimetablePeriod:
    slot_id: str
    day: str
    period: str
    period_number: int
    start_time: str
    end_time: str
    course_abbr: str
    course_name: str
    course_code: str
    instructor: str
    room: str
    section: str
    session_type: str

    def subjects(self) -> list[str]:
        return [p.strip().upper() for p in self.course_abbr.split('/') if p.strip()]


@dataclass
class SessionGroup:
    session_id: str
    day: str
    date: str
    periods: list[TimetablePeriod]
    start_time: str
    end_time: str
    course_abbr: str
    course_name: str
    subject_views: list[str]
    room: str
    section: str

    def to_row(self) -> dict:
        return {
            'session_id': self.session_id,
            'date': self.date,
            'day': self.day,
            'periods': ','.join(p.period for p in self.periods),
            'slot_ids': ','.join(p.slot_id for p in self.periods),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'course_abbr': self.course_abbr,
            'course_name': self.course_name,
            'subject_views': ','.join(self.subject_views),
            'room': self.room,
            'section': self.section,
        }


def safe_print(msg: str) -> None:
    print(str(msg).encode('utf-8', errors='replace').decode('utf-8', errors='replace'), flush=True)


def parse_dt_from_filename(path: Path) -> tuple[datetime, datetime] | None:
    stamps = TIMESTAMP_RE.findall(path.name)
    if len(stamps) < 2:
        return None
    try:
        start = datetime.strptime(stamps[-2], '%Y%m%d%H%M%S')
        end = datetime.strptime(stamps[-1], '%Y%m%d%H%M%S')
        if end <= start:
            return None
        return start, end
    except ValueError:
        return None


def parse_hhmm(value: str) -> time:
    value = str(value or '').strip()
    for fmt in ('%H:%M:%S', '%H:%M'):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            pass
    raise ValueError(f'Invalid time: {value}')


def combine_dt(day: date, hhmm: str) -> datetime:
    return datetime.combine(day, parse_hhmm(hhmm))


def date_day_name(date_text: str) -> str:
    d = datetime.strptime(date_text, '%Y-%m-%d').date()
    return DAY_NAMES[d.weekday()]


def period_num(value: str | int) -> int:
    raw = str(value).strip().upper().replace('P', '')
    return int(raw)


def slot_id_for(day: str, period: str | int) -> str:
    return f'{day[:3].upper()}_P{period_num(period)}'


def normalize_day(day: str) -> str:
    d = str(day or '').strip().lower()
    mapping = {
        'mon': 'Monday', 'monday': 'Monday',
        'tue': 'Tuesday', 'tuesday': 'Tuesday',
        'wed': 'Wednesday', 'wednesday': 'Wednesday',
        'thu': 'Thursday', 'thursday': 'Thursday',
        'fri': 'Friday', 'friday': 'Friday',
        'sat': 'Saturday', 'saturday': 'Saturday',
        'sun': 'Sunday', 'sunday': 'Sunday',
    }
    return mapping.get(d[:3], str(day).title())


def read_timetable(path: Path) -> list[TimetablePeriod]:
    if not path.exists():
        raise FileNotFoundError(f'Timetable not found: {path}')
    rows: list[TimetablePeriod] = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = normalize_day(row.get('day') or row.get('Day') or '')
            period_raw = row.get('period') or row.get('Period') or ''
            if not day or not period_raw:
                continue
            period = f'P{period_num(period_raw)}'
            start = (row.get('start_time') or row.get('Start_Time') or '').strip()
            end = (row.get('end_time') or row.get('End_Time') or '').strip()
            if not start or not end:
                continue
            course_abbr = (row.get('course_abbr') or row.get('Course_Abbr') or row.get('subject') or row.get('Subject') or '').strip().upper()
            if not course_abbr or 'LUNCH' in course_abbr.upper():
                continue
            rows.append(TimetablePeriod(
                slot_id=(row.get('slot_id') or row.get('Slot_ID') or slot_id_for(day, period)).strip(),
                day=day,
                period=period,
                period_number=period_num(period),
                start_time=start,
                end_time=end,
                course_abbr=course_abbr,
                course_name=(row.get('course_name') or row.get('Course_Name') or course_abbr).strip(),
                course_code=(row.get('course_code') or row.get('Course_Code') or '-').strip(),
                instructor=(row.get('instructor') or row.get('Instructor') or row.get('teacher') or '-').strip(),
                room=(row.get('room') or row.get('Room') or '-').strip(),
                section=(row.get('section') or row.get('Section') or 'B51').strip(),
                session_type=(row.get('session_type') or row.get('Session_Type') or 'Theory').strip(),
            ))
    return sorted(rows, key=lambda r: (r.day, r.period_number))


def group_sessions(periods: list[TimetablePeriod], date_text: str) -> list[SessionGroup]:
    if not periods:
        return []
    sessions: list[SessionGroup] = []
    current: list[TimetablePeriod] = [periods[0]]

    def same_session(a: TimetablePeriod, b: TimetablePeriod) -> bool:
        # Consecutive periods with same subject track are one teaching session.
        # This handles P1+P2, P3+P4, lab doubles, etc.
        if b.period_number != a.period_number + 1:
            return False
        if a.course_abbr != b.course_abbr:
            return False
        return True

    for row in periods[1:]:
        if same_session(current[-1], row):
            current.append(row)
        else:
            sessions.append(make_session(current, date_text, len(sessions) + 1))
            current = [row]
    sessions.append(make_session(current, date_text, len(sessions) + 1))
    return sessions


def make_session(periods: list[TimetablePeriod], date_text: str, idx: int) -> SessionGroup:
    first, last = periods[0], periods[-1]
    subject_clean = first.course_abbr.replace('/', '_')
    session_id = f'{first.day[:3].upper()}_S{idx}_{periods[0].period}_{periods[-1].period}_{subject_clean}'
    views = []
    for p in periods:
        for s in p.subjects():
            if s not in views:
                views.append(s)
    return SessionGroup(
        session_id=session_id,
        day=first.day,
        date=date_text,
        periods=periods,
        start_time=first.start_time,
        end_time=last.end_time,
        course_abbr=first.course_abbr,
        course_name=first.course_name,
        subject_views=views,
        room=first.room,
        section=first.section,
    )


def scan_real_cctv(date_text: str, real_root: Path = DEFAULT_REAL_ROOT) -> list[VideoChunk]:
    date_dir = real_root / date_text
    if not date_dir.exists():
        raise FileNotFoundError(f'Real CCTV date folder not found: {date_dir}')
    chunks: list[VideoChunk] = []
    for camera_dir in sorted([p for p in date_dir.iterdir() if p.is_dir()]):
        camera = camera_dir.name.lower().strip()
        for file in sorted(camera_dir.iterdir()):
            if not file.is_file() or file.suffix.lower() not in VIDEO_EXTS:
                continue
            parsed = parse_dt_from_filename(file)
            if not parsed:
                safe_print(f'WARNING: Could not read timestamps from filename: {file.name}')
                continue
            start_dt, end_dt = parsed
            if start_dt.strftime('%Y-%m-%d') != date_text:
                safe_print(f'WARNING: File date mismatch for {file.name}: {start_dt.date()}')
            chunks.append(VideoChunk(
                date=date_text,
                camera=camera,
                file_path=str(file),
                start_dt=start_dt,
                end_dt=end_dt,
                duration_seconds=(end_dt - start_dt).total_seconds(),
            ))
    return sorted(chunks, key=lambda c: (c.camera, c.start_dt, c.file_path))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows and not fieldnames:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_manifest(date_text: str, chunks: list[VideoChunk], data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    rows = [c.to_row() for c in chunks]
    path = data_dir / f'real_cctv_timeline_{date_text}.csv'
    write_csv(path, rows)
    latest = data_dir / 'real_cctv_timeline_latest.csv'
    write_csv(latest, rows)
    return path


def find_chunk_for_window(chunks: list[VideoChunk], camera: str, start: datetime, end: datetime) -> VideoChunk | None:
    cam_chunks = [c for c in chunks if c.camera == camera]
    # Perfect: full window inside chunk.
    for c in cam_chunks:
        if c.start_dt <= start and c.end_dt >= end:
            return c
    # Accept: checkpoint start inside chunk, but window may run past boundary.
    for c in cam_chunks:
        if c.start_dt <= start < c.end_dt:
            return c
    return None


def run_ffmpeg_cut(src: Path, dst: Path, offset_seconds: float, duration_seconds: float, reencode: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if reencode:
        command = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', f'{offset_seconds:.3f}', '-i', str(src), '-t', f'{duration_seconds:.3f}',
            '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', str(dst),
        ]
    else:
        command = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', f'{offset_seconds:.3f}', '-i', str(src), '-t', f'{duration_seconds:.3f}',
            '-c', 'copy', str(dst),
        ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        if not reencode:
            # Some MP4s cannot be accurately stream-copied. Try re-encode fallback.
            run_ffmpeg_cut(src, dst, offset_seconds, duration_seconds, reencode=True)
            return
        raise RuntimeError(f'ffmpeg cut failed for {src.name}: {result.stderr.strip()}')


def period_checkpoints(period: TimetablePeriod, real_date: date, settle_minutes: float = 10, every_minutes: float = 10, clip_seconds: float = 20) -> list[tuple[int, datetime, datetime, datetime]]:
    start = combine_dt(real_date, period.start_time)
    end = combine_dt(real_date, period.end_time)
    points: list[datetime] = []
    point = start + timedelta(minutes=settle_minutes)
    while point < end:
        points.append(point)
        point += timedelta(minutes=every_minutes)
    if not points or points[-1] != end:
        points.append(end)
    checkpoints = []
    for i, cp in enumerate(points, start=1):
        # Final checkpoint clip ends at period end to avoid capturing next period movement.
        if cp >= end:
            window_start = max(start, end - timedelta(seconds=clip_seconds))
        else:
            window_start = cp
        window_end = min(window_start + timedelta(seconds=clip_seconds), end)
        checkpoints.append((i, cp, window_start, window_end))
    return checkpoints


def prepare_period(period: TimetablePeriod, date_text: str, chunks: list[VideoChunk], prepared_root: Path, current_slot_root: Path | None = None, clip_seconds: float = 20, settle_minutes: float = 10, every_minutes: float = 10, reencode: bool = False) -> dict:
    real_date = datetime.strptime(date_text, '%Y-%m-%d').date()
    cameras = sorted({c.camera for c in chunks})
    slot_dir = prepared_root / date_text / period.slot_id
    if slot_dir.exists():
        shutil.rmtree(slot_dir)
    slot_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    planned = period_checkpoints(period, real_date, settle_minutes, every_minutes, clip_seconds)
    available_cp = 0
    extracted = 0
    skipped = 0
    for cp_index, cp_time, window_start, window_end in planned:
        cp_name = f'CP{cp_index}_{cp_time.strftime("%H%M")}'
        cp_dir = slot_dir / cp_name
        cp_has_any = False
        for camera in cameras:
            chunk = find_chunk_for_window(chunks, camera, window_start, window_end)
            if not chunk:
                rows.append({
                    'date': date_text,
                    'slot_id': period.slot_id,
                    'period': period.period,
                    'checkpoint': cp_name,
                    'checkpoint_time': cp_time.strftime('%H:%M:%S'),
                    'camera': camera,
                    'status': 'skipped_no_footage',
                    'source_file': '',
                    'output_file': '',
                    'window_start': window_start.strftime('%H:%M:%S'),
                    'window_end': window_end.strftime('%H:%M:%S'),
                    'offset_seconds': '',
                    'duration_seconds': '',
                })
                skipped += 1
                continue
            clip_start = max(window_start, chunk.start_dt)
            clip_end = min(window_end, chunk.end_dt)
            duration = max(0.0, (clip_end - clip_start).total_seconds())
            if duration < 5:
                rows.append({
                    'date': date_text,
                    'slot_id': period.slot_id,
                    'period': period.period,
                    'checkpoint': cp_name,
                    'checkpoint_time': cp_time.strftime('%H:%M:%S'),
                    'camera': camera,
                    'status': 'skipped_too_short',
                    'source_file': chunk.file_path,
                    'output_file': '',
                    'window_start': window_start.strftime('%H:%M:%S'),
                    'window_end': window_end.strftime('%H:%M:%S'),
                    'offset_seconds': '',
                    'duration_seconds': round(duration, 2),
                })
                skipped += 1
                continue
            dst = cp_dir / f'{camera}.mp4'
            offset = (clip_start - chunk.start_dt).total_seconds()
            run_ffmpeg_cut(Path(chunk.file_path), dst, offset, duration, reencode=reencode)
            cp_has_any = True
            extracted += 1
            rows.append({
                'date': date_text,
                'slot_id': period.slot_id,
                'period': period.period,
                'checkpoint': cp_name,
                'checkpoint_time': cp_time.strftime('%H:%M:%S'),
                'camera': camera,
                'status': 'prepared',
                'source_file': chunk.file_path,
                'output_file': str(dst),
                'window_start': window_start.strftime('%H:%M:%S'),
                'window_end': window_end.strftime('%H:%M:%S'),
                'offset_seconds': round(offset, 2),
                'duration_seconds': round(duration, 2),
            })
        if cp_has_any:
            available_cp += 1
    prep_csv = DEFAULT_DATA_DIR / f'prepared_real_slot_{date_text}_{period.slot_id}.csv'
    write_csv(prep_csv, rows)

    current_dir = None
    if current_slot_root:
        current_dir = current_slot_root / period.slot_id
        if current_dir.exists():
            shutil.rmtree(current_dir)
        if slot_dir.exists():
            shutil.copytree(slot_dir, current_dir)
    return {
        'slot_id': period.slot_id,
        'period': period.period,
        'subject': period.course_abbr,
        'start_time': period.start_time,
        'end_time': period.end_time,
        'planned_checkpoints': len(planned),
        'available_checkpoints': available_cp,
        'camera_count': len(cameras),
        'clips_extracted': extracted,
        'skipped_camera_windows': skipped,
        'prepared_dir': str(slot_dir),
        'current_slot_dir': str(current_dir) if current_dir else '',
        'manifest_csv': str(prep_csv),
    }


def periods_for_date(timetable: list[TimetablePeriod], date_text: str) -> list[TimetablePeriod]:
    day = date_day_name(date_text)
    return [p for p in timetable if p.day == day]


def select_session(sessions: list[SessionGroup], session_id: str | None) -> SessionGroup:
    if not session_id:
        if not sessions:
            raise ValueError('No sessions available')
        return sessions[0]
    wanted = session_id.strip().upper()
    for session in sessions:
        if session.session_id.upper() == wanted:
            return session
    raise ValueError(f'Session not found: {session_id}')


def find_period(periods: list[TimetablePeriod], slot_id: str) -> TimetablePeriod:
    wanted = slot_id.strip().upper()
    for p in periods:
        if p.slot_id.upper() == wanted:
            return p
    raise ValueError(f'Slot not found for this date/day: {slot_id}')


def scan_command(args) -> None:
    chunks = scan_real_cctv(args.date, Path(args.real_root))
    manifest = save_manifest(args.date, chunks, Path(args.data_dir))
    safe_print(f'Found {len(chunks)} CCTV chunk(s) for {args.date}')
    for chunk in chunks:
        safe_print(f'{chunk.camera:10s} {chunk.start_dt.strftime("%H:%M:%S")} - {chunk.end_dt.strftime("%H:%M:%S")}  {Path(chunk.file_path).name}')
    safe_print(f'Manifest saved: {manifest}')


def sessions_command(args) -> None:
    timetable = read_timetable(Path(args.timetable))
    periods = periods_for_date(timetable, args.date)
    sessions = group_sessions(periods, args.date)
    rows = [s.to_row() for s in sessions]
    out = Path(args.data_dir) / f'real_cctv_sessions_{args.date}.csv'
    write_csv(out, rows)
    safe_print(f'Sessions for {args.date} ({date_day_name(args.date)}):')
    for s in sessions:
        safe_print(f'{s.session_id} | {s.start_time}-{s.end_time} | periods {",".join(p.period for p in s.periods)} | {s.course_abbr} | views {",".join(s.subject_views)}')
    safe_print(f'Session list saved: {out}')


def prepare_command(args) -> None:
    chunks = scan_real_cctv(args.date, Path(args.real_root))
    save_manifest(args.date, chunks, Path(args.data_dir))
    timetable = read_timetable(Path(args.timetable))
    periods = periods_for_date(timetable, args.date)
    targets: list[TimetablePeriod] = []
    if args.slot_id:
        targets = [find_period(periods, args.slot_id)]
    elif args.session_id:
        session = select_session(group_sessions(periods, args.date), args.session_id)
        targets = session.periods
    else:
        targets = periods
    summaries = []
    for period in targets:
        safe_print(f'Preparing {period.slot_id} {period.course_abbr} {period.start_time}-{period.end_time} ...')
        summaries.append(prepare_period(
            period=period,
            date_text=args.date,
            chunks=chunks,
            prepared_root=Path(args.prepared_root),
            current_slot_root=Path(args.current_slot_root) if args.also_current_slot_folder else None,
            clip_seconds=float(args.checkpoint_clip_seconds),
            settle_minutes=float(args.settle_minutes),
            every_minutes=float(args.checkpoint_every_minutes),
            reencode=bool(args.reencode),
        ))
    out = Path(args.data_dir) / f'real_cctv_prepared_summary_{args.date}.csv'
    write_csv(out, summaries)
    safe_print('Prepared slot summary:')
    for s in summaries:
        safe_print(f"{s['slot_id']}: {s['clips_extracted']} clips, {s['available_checkpoints']}/{s['planned_checkpoints']} checkpoints, dir={s['prepared_dir']}")
    safe_print(f'Summary saved: {out}')


def process_command(args) -> None:
    # Prepare first, then run the existing checkpoint attendance engine for each target period.
    chunks = scan_real_cctv(args.date, Path(args.real_root))
    save_manifest(args.date, chunks, Path(args.data_dir))
    timetable = read_timetable(Path(args.timetable))
    periods = periods_for_date(timetable, args.date)
    if args.slot_id:
        targets = [find_period(periods, args.slot_id)]
    elif args.session_id:
        targets = select_session(group_sessions(periods, args.date), args.session_id).periods
    else:
        targets = periods
    results = []
    for period in targets:
        summary = prepare_period(
            period=period,
            date_text=args.date,
            chunks=chunks,
            prepared_root=Path(args.prepared_root),
            current_slot_root=None,
            clip_seconds=float(args.checkpoint_clip_seconds),
            settle_minutes=float(args.settle_minutes),
            every_minutes=float(args.checkpoint_every_minutes),
            reencode=bool(args.reencode),
        )
        video_dir = Path(summary['prepared_dir'])
        available = int(summary['available_checkpoints'])
        if available <= 0:
            safe_print(f'SKIP {period.slot_id}: no checkpoint footage available.')
            results.append({**summary, 'attendance_status': 'skipped_no_footage', 'return_code': ''})
            continue
        # Dynamic rule: with partial footage, require majority of available checkpoints.
        present_rule = max(1, min(3, (available // 2) + 1)) if available >= 3 else max(1, available)
        review_rule = max(1, present_rule - 1)
        strong_rule = max(present_rule, available)
        subject_track = str(getattr(args, 'subject_track', '') or '').strip().upper()
        session_id = ''
        if subject_track:
            session_id = build_attendance_session_id(args.date, period.section, period.period, subject_track)
        command = [
            sys.executable,
            str(ROOT_DIR / 'scripts' / 'mark_attendance_checkpoints.py'),
            '--timetable', str(Path(args.timetable)),
            '--slot-id', period.slot_id,
            '--video-dir', str(video_dir),
            '--checkpoint-mode', 'clip-folders',
            '--sample-fps', str(args.sample_fps),
            '--log-mode', str(args.log_mode),
            '--output-layout', 'organized',
            '--match-threshold', str(args.match_threshold),
            '--margin-threshold', str(args.margin_threshold),
            '--checkpoint-min-detections', str(args.checkpoint_min_detections),
            '--present-checkpoints', str(present_rule),
            '--strong-checkpoints', str(strong_rule),
            '--review-checkpoints', str(review_rule),
            '--settle-minutes', str(args.settle_minutes),
            '--checkpoint-every-minutes', str(args.checkpoint_every_minutes),
            '--checkpoint-clip-seconds', str(args.checkpoint_clip_seconds),
            '--aggregate', str(args.aggregate),
            '--session-date', str(args.date),
            '--input-slot', period.slot_id,
            '--input-source-type', 'prepared_slot',
            '--input-source-path', str(video_dir),
        ]
        if subject_track:
            command.extend(['--session-id', session_id, '--subject-abbr', subject_track])
        safe_print(f'Running attendance for {period.slot_id} using {available}/{summary["planned_checkpoints"]} available checkpoints...')
        env = dict(**{k: v for k, v in __import__('os').environ.items()}, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        proc = subprocess.run(command, cwd=str(ROOT_DIR), text=True, encoding='utf-8', errors='replace', env=env)
        results.append({**summary, 'attendance_status': 'completed' if proc.returncode == 0 else 'failed', 'return_code': proc.returncode})
        if proc.returncode != 0 and not args.continue_on_error:
            raise RuntimeError(f'Attendance failed for {period.slot_id}')
    out = Path(args.data_dir) / f'real_cctv_process_summary_{args.date}.csv'
    write_csv(out, results)
    safe_print(f'Process summary saved: {out}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Real CCTV timeline/session processor for Sreenidhi Smart Attendance')
    sub = parser.add_subparsers(dest='command', required=True)

    def add_common(p):
        p.add_argument('--date', required=True, help='Date in YYYY-MM-DD, e.g. 2026-06-30')
        p.add_argument('--real-root', default=str(DEFAULT_REAL_ROOT), help='Root folder for raw real CCTV videos')
        p.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR), help='Folder to save manifests')

    p = sub.add_parser('scan', help='Scan raw CCTV chunks and build timeline manifest')
    add_common(p)
    p.set_defaults(func=scan_command)

    p = sub.add_parser('sessions', help='Show timetable sessions for the date')
    add_common(p)
    p.add_argument('--timetable', default=str(DEFAULT_TIMETABLE))
    p.set_defaults(func=sessions_command)

    p = sub.add_parser('prepare', help='Prepare checkpoint clips for a slot/session/date')
    add_common(p)
    p.add_argument('--timetable', default=str(DEFAULT_TIMETABLE))
    p.add_argument('--prepared-root', default=str(DEFAULT_PREPARED_ROOT))
    p.add_argument('--current-slot-root', default=str(DEFAULT_CURRENT_SLOT_ROOT))
    p.add_argument('--slot-id', default='')
    p.add_argument('--session-id', default='')
    p.add_argument('--also-current-slot-folder', action='store_true', help='Also copy prepared clips to cctv_videos/SLOT_ID for the existing frontend Process button')
    p.add_argument('--settle-minutes', type=float, default=10)
    p.add_argument('--checkpoint-every-minutes', type=float, default=10)
    p.add_argument('--checkpoint-clip-seconds', type=float, default=20)
    p.add_argument('--reencode', action='store_true')
    p.set_defaults(func=prepare_command)

    p = sub.add_parser('process', help='Prepare clips and run checkpoint attendance for a slot/session/date')
    add_common(p)
    p.add_argument('--timetable', default=str(DEFAULT_TIMETABLE))
    p.add_argument('--prepared-root', default=str(DEFAULT_PREPARED_ROOT))
    p.add_argument('--slot-id', default='')
    p.add_argument('--session-id', default='')
    p.add_argument('--settle-minutes', type=float, default=10)
    p.add_argument('--checkpoint-every-minutes', type=float, default=10)
    p.add_argument('--checkpoint-clip-seconds', type=float, default=20)
    p.add_argument('--sample-fps', type=float, default=2)
    p.add_argument('--log-mode', default='accepted', choices=['accepted', 'full', 'none'])
    p.add_argument('--match-threshold', type=float, default=0.48)
    p.add_argument('--margin-threshold', type=float, default=0.08)
    p.add_argument('--checkpoint-min-detections', type=int, default=2)
    p.add_argument('--aggregate', default='top3')
    p.add_argument('--subject-track', default='', help='Optional logical subject track, e.g. CVO or CCM')
    p.add_argument('--reencode', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    p.set_defaults(func=process_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
