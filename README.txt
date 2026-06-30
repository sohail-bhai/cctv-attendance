MVP 2 - Timetable-Based Attendance
==================================

Files included:
1. timetable_b51_2026_2027.csv
2. scripts/mark_attendance_timetable.py
3. src/face_attendance/timetable.py

Where to paste:
- Put timetable_b51_2026_2027.csv in your project root folder.
- Put mark_attendance_timetable.py inside your scripts/ folder.
- Put timetable.py inside src/face_attendance/ folder.

Main command example:
python scripts\mark_attendance_timetable.py --timetable timetable_b51_2026_2027.csv --slot-id MON_P1 --video-dir cctv_videos\test_video1 --display --save-unknown --match-threshold 0.48 --margin-threshold 0.08 --min-detections 15

Alternative slot selection:
python scripts\mark_attendance_timetable.py --timetable timetable_b51_2026_2027.csv --day Monday --period 1 --video-dir cctv_videos\test_video1 --match-threshold 0.48 --margin-threshold 0.08 --min-detections 15

Or by time:
python scripts\mark_attendance_timetable.py --timetable timetable_b51_2026_2027.csv --day Monday --class-time 09:10 --video-dir cctv_videos\test_video1 --match-threshold 0.48 --margin-threshold 0.08 --min-detections 15

To list timetable slots:
python scripts\mark_attendance_timetable.py --timetable timetable_b51_2026_2027.csv --list-slots

Outputs:
- attendance_SLOT_SUBJECT_DAY_PERIOD_TIMESTAMP.csv
- detection_log_SLOT_SUBJECT_DAY_PERIOD_TIMESTAMP.csv
- slot_summary_SLOT_SUBJECT_DAY_PERIOD_TIMESTAMP.csv
- attendance_SLOT_SUBJECT_DAY_PERIOD_TIMESTAMP.xlsx

Excel sheets:
- Slot Summary
- Attendance
- Detection Log

Status logic in MVP 2:
- Present: Detection_Count >= --min-detections
- Needs Review: Detection_Count >= --review-min-detections but below --min-detections
- Absent: Detection_Count below review minimum

Default review minimum is half of min-detections.
Example: if --min-detections 15, Needs Review starts around 7 detections.

MVP 3 will add checkpoint voting logic.
