# Sreenidhi Smart Attendance Admin - React Frontend

This is the improved React admin UI for the CCTV-Based Smart Attendance System.

## Run

```powershell
cd frontend
npm install
npm run dev
```

Backend should run separately:

```powershell
python app.py
```

Open `http://localhost:5173`.

## Best current workflow

1. Use **Camera Angles** to upload `cam1.mp4`, `cam2.mp4`, `cam3.mp4`.
2. Use **Timetable Control** to select the subject slot and start processing.
3. Use **Report Analyzer** to upload generated CSV files and inspect evidence.
4. Use **Manual Review** only for justified corrections.

## Notes

This frontend supports the older Flask API but is designed around the newer YuNet + SFace + timetable MVP 2 logic.
