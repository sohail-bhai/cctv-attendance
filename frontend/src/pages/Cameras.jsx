import { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';
import { apiGet, apiUploadVideos } from '../api/client.js';

function formatSize(bytes = 0) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / (1024 ** i)).toFixed(i ? 1 : 0)} ${units[i]}`;
}

export default function Cameras() {
  const [videos, setVideos] = useState([]);
  const [message, setMessage] = useState(null);
  const [progress, setProgress] = useState(0);

  const load = () => apiGet('/api/videos', { videos: [] }).then((data) => setVideos(data?.videos || []));
  useEffect(load, []);

  const upload = async (files) => {
    if (!files?.length) return;
    setMessage({ type: 'info', text: 'Uploading camera angle videos...' });
    setProgress(0);
    try {
      const data = await apiUploadVideos(files, setProgress);
      setMessage({ type: 'success', text: data.message || 'Videos uploaded.' });
      load();
    } catch (error) {
      setMessage({ type: 'error', text: error.message });
    }
  };

  const totalSize = videos.reduce((sum, item) => sum + (item.size || 0), 0);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Multi-camera Evidence"
        title="Camera Angle Manager"
        subtitle="Upload all camera-angle clips for the same class slot together. Example: cam1, cam2, cam3."
        actions={<label className="button">Upload Videos<input hidden multiple type="file" accept=".mp4,.avi,.mov,.mkv" onChange={(event) => upload(event.target.files)} /></label>}
      />

      {message && <div className={`notice ${message.type}`}>{message.text}{progress > 0 && progress < 100 ? ` · ${progress}%` : ''}</div>}

      <section className="stats-grid">
        <StatCard label="Uploaded videos" value={videos.length} hint="Each file = one camera angle" />
        <StatCard label="Storage used" value={formatSize(totalSize)} hint="Testing folder only" tone="info" />
        <StatCard label="Recommended naming" value="cam1.mp4" hint="cam2.mp4, cam3.mp4" tone="success" />
        <StatCard label="Max file size" value="1 GB" hint="Backend upload limit" tone="warning" />
      </section>

      <section className="panel">
        <div className="panel-title-row">
          <h3>Current video files</h3>
          <span className="muted">From /api/videos</span>
        </div>
        <div className="camera-grid">
          {videos.map((video, index) => (
            <article className="camera-card" key={video.name}>
              <span className="camera-number">cam{index + 1}</span>
              <strong>{video.name}</strong>
              <p>{formatSize(video.size)} · {video.modified ? new Date(video.modified).toLocaleString('en-IN') : 'No date'}</p>
            </article>
          ))}
          {videos.length === 0 && <p className="empty-text">No uploaded camera videos found.</p>}
        </div>
      </section>

      <section className="panel">
        <h3>Why camera-angle evidence matters</h3>
        <div className="tip-grid">
          <div><strong>Head down problem</strong><p>One camera may miss a student writing, but another angle or later checkpoint may capture the face.</p></div>
          <div><strong>Do not count raw frames blindly</strong><p>Students close to the camera get inflated detections. MVP 3 should use checkpoint voting.</p></div>
          <div><strong>Quality reporting</strong><p>Report must show detected faces, recognized faces, rejected faces, and unknown faces per camera.</p></div>
        </div>
      </section>
    </div>
  );
}
