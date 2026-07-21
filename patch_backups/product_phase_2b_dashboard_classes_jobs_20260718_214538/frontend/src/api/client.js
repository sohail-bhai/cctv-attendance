function getStoredUser() {
  try {
    return JSON.parse(localStorage.getItem('sreenidhi_attendance_user') || 'null');
  } catch {
    return null;
  }
}

function authHeaders(extra = {}) {
  const user = getStoredUser();
  return {
    ...extra,
    ...(user?.id ? { 'X-User-Id': user.id } : {}),
  };
}

export async function apiGet(path, fallback = null) {
  try {
    const response = await fetch(path, { headers: authHeaders() });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return await response.json();
  } catch (error) {
    console.warn(`[API GET failed] ${path}`, error.message);
    return fallback;
  }
}

export async function apiPost(path, payload = {}) {
  const response = await fetch(path, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.success === false) {
    throw new Error(data.error || data.message || `${response.status} ${response.statusText}`);
  }
  return data;
}

export async function apiCancelJob(jobId) {
  return apiPost(`/api/cancel/${jobId}`, {});
}

export async function apiUploadVideos(files, onProgress) {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append('video', file));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/videos/upload');
    const user = getStoredUser();
    if (user?.id) xhr.setRequestHeader('X-User-Id', user.id);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      try {
        const data = JSON.parse(xhr.responseText || '{}');
        if (xhr.status >= 200 && xhr.status < 300 && data.success !== false) resolve(data);
        else reject(new Error(data.error || data.message || 'Upload failed'));
      } catch (error) {
        reject(error);
      }
    };
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.send(formData);
  });
}
