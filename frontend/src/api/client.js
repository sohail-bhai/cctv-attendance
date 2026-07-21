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
    ...(user?.sessionToken ? { Authorization: `Bearer ${user.sessionToken}` } : {}),
  };
}

function messageFromPayload(payload, fallback) {
  if (payload && typeof payload === 'object') {
    return payload.error || payload.message || fallback;
  }
  return fallback;
}

async function readResponsePayload(response) {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json().catch(() => ({}));
  }
  const text = await response.text().catch(() => '');
  return text ? { message: text } : {};
}

export async function apiGetResult(path, options = {}) {
  const startedAt = Date.now();
  try {
    const response = await fetch(path, {
      ...options,
      method: 'GET',
      headers: authHeaders(options.headers || {}),
    });
    const data = await readResponsePayload(response);
    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        data: null,
        error: messageFromPayload(data, `${response.status} ${response.statusText}`),
        networkError: false,
        elapsedMs: Date.now() - startedAt,
      };
    }
    return {
      ok: true,
      status: response.status,
      data,
      error: null,
      networkError: false,
      elapsedMs: Date.now() - startedAt,
    };
  } catch (error) {
    console.warn(`[API GET failed] ${path}`, error.message);
    return {
      ok: false,
      status: 0,
      data: null,
      error: error.message || 'Backend request failed.',
      networkError: true,
      elapsedMs: Date.now() - startedAt,
    };
  }
}

export async function apiGet(path, fallback = null) {
  const result = await apiGetResult(path);
  return result.ok ? result.data : fallback;
}

export async function apiPost(path, payload = {}) {
  let response;
  try {
    response = await fetch(path, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    });
  } catch (error) {
    throw new Error(`Backend unavailable: ${error.message || 'network request failed'}`);
  }

  const data = await readResponsePayload(response);
  if (!response.ok || data.success === false) {
    throw new Error(messageFromPayload(data, `${response.status} ${response.statusText}`));
  }
  return data;
}

export async function apiCancelJob(jobId) {
  return apiPost(`/api/cancel/${encodeURIComponent(jobId)}`, {});
}

export async function apiUploadVideos(files, onProgress) {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append('video', file));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/videos/upload');
    const user = getStoredUser();
    if (user?.id) xhr.setRequestHeader('X-User-Id', user.id);
    if (user?.sessionToken) xhr.setRequestHeader('Authorization', `Bearer ${user.sessionToken}`);
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

export async function apiPostForm(path, formData) {
  let response;
  try {
    response = await fetch(path, {
      method: 'POST',
      headers: authHeaders(),
      body: formData,
    });
  } catch (error) {
    throw new Error(`Backend unavailable: ${error.message || 'network request failed'}`);
  }

  const data = await readResponsePayload(response);
  if (!response.ok || data.success === false) {
    throw new Error(messageFromPayload(data, `${response.status} ${response.statusText}`));
  }
  return data;
}

