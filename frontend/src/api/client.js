const STORAGE_KEY = 'sreenidhi_attendance_user';
export const AUTH_INVALID_EVENT = 'sreenidhi-auth-invalid';

function getStoredUser() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
  } catch {
    return null;
  }
}

export function authHeadersForUser(user, extra = {}) {
  if (!user?.sessionToken) return { ...extra };
  return {
    ...extra,
    Authorization: `Bearer ${user.sessionToken}`,
    ...(user?.id ? { 'X-User-Id': user.id } : {}),
  };
}

function authHeaders(extra = {}) {
  return authHeadersForUser(getStoredUser(), extra);
}

export function shouldInvalidateStoredSession(path, status) {
  return Number(status) === 401 && path !== '/api/auth/login';
}

function handleAuthenticationResponse(path, status) {
  if (!shouldInvalidateStoredSession(path, status)) return;
  localStorage.removeItem(STORAGE_KEY);
  window.dispatchEvent(new Event(AUTH_INVALID_EVENT));
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
    handleAuthenticationResponse(path, response.status);
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
  handleAuthenticationResponse(path, response.status);
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
    const headers = authHeadersForUser(user);
    Object.entries(headers).forEach(([name, value]) => xhr.setRequestHeader(name, value));
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      handleAuthenticationResponse('/api/videos/upload', xhr.status);
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
  handleAuthenticationResponse(path, response.status);
  if (!response.ok || data.success === false) {
    throw new Error(messageFromPayload(data, `${response.status} ${response.statusText}`));
  }
  return data;
}

export async function apiGetBlobResult(path) {
  try {
    const response = await fetch(path, {
      method: 'GET',
      headers: authHeaders(),
      cache: 'no-store',
    });
    handleAuthenticationResponse(path, response.status);
    if (!response.ok) {
      const data = await readResponsePayload(response);
      return { ok: false, status: response.status, blob: null, error: messageFromPayload(data, `${response.status} ${response.statusText}`) };
    }
    return { ok: true, status: response.status, blob: await response.blob(), error: null };
  } catch (error) {
    return { ok: false, status: 0, blob: null, error: error.message || 'Backend request failed.' };
  }
}

export async function apiDownload(path) {
  const result = await apiGetBlobResult(path);
  if (!result.ok) throw new Error(result.error || 'Download failed.');
  const objectUrl = URL.createObjectURL(result.blob);
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = decodeURIComponent(String(path).split('/').pop() || 'download');
  link.rel = 'noopener';
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
}
