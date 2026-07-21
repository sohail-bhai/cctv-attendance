export const SUBJECT_INFO = {
  "SWE": {
    "abbr": "SWE",
    "courseCode": "24CSEJ601",
    "courseName": "Software Engineering",
    "facultyId": "keerthi",
    "facultyName": "Dr. Keerthi G"
  },
  "CCM": {
    "abbr": "CCM",
    "courseCode": "24CSEJ602",
    "courseName": "Cloud Computing",
    "facultyId": "pranay",
    "facultyName": "Dr. Pranayaanath Reddy A"
  },
  "CVO": {
    "abbr": "CVO",
    "courseCode": "24AMLJ502",
    "courseName": "Computer Vision through OpenCV",
    "facultyId": "vikas",
    "facultyName": "Mr. Vikas B"
  }
};

export const USERS = [
  {
    "id": "admin",
    "username": "admin",
    "password": "admin123",
    "name": "HOD",
    "role": "admin",
    "roleLabel": "Head of Department",
    "subjects": [
      "SWE",
      "CCM",
      "CVO"
    ],
    "facultyName": "HOD",
    "canManageSystem": true,
    "canSeeAll": true
  },
  {
    "id": "keerthi",
    "username": "keerthi",
    "password": "swe123",
    "name": "Dr. Keerthi G",
    "role": "faculty",
    "roleLabel": "Faculty - Software Engineering",
    "subjects": [
      "SWE"
    ],
    "facultyName": "Dr. Keerthi G",
    "canManageSystem": false,
    "canSeeAll": false
  },
  {
    "id": "vikas",
    "username": "vikas",
    "password": "cvo123",
    "name": "Mr. Vikas B",
    "role": "faculty",
    "roleLabel": "Faculty - Computer Vision through OpenCV",
    "subjects": [
      "CVO"
    ],
    "facultyName": "Mr. Vikas B",
    "canManageSystem": false,
    "canSeeAll": false
  },
  {
    "id": "pranay",
    "username": "pranay",
    "password": "ccm123",
    "name": "Dr. Pranayaanath Reddy A",
    "role": "faculty",
    "roleLabel": "Faculty - Cloud Computing",
    "subjects": [
      "CCM"
    ],
    "facultyName": "Dr. Pranayaanath Reddy A",
    "canManageSystem": false,
    "canSeeAll": false
  }
];

export function publicUser(user) {
  if (!user) return null;
  const { password, ...safe } = user;
  return safe;
}

export function findLogin(username, password) {
  const u = USERS.find((user) => user.username.toLowerCase() === String(username || '').trim().toLowerCase() && user.password === password);
  return publicUser(u);
}

export function isAdmin(user) {
  return user?.role === 'admin' || user?.canSeeAll;
}

export function splitSubjects(value) {
  return String(value || '')
    .toUpperCase()
    .replace(/-/g, '/')
    .split('/')
    .map((x) => x.replace(/\b(LAB|THEORY)\b/g, '').trim())
    .filter(Boolean);
}

export function rowSubjects(row) {
  return splitSubjects(row?.subject || row?.course_abbr || row?.Course_Abbr || row?.Subject);
}

export function canAccessSubject(user, subject) {
  if (isAdmin(user)) return true;
  const allowed = new Set((user?.subjects || []).map((s) => String(s).toUpperCase()));
  return allowed.has(String(subject || '').toUpperCase());
}

export function canAccessRow(user, row) {
  if (isAdmin(user)) return true;
  return rowSubjects(row).some((subject) => canAccessSubject(user, subject));
}

export function displaySubjectForUser(row, user) {
  const subjects = rowSubjects(row);
  if (isAdmin(user) || subjects.length <= 1) return row?.subject || row?.course_abbr || row?.Subject || '-';
  const match = subjects.find((subject) => canAccessSubject(user, subject));
  return match || row?.subject || '-';
}

export function subjectInfo(subject) {
  return SUBJECT_INFO[String(subject || '').toUpperCase()] || null;
}
