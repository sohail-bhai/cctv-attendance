export default function FileDrop({
  title,
  label,
  hint,
  help,
  accept = '.csv',
  onFile,
  onChange,
}) {
  const mainLabel = title || label || 'Upload file';
  const subText = hint || help || 'Choose a CSV file';

  const handleChange = (event) => {
    const file = event.target.files?.[0] || null;
    onFile?.(file);
    onChange?.(file);
  };

  return (
    <label className="file-drop compact-file-drop file-drop-v2">
      <input type="file" accept={accept} onChange={handleChange} />
      <span className="file-icon" aria-hidden="true">⇪</span>
      <span className="file-drop-text">
        <strong>{mainLabel}</strong>
        {subText && <small>{subText}</small>}
      </span>
    </label>
  );
}
