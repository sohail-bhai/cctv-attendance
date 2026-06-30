import { useEffect, useMemo, useRef, useState } from 'react';

function normaliseOptions(options = []) {
  return options.map((option) => {
    if (typeof option === 'string' || typeof option === 'number') {
      return { value: String(option), label: String(option), description: '' };
    }
    return {
      value: String(option.value ?? option.label ?? ''),
      label: String(option.label ?? option.value ?? ''),
      description: option.description || option.meta || '',
      disabled: Boolean(option.disabled),
    };
  });
}

export default function NiceSelect({
  value,
  options = [],
  onChange,
  placeholder = 'Select',
  className = '',
  compact = false,
  disabled = false,
}) {
  const ref = useRef(null);
  const [open, setOpen] = useState(false);
  const items = useMemo(() => normaliseOptions(options), [options]);
  const selected = items.find((item) => item.value === String(value)) || null;

  useEffect(() => {
    const handleClick = (event) => {
      if (!ref.current?.contains(event.target)) setOpen(false);
    };
    const handleKey = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleKey);
    };
  }, []);

  const choose = (item) => {
    if (item.disabled) return;
    onChange?.(item.value);
    setOpen(false);
  };

  return (
    <div ref={ref} className={`nice-select ${compact ? 'compact' : ''} ${disabled ? 'disabled' : ''} ${className}`}>
      <button
        type="button"
        className="nice-select-trigger"
        onClick={() => !disabled && setOpen((x) => !x)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span>
          <strong>{selected?.label || placeholder}</strong>
          {selected?.description && <small>{selected.description}</small>}
        </span>
        <b>{open ? '⌃' : '⌄'}</b>
      </button>
      {open && (
        <div className="nice-select-menu" role="listbox">
          {items.map((item) => (
            <button
              type="button"
              role="option"
              aria-selected={item.value === String(value)}
              key={item.value}
              className={`nice-select-option ${item.value === String(value) ? 'active' : ''}`}
              disabled={item.disabled}
              onClick={() => choose(item)}
            >
              <span>
                <strong>{item.label}</strong>
                {item.description && <small>{item.description}</small>}
              </span>
              {item.value === String(value) && <em>✓</em>}
            </button>
          ))}
          {items.length === 0 && <div className="nice-select-empty">No options available</div>}
        </div>
      )}
    </div>
  );
}
