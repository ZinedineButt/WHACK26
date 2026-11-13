import React, { useEffect, useState } from 'react'

export default function StatusSelect({
  id,
  value,
  options,
  onChange,
  disabled = false,
}) {
  return (
    <select
      id={id}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      disabled={disabled}
      aria-label="Task status"
      className="status-select"
    >
      {options.map((status) => (
        <option key={status} value={status}>
          {status}
        </option>
      ))}
    </select>
  )
}