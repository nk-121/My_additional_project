import React from 'react'
import { useSelector } from 'react-redux'
import { Link } from 'react-router-dom'

export default function ModelList() {
  const list = useSelector((s) => s.models.list)
  const status = useSelector((s) => s.models.status)

  if (status === 'loading') return <p>Loading models...</p>
  if (!list.length) return <p>No models available.</p>

  return (
    <div>
      <h2>Available Models</h2>
      <ul>
        {list.map((m) => (
          <li key={m}>
            <Link to={`/models/${m}`}>{m}</Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
