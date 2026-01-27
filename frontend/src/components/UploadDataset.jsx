import React, { useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function UploadDataset() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)

  const submit = async (e) => {
    e.preventDefault()
    if (!file) return
    const fd = new FormData()
    fd.append('file', file)
    const res = await axios.post(`${API}/api/models/upload`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
    setResult(res.data)
  }

  return (
    <div>
      <h2>Upload Dataset (placeholder)</h2>
      <form onSubmit={submit}>
        <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button type="submit">Upload</button>
      </form>
      {result && (
        <div>
          <p>Saved: {result.filename}</p>
          <p>Path: {result.saved_to}</p>
        </div>
      )}
    </div>
  )
}
