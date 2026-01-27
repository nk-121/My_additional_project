import React, { useEffect } from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import ModelList from './components/ModelList'
import ModelDetail from './components/ModelDetail'
import UploadDataset from './components/UploadDataset'
import { useDispatch } from 'react-redux'
import { fetchModels } from './features/modelsSlice'

export default function App() {
  const dispatch = useDispatch()

  useEffect(() => {
    dispatch(fetchModels())
  }, [dispatch])

  return (
    <div className="container">
      <header>
        <h1>Rain Prediction — Model Explorer</h1>
        <nav>
          <Link to="/">Models</Link>
          {' | '}
          <Link to="/upload">Upload Dataset</Link>
        </nav>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<ModelList />} />
          <Route path="/models/:name" element={<ModelDetail />} />
          <Route path="/upload" element={<UploadDataset />} />
        </Routes>
      </main>
    </div>
  )
}
