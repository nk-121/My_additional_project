import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import { fetchMetrics } from '../features/modelsSlice'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function ModelDetail() {
  const { name } = useParams()
  const dispatch = useDispatch()
  const metrics = useSelector((s) => s.models.metrics[name])
  const [features, setFeatures] = useState('')
  const [prediction, setPrediction] = useState(null)

  useEffect(() => { if (name) dispatch(fetchMetrics(name)) }, [name, dispatch])

  const handlePredict = async () => {
    try {
      const arr = features.split(',').map((s) => s.trim())
      const res = await axios.post(`${API}/api/models/${name}/predict`, { features: arr })
      setPrediction(res.data.prediction)
    } catch (e) {
      setPrediction('error')
    }
  }

  return (
    <div>
      <h2>Model: {name}</h2>
      {metrics ? (
        <div>
          <p><strong>{metrics.name}</strong></p>
          <p>Accuracy: {metrics.accuracy}</p>
          <p>{metrics.notes}</p>
        </div>
      ) : (
        <p>Loading metrics…</p>
      )}

      <section>
        <h3>Run a quick (placeholder) prediction</h3>
        <p>Enter comma-separated feature values (placeholder)</p>
        <input value={features} onChange={(e) => setFeatures(e.target.value)} placeholder="1.2, 3, 4" />
        <button onClick={handlePredict}>Predict</button>
        {prediction !== null && <p>Prediction: {String(prediction)}</p>}
      </section>
    </div>
  )
}
