import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const fetchModels = createAsyncThunk('models/fetchModels', async () => {
  const res = await axios.get(`${API}/api/models`)
  return res.data.models
})

export const fetchMetrics = createAsyncThunk('models/fetchMetrics', async (name) => {
  const res = await axios.get(`${API}/api/models/${name}/metrics`)
  return { name, metrics: res.data }
})

const modelsSlice = createSlice({
  name: 'models',
  initialState: { list: [], status: 'idle', metrics: {}, error: null },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchModels.pending, (state) => { state.status = 'loading' })
      .addCase(fetchModels.fulfilled, (state, action) => { state.status = 'succeeded'; state.list = action.payload })
      .addCase(fetchModels.rejected, (state, action) => { state.status = 'failed'; state.error = action.error.message })
      .addCase(fetchMetrics.fulfilled, (state, action) => { state.metrics[action.payload.name] = action.payload.metrics })
  }
})

export default modelsSlice.reducer
