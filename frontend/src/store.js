import { configureStore } from '@reduxjs/toolkit'
import modelsReducer from './features/modelsSlice'

export default configureStore({
  reducer: {
    models: modelsReducer,
  },
})
