import { useState, useEffect } from 'react'
import Sidebar from './Sidebar'
import HomePage from './pages/HomePage'
import PredictPage from './pages/PredictPage'
import ViewChartPage from './pages/ViewChartPage'
import './App.css'

const API_BASE = '/api'

const defaultForm = {
  CreditScore: 650,
  Geography: 'France',
  Gender: 'Male',
  Age: 38,
  Tenure: 5,
  Balance: 0,
  NumOfProducts: 2,
  HasCrCard: 1,
  IsActiveMember: 1,
  EstimatedSalary: 100000,
}

const NAV_ITEMS = [
  { id: 'home', label: 'Home', icon: 'home' },
  { id: 'predict', label: 'Predict', icon: 'predict' },
  { id: 'charts', label: 'View Charts', icon: 'chart' },
]

const PAGE_TITLES = {
  home: 'Home',
  predict: 'Predict churn',
  charts: 'Model Charts',
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [currentPage, setCurrentPage] = useState('predict')
  const [form, setForm] = useState(defaultForm)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelMetrics, setModelMetrics] = useState(null)
  const [bestModel, setBestModel] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/model-metrics`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.models) setModelMetrics(data.models)
        if (data?.best_model) setBestModel(data.best_model)
      })
      .catch(() => { })
  }, [])

  const bestModelKey = bestModel === 'Gradient Boosting' ? 'Gradient_Boosting' : bestModel === 'Decision Tree' ? 'Decision_Tree' : 'XGBoost'

  const handleChange = (e) => {
    const { name, value } = e.target
    const parsed = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'].includes(name)
      ? parseFloat(value) || 0
      : ['HasCrCard', 'IsActiveMember'].includes(name)
        ? parseInt(value, 10)
        : value
    setForm((prev) => ({ ...prev, [name]: parsed }))
    setResult(null)
    setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const text = await res.text()
      let data = null
      if (text && text.trim()) {
        try {
          data = JSON.parse(text)
        } catch {
          setError('Invalid response from server. Check that the backend is running on port 8000.')
          return
        }
      }
      if (!res.ok) {
        const msg = data?.detail || (typeof data?.detail === 'string' ? data.detail : null) || text || `Request failed (${res.status})`
        setError(Array.isArray(msg) ? msg.join(', ') : String(msg))
        return
      }
      if (!data || typeof data !== 'object') {
        setError('Empty response from server. Run the training script, then restart the backend.')
        return
      }
      setResult(data)
    } catch (err) {
      setError(err.message || 'Network error. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="dashboard">
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        navItems={NAV_ITEMS}
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        modelMetrics={modelMetrics}
        bestModel={bestModel}
      />

      <div className="dashboard-main-wrap">
        <div className="dashboard-content">

          {currentPage === 'home' && (
            <HomePage
              onNavigate={setCurrentPage}
              modelMetrics={modelMetrics}
              bestModel={bestModel}
            />
          )}

          {currentPage === 'predict' && (
            <PredictPage
              form={form}
              handleChange={handleChange}
              handleSubmit={handleSubmit}
              loading={loading}
              error={error}
              result={result}
              bestModel={bestModel}
              bestModelKey={bestModelKey}
            />
          )}

          {currentPage === 'charts' && <ViewChartPage modelMetrics={modelMetrics} />}
        </div>
      </div>
    </div>
  )
}
