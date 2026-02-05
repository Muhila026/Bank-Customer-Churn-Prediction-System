import { useState, useEffect } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts'
import Sidebar from './Sidebar'
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

export default function App() {
  const [form, setForm] = useState(defaultForm)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelMetrics, setModelMetrics] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE}/model-metrics`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => data?.models && setModelMetrics(data.models))
      .catch(() => { })
  }, [])

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
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} modelMetrics={modelMetrics} />
      <header className="dashboard-topbar">
        <button
          type="button"
          className="dashboard-menu-btn"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open menu"
          aria-expanded={sidebarOpen}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12h18M3 6h18M3 18h18" />
          </svg>
        </button>
        <div className="dashboard-topbar-brand">
          <div className="dashboard-topbar-logo" aria-hidden="true">
            <svg width="32" height="32" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="4" y="14" width="32" height="20" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
              <path d="M4 20h32" stroke="currentColor" strokeWidth="2" />
              <path d="M12 20v8M20 20v8M28 20v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M8 10h24v4H8z" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
          </div>
          <div className="dashboard-topbar-titles">
            <h1 className="dashboard-topbar-title">Bank Customer Churn</h1>
            <p className="dashboard-topbar-subtitle">Predict churn • XGBoost, GBM & DT</p>
          </div>
        </div>
      </header>

      <div className="dashboard-body">
        <div className="dashboard-content">
          <div className="dashboard-page-header">
            <h2 className="dashboard-page-title">Predict churn</h2>
            <p className="dashboard-page-desc">Enter customer details and run prediction.</p>
          </div>

          <main className="dashboard-main main-grid">
            <div className="main-left">
              <form className="card form-card dashboard-card" onSubmit={handleSubmit}>
                <h2>Customer details</h2>

                <div className="grid">
                  <label>
                    <span>Credit score (0–1000)</span>
                    <input
                      type="number"
                      name="CreditScore"
                      min={0}
                      max={1000}
                      value={form.CreditScore}
                      onChange={handleChange}
                    />
                  </label>
                  <label>
                    <span>Geography</span>
                    <select name="Geography" value={form.Geography} onChange={handleChange}>
                      <option value="France">France</option>
                      <option value="Spain">Spain</option>
                      <option value="Germany">Germany</option>
                    </select>
                  </label>
                  <label className="gender-label">
                    <span>Gender</span>
                    <div className="gender-options" role="group" aria-label="Gender">
                      <button
                        type="button"
                        className={`gender-option ${form.Gender === 'Male' ? 'selected' : ''}`}
                        onClick={() => {
                          setForm((prev) => ({ ...prev, Gender: 'Male' }))
                          setResult(null)
                          setError(null)
                        }}
                        aria-pressed={form.Gender === 'Male'}
                      >
                        <span className="gender-icon gender-icon-male" aria-hidden="true">
                          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                            <circle cx="10" cy="14" r="5" />
                            <path stroke="currentColor" strokeWidth="2" strokeLinecap="round" fill="none" d="M19 3l-5.5 5.5M19 3h-5M19 3v5" />
                          </svg>
                        </span>
                        <span>Male</span>
                      </button>
                      <button
                        type="button"
                        className={`gender-option ${form.Gender === 'Female' ? 'selected' : ''}`}
                        onClick={() => {
                          setForm((prev) => ({ ...prev, Gender: 'Female' }))
                          setResult(null)
                          setError(null)
                        }}
                        aria-pressed={form.Gender === 'Female'}
                      >
                        <span className="gender-icon gender-icon-female" aria-hidden="true">
                          <svg width="24" height="24" viewBox="0 0 24 24">
                            <circle cx="12" cy="9" r="5" fill="currentColor" />
                            <path d="M12 14v6M9 17h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" fill="none" />
                          </svg>
                        </span>
                        <span>Female</span>
                      </button>
                    </div>
                  </label>
                  <label>
                    <span>Age</span>
                    <input
                      type="number"
                      name="Age"
                      min={0}
                      max={120}
                      value={form.Age}
                      onChange={handleChange}
                    />
                  </label>
                  <label>
                    <span>Tenure (years)</span>
                    <input
                      type="number"
                      name="Tenure"
                      min={0}
                      max={20}
                      value={form.Tenure}
                      onChange={handleChange}
                    />
                  </label>
                  <label>
                    <span>Balance</span>
                    <input
                      type="number"
                      name="Balance"
                      min={0}
                      step={0.01}
                      value={form.Balance}
                      onChange={handleChange}
                    />
                  </label>
                  <label>
                    <span>Number of products</span>
                    <input
                      type="number"
                      name="NumOfProducts"
                      min={0}
                      max={10}
                      value={form.NumOfProducts}
                      onChange={handleChange}
                    />
                  </label>
                  <label>
                    <span>Has credit card</span>
                    <select name="HasCrCard" value={form.HasCrCard} onChange={handleChange}>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </label>
                  <label>
                    <span>Active member</span>
                    <select name="IsActiveMember" value={form.IsActiveMember} onChange={handleChange}>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </label>
                  <label>
                    <span>Estimated salary</span>
                    <input
                      type="number"
                      name="EstimatedSalary"
                      min={0}
                      step={0.01}
                      value={form.EstimatedSalary}
                      onChange={handleChange}
                    />
                  </label>
                </div>

                <div className="actions">
                  <button type="submit" className="btn btn-primary" disabled={loading}>
                    {loading ? (
                      <span className="btn-loading">
                        <span className="btn-spinner" aria-hidden="true" />
                        <span>Analyzing…</span>
                      </span>
                    ) : (
                      'Predict churn'
                    )}
                  </button>
                </div>

                {error && (
                  <div className="message error" role="alert">
                    {error}
                  </div>
                )}
              </form>

              {/* Details under the form: bottom line, how likely, recommendation, other models */}
              {result && (
                <div className="details-under-form">
                  {/* Single “how likely” strip + gauge */}
                  {(() => {
                    const models = [
                      { key: 'XGBoost', prob: result.XGBoost?.exited_probability ?? 0 },
                      { key: 'Gradient_Boosting', prob: result.Gradient_Boosting?.exited_probability ?? 0 },
                      { key: 'Decision_Tree', prob: result.Decision_Tree?.exited_probability ?? 0 },
                    ]
                    const avgProb = models.length ? models.reduce((s, m) => s + m.prob, 0) / models.length : 0
                    const avgPct = (avgProb * 100).toFixed(1)
                    const riskLevel = avgProb >= 0.6 ? 'high' : avgProb >= 0.5 ? 'moderate' : 'low'
                    const riskCopy = riskLevel === 'low' ? 'Very unlikely to leave' : riskLevel === 'moderate' ? 'Some chance they might leave' : 'Likely to leave'
                    return (
                      <div className="result-risk-strip card">
                        <p className="result-risk-question">How likely is this customer to leave?</p>
                        <div className="result-risk-row">
                          <span className={`result-risk-copy risk-${riskLevel}`}>{riskCopy}</span>
                          <span className={`result-risk-pct risk-${riskLevel}`}>{avgPct}%</span>
                        </div>
                        <div className="viz-gauge-bar result-risk-bar">
                          <div
                            className="viz-gauge-fill"
                            style={{ width: `${Math.min(100, avgProb * 100)}%` }}
                            data-risk={riskLevel}
                          />
                        </div>
                      </div>
                    )
                  })()}

                  {result.Gradient_Boosting && (
                    <div className={`card result-card result-card--featured result-card--human model-result ${result.Gradient_Boosting.prediction === 'Churned' ? 'churned' : 'stayed'}`}>
                      <p className="result-card-human-label">Our main recommendation</p>
                      <p className="result-card-human-headline">
                        {result.Gradient_Boosting.prediction === 'Churned'
                          ? 'This customer may leave — worth reaching out.'
                          : "This customer is likely to stay. Keep doing what you're doing."}
                      </p>
                      <p className="result-card-human-detail">
                        {result.Gradient_Boosting.prediction === 'Churned'
                          ? `We estimate a ${(result.Gradient_Boosting.exited_probability * 100).toFixed(1)}% chance they'll close their account. A personal call or a tailored offer could help them stay.`
                          : `Only about ${(result.Gradient_Boosting.exited_probability * 100).toFixed(1)}% chance they'll leave. No urgent action needed — just keep engaging positively.`}
                      </p>
                      <p className="result-card-human-meta">Based on our Gradient Boosting model</p>
                    </div>
                  )}

                  <p className="results-intro results-intro--human">What the other two models say (same customer, same data):</p>
                  <div className="results-grid results-grid--two results-grid--human">
                    {[
                      { key: 'XGBoost', label: 'XGBoost' },
                      { key: 'Decision_Tree', label: 'Decision Tree' },
                    ].map(({ key, label }) => {
                      const m = result[key]
                      if (!m) return null
                      const isChurned = m.prediction === 'Churned'
                      const pct = (m.exited_probability * 100).toFixed(1)
                      return (
                        <div
                          key={key}
                          className={`card result-card result-card--compact model-result ${isChurned ? 'churned' : 'stayed'}`}
                        >
                          <p className="result-card-compact-label">{label}</p>
                          <p className="result-card-compact-line">
                            {isChurned ? 'May leave' : 'Likely to stay'} — {pct}% chance they leave.
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>

            <div className="main-right">
              {result ? (
                <div className="chat-items">
                  {result.message && (
                    <div className="result-takeaway card chat-item">
                      <p className="result-takeaway-label">Here's the bottom line</p>
                      <p className="result-takeaway-text">{result.message}</p>
                    </div>
                  )}
                  <div className="prediction-viz card prediction-viz--human chat-item">
                    <h3 className="viz-title viz-title--human">Leave probability by model</h3>
                    <div className="viz-chart-wrap">
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart
                          data={[
                            { name: 'XGBoost', probability: (result.XGBoost?.exited_probability ?? 0) * 100, fill: '#0369a1' },
                            { name: 'Gradient\u00A0Boosting', probability: (result.Gradient_Boosting?.exited_probability ?? 0) * 100, fill: '#c2410c' },
                            { name: 'Decision Tree', probability: (result.Decision_Tree?.exited_probability ?? 0) * 100, fill: '#7c3aed' },
                          ]}
                          margin={{ top: 10, right: 16, bottom: 8, left: 8 }}
                          layout="vertical"
                        >
                          <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10 }} />
                          <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12, fontWeight: 500 }} />
                          <Tooltip formatter={(v) => [`${Number(v).toFixed(1)}%`, 'Leave probability']} />
                          <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
                            {[
                              { fill: '#0369a1' },
                              { fill: '#c2410c' },
                              { fill: '#7c3aed' },
                            ].map((c, i) => (
                              <Cell key={i} fill={c.fill} stroke={i === 1 ? '#9a3412' : undefined} strokeWidth={i === 1 ? 2 : 0} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="prediction-viz card prediction-viz--human chat-item">
                    <h3 className="viz-title viz-title--human">Do the models agree?</h3>
                    <div className="viz-chart-wrap viz-pie-wrap">
                      {(() => {
                        const churned = [result.XGBoost, result.Gradient_Boosting, result.Decision_Tree].filter((m) => m?.prediction === 'Churned').length
                        const stayed = 3 - churned
                        const pieData = [
                          { name: 'Say they’ll leave', value: churned, fill: '#b91c1c' },
                          { name: 'Say they’ll stay', value: stayed, fill: '#047857' },
                        ].filter((d) => d.value > 0)
                        return (
                          <ResponsiveContainer width="100%" height={180}>
                            <PieChart>
                              <Pie
                                data={pieData}
                                dataKey="value"
                                nameKey="name"
                                cx="50%"
                                cy="50%"
                                innerRadius={40}
                                outerRadius={60}
                                paddingAngle={2}
                                label={({ name, value }) => `${name}: ${value} of 3`}
                              >
                                {pieData.map((entry, i) => (
                                  <Cell key={i} fill={entry.fill} />
                                ))}
                              </Pie>
                              <Tooltip formatter={(v) => [`${v} of 3 models`, '']} />
                              <Legend />
                            </PieChart>
                          </ResponsiveContainer>
                        )
                      })()}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="card result-placeholder">
                  <p>Fill the form and click Predict churn to see results here.</p>
                </div>
              )}
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}
