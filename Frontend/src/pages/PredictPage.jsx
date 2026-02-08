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

export default function PredictPage({
  form,
  handleChange,
  handleSubmit,
  loading,
  error,
  result,
  bestModel,
  bestModelKey,
}) {
  return (
    <>
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
              <label>
                <span>Age</span>
                <input type="number" name="Age" min={0} max={120} value={form.Age} onChange={handleChange} />
              </label>
              <label>
                <span>Gender</span>
                <div className="gender-options" role="group" aria-label="Gender">
                  <button
                    type="button"
                    className={`gender-option ${form.Gender === 'Male' ? 'selected' : ''}`}
                    onClick={() => {
                      handleChange({ target: { name: 'Gender', value: 'Male' } })
                    }}
                    aria-pressed={form.Gender === 'Male'}
                  >
                    <span>Male</span>
                  </button>
                  <button
                    type="button"
                    className={`gender-option ${form.Gender === 'Female' ? 'selected' : ''}`}
                    onClick={() => {
                      handleChange({ target: { name: 'Gender', value: 'Female' } })
                    }}
                    aria-pressed={form.Gender === 'Female'}
                  >
                    <span>Female</span>
                  </button>
                </div>
              </label>
              <label>
                <span>Tenure (years)</span>
                <input type="number" name="Tenure" min={0} max={20} value={form.Tenure} onChange={handleChange} />
              </label>
              <label>
                <span>Balance</span>
                <input type="number" name="Balance" min={0} step={0.01} value={form.Balance} onChange={handleChange} />
              </label>
              <label>
                <span>Number of products</span>
                <input type="number" name="NumOfProducts" min={0} max={10} value={form.NumOfProducts} onChange={handleChange} />
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
                <input type="number" name="EstimatedSalary" min={0} step={0.01} value={form.EstimatedSalary} onChange={handleChange} />
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
        </div >

        <div className="main-right">
          {result ? (
            <div className="chat-items dashboard-fade-in">
              {result.message && (
                <div className="result-takeaway card chat-item">
                  <p className="result-takeaway-label">Here's the bottom line</p>
                  <p className="result-takeaway-text text-accent">{result.message}</p>
                </div>
              )}

              <h3 className="viz-title" style={{ marginTop: '1.5rem' }}>Model Predictions</h3>
              <div className="model-results-grid">
                {[
                  { key: 'XGBoost', label: 'XGBoost' },
                  { key: 'Gradient_Boosting', label: 'Gradient Boosting' },
                  { key: 'Decision_Tree', label: 'Decision Tree' },
                ].map(({ key, label }) => {
                  const m = result[key]
                  if (!m) return null
                  const isBest = key === bestModelKey
                  const prob = (m.exited_probability * 100).toFixed(1)
                  const isChurn = m.prediction === 'Churned'

                  return (
                    <div key={key} className={`model-result-card ${isBest ? 'best-model-card' : ''}`}>
                      <h4 className="model-card-title">{label}</h4>
                      <div className="model-card-prediction" style={{ color: isChurn ? 'var(--danger)' : 'var(--success)' }}>
                        {isChurn ? 'Churn' : 'Stay'}
                      </div>
                      <span className="model-card-prob">{prob}% probability</span>
                    </div>
                  )
                })}
              </div>

              <h3 className="viz-title">Visual Analysis</h3>
              <div className="charts-container">
                <div className="prediction-viz card">
                  <h4 className="viz-title">Leave Probability</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart
                      data={[
                        { name: 'XGB', prob: (result.XGBoost?.exited_probability ?? 0) * 100, fill: '#0369a1' },
                        { name: 'GBM', prob: (result.Gradient_Boosting?.exited_probability ?? 0) * 100, fill: '#c2410c' },
                        { name: 'DT', prob: (result.Decision_Tree?.exited_probability ?? 0) * 100, fill: '#7c3aed' },
                      ]}
                      margin={{ top: 10, right: 10, bottom: 0, left: -20 }}
                    >
                      <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v) => [`${Number(v).toFixed(1)}%`, 'Prob']} />
                      <Bar dataKey="prob" radius={[4, 4, 0, 0]}>
                        <Cell fill="#0369a1" />
                        <Cell fill="#c2410c" />
                        <Cell fill="#7c3aed" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="prediction-viz card">
                  <h4 className="viz-title">Model Agreement</h4>
                  <div style={{ height: 200, width: '100%' }}>
                    {(() => {
                      const churned = [result.XGBoost, result.Gradient_Boosting, result.Decision_Tree].filter((m) => m?.prediction === 'Churned').length
                      const stayed = 3 - churned
                      const pieData = [
                        { name: 'Churn', value: churned, fill: '#b91c1c' },
                        { name: 'Stay', value: stayed, fill: '#047857' },
                      ].filter(d => d.value > 0)

                      return (
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={pieData}
                              cx="50%"
                              cy="50%"
                              innerRadius={50}
                              outerRadius={70}
                              paddingAngle={5}
                              dataKey="value"
                            >
                              {pieData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                              ))}
                            </Pie>
                            <Tooltip />
                            <Legend verticalAlign="bottom" height={36} />
                          </PieChart>
                        </ResponsiveContainer>
                      )
                    })()}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="card result-placeholder">
              <p>Fill the form and click Predict churn to see results here.</p>
            </div>
          )}
        </div>
      </main >
    </>
  )
}
