export default function HomePage({ onNavigate, modelMetrics, bestModel }) {
  const getBestAccuracy = () => {
    if (!modelMetrics || !bestModel) return '—'
    const m = modelMetrics[bestModel]
    const acc = m?.accuracy_pct ?? (m?.accuracy ? m.accuracy * 100 : 0)
    return acc ? `${acc.toFixed(1)}%` : '—'
  }

  return (
    <section className="page-home dashboard-fade-in">
      <div className="home-hero card">
        <div className="home-hero-content">
          <h2 className="home-hero-title">Welcome Back</h2>
          <p className="home-hero-desc">
            Your customer churn prediction system is ready.
            {bestModel && (
              <>
                {' '}Current best performing model is <strong className="text-accent">{bestModel}</strong> with <strong className="text-success">{getBestAccuracy()}</strong> accuracy.
              </>
            )}
            <br />
            <span style={{ display: 'block', marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--muted)' }}>
              Use this system to identify at-risk customers early and take proactive retention actions.
            </span>
          </p>
          <div className="home-hero-actions">
            <button className="btn btn-primary" onClick={() => onNavigate('predict')}>
              Start Prediction
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate('charts')}>
              View Analytics
            </button>
          </div>
        </div>
        <div className="home-hero-stats">
          <div className="hero-stat-item">
            <span className="hero-stat-value">3</span>
            <span className="hero-stat-label">Active Models</span>
          </div>
          <div className="hero-stat-item">
            <span className="hero-stat-value text-success">{getBestAccuracy()}</span>
            <span className="hero-stat-label">Best Accuracy</span>
          </div>
        </div>
      </div>

      <div className="home-grid">
        <div className="card home-section">
          <h3 className="section-title">System Overview</h3>
          <div className="model-list">
            {['XGBoost', 'Gradient Boosting', 'Decision Tree'].map((model) => (
              <div key={model} className="model-list-item">
                <div className="model-info">
                  <span className="model-name">{model}</span>
                  <span className="model-status">Active</span>
                </div>
                {modelMetrics && modelMetrics[model] && (
                  <div className="model-metric">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-value">
                      {((modelMetrics[model].accuracy_pct ?? (modelMetrics[model].accuracy * 100)) || 0).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="card home-section">
          <h3 className="section-title">Quick Actions</h3>
          <div className="quick-actions-list">
            <button className="quick-action-btn" onClick={() => onNavigate('predict')}>
              <div className="action-icon bg-accent-soft">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                  <polyline points="13 2 13 9 20 9"></polyline>
                </svg>
              </div>
              <div className="action-details">
                <span className="action-title">New Prediction</span>
                <span className="action-desc">Analyze a single customer</span>
              </div>
              <svg className="action-arrow" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </button>

            <button className="quick-action-btn" onClick={() => onNavigate('charts')}>
              <div className="action-icon bg-success-soft">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 20V10M12 20V4M6 20v-6" />
                </svg>
              </div>
              <div className="action-details">
                <span className="action-title">Model Performance</span>
                <span className="action-desc">Compare accuracy & AUC</span>
              </div>
              <svg className="action-arrow" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
