const MODEL_NAMES = ['XGBoost', 'Gradient Boosting', 'Decision Tree']

const navItems = [
  { id: 'predict', label: 'Predict churn', icon: 'predict' },
]

const SidebarLogo = () => (
  <div className="sidebar-logo" aria-hidden="true">
    <svg width="32" height="32" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="4" y="14" width="32" height="20" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
      <path d="M4 20h32" stroke="currentColor" strokeWidth="2" />
      <path d="M12 20v8M20 20v8M28 20v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M8 10h24v4H8z" stroke="currentColor" strokeWidth="2" fill="none" />
    </svg>
  </div>
)

export default function Sidebar({ isOpen, onClose, modelMetrics }) {
  return (
    <>
      <div
        className={`sidebar-overlay ${isOpen ? 'sidebar-overlay--open' : ''}`}
        onClick={onClose}
        onKeyDown={(e) => e.key === 'Escape' && onClose()}
        role="button"
        tabIndex={-1}
        aria-label="Close menu"
        aria-hidden={!isOpen}
      />
      <aside
        className={`sidebar ${isOpen ? 'sidebar--open' : ''}`}
        aria-label="Main navigation"
      >
        <div className="sidebar-inner">
          <div className="sidebar-header">
            <SidebarLogo />
            <span className="sidebar-brand">Bank Churn</span>
            <button
              type="button"
              className="sidebar-close"
              onClick={onClose}
              aria-label="Close sidebar"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
          <nav className="sidebar-nav">
            {navItems.map((item) => (
              <button
                key={item.id}
                type="button"
                className="sidebar-nav-item sidebar-nav-item--active"
                onClick={onClose}
              >
                <span className="sidebar-nav-icon" aria-hidden="true">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                    <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="2" fill="none" />
                  </svg>
                </span>
                <span>{item.label}</span>
              </button>
            ))}
          </nav>
          {modelMetrics && Object.keys(modelMetrics).length > 0 && (
            <div className="sidebar-metrics">
              <p className="sidebar-metrics-title">Model accuracy</p>
              {MODEL_NAMES.map((name) => {
                const m = modelMetrics[name]
                const accPct = m?.accuracy_pct != null ? m.accuracy_pct : (m?.accuracy != null ? m.accuracy * 100 : null)
                const rocPct = m?.roc_auc != null ? m.roc_auc * 100 : null
                const pct = accPct ?? rocPct
                const showAccuracy = accPct != null
                return (
                  <div key={name} className="sidebar-metric-item">
                    <span className="sidebar-metric-name">{name}</span>
                    <span className="sidebar-metric-value">{pct != null ? `${typeof pct === 'number' ? pct.toFixed(2) : pct}%` : '—'}</span>
                    <span className="sidebar-metric-desc">{showAccuracy ? 'Accuracy' : rocPct != null ? 'ROC-AUC' : '—'}</span>
                    {showAccuracy && rocPct != null && (
                      <span className="sidebar-metric-roc">ROC-AUC: {(rocPct).toFixed(2)}%</span>
                    )}
                  </div>
                )
              })}
            </div>
          )}
          <div className="sidebar-footer">
            <p className="sidebar-footer-text">Churn prediction with XGBoost, GBM & DT</p>
          </div>
        </div>
      </aside>
    </>
  )
}
