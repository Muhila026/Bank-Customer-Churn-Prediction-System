import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Legend,
    CartesianGrid
} from 'recharts'

export default function ViewChartPage({ modelMetrics }) {
    if (!modelMetrics || Object.keys(modelMetrics).length === 0) {
        return (
            <div className="dashboard-content">
                <div className="dashboard-page-header">
                    <h2 className="dashboard-page-title">Model Performance Charts</h2>
                    <p className="dashboard-page-desc">Loading model metrics...</p>
                </div>
            </div>
        )
    }

    const data = ['XGBoost', 'Gradient Boosting', 'Decision Tree'].map(name => {
        const m = modelMetrics[name] || {}
        return {
            name,
            Accuracy: (m.accuracy_pct || m.accuracy * 100 || 0).toFixed(2),
            'ROC AUC': (m.roc_auc || 0).toFixed(2) * 100, // Assuming roc_auc is 0-1, convert to percentage for chart
        }
    })

    return (
        <div className="dashboard-content">
            <div className="dashboard-page-header">
                <h2 className="dashboard-page-title">Model Performance Charts</h2>
                <p className="dashboard-page-desc">Visual comparison of model accuracy and ROC-AUC scores.</p>
            </div>

            <main className="dashboard-main">
                <div className="card dashboard-card" style={{ height: '400px', padding: '1.5rem' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                            data={data}
                            margin={{
                                top: 20,
                                right: 30,
                                left: 20,
                                bottom: 5,
                            }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="name" tick={{ fill: '#cbd5e1' }} />
                            <YAxis tick={{ fill: '#cbd5e1' }} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(30, 41, 59, 0.9)',
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    borderRadius: '8px',
                                    color: '#fff'
                                }}
                            />
                            <Legend wrapperStyle={{ color: '#cbd5e1' }} />
                            <Bar dataKey="Accuracy" fill="#38bdf8" name="Accuracy (%)" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="ROC AUC" fill="#34d399" name="ROC AUC (%)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                <div className="card dashboard-card">
                    <h3>Metrics Summary</h3>
                    <div className="dashboard-stats" style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginTop: '1rem' }}>
                        {data.map((d) => (
                            <div key={d.name} className="dashboard-stat-card">
                                <h4 className="dashboard-stat-label">{d.name}</h4>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                    <div>
                                        <span style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>Accuracy</span>
                                        <p className="dashboard-stat-value">{d.Accuracy}%</p>
                                    </div>
                                    <div>
                                        <span style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>ROC AUC</span>
                                        <p className="dashboard-stat-value" style={{ fontSize: '1rem', color: 'var(--success)' }}>{d['ROC AUC']}%</p>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    )
}
