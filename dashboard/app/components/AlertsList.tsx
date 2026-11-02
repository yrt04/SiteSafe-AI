export interface Alert {
  id: string;
  message: string;
  location: string;
  severity: 'high' | 'medium' | 'low' | 'info';
  timestamp: string;
}

interface AlertsListProps {
  alerts: Alert[];
}

const SEVERITY_ICONS: Record<Alert['severity'], string> = {
  high: '🔴',
  medium: '🟠',
  low: '🟢',
  info: '🔵',
};

export default function AlertsList({ alerts }: AlertsListProps) {
  return (
    <div className="alerts-panel slide-in">
      <div className="alerts-header">
        <h2>Recent Alerts</h2>
        <a href="#" className="view-all-link">View All</a>
      </div>
      <div className="alerts-list">
        {alerts.length === 0 && (
          <div className="alert-item severity-info">
            <div className="alert-content">
              <div className="alert-message" style={{ color: 'var(--text-muted)' }}>No recent alerts.</div>
            </div>
          </div>
        )}
        {alerts.map((alert) => (
          <div key={alert.id} className={`alert-item severity-${alert.severity}`}>
            <div className="alert-icon">{SEVERITY_ICONS[alert.severity] ?? '⚪'}</div>
            <div className="alert-content">
              <div className="alert-message">{alert.message}</div>
              <div className="alert-meta">
                <span className="alert-location">{alert.location}</span>
                <span className="separator">•</span>
                <span className="alert-time">{alert.timestamp}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
