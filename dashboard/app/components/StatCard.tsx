interface StatCardProps {
  icon: string;
  label: string;
  value: string | number;
  trend?: { value: number; direction: 'up' | 'down' };
  accentColor?: string;
}

export default function StatCard({ icon, label, value, trend, accentColor }: StatCardProps) {
  return (
    <div className="stat-card" style={{ borderTop: accentColor ? `3px solid ${accentColor}` : undefined }}>
      <div className="stat-header">
        <span className="stat-icon">{icon}</span>
      </div>
      <div className="stat-body">
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
      {trend && (
        <div className={`stat-trend ${trend.direction}`}>
          <span className="trend-arrow">{trend.direction === 'up' ? '↑' : '↓'}</span>
          <span className="trend-value">{trend.value}%</span>
        </div>
      )}
    </div>
  );
}
