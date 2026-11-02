export interface TimelineEvent {
  id: string;
  message: string;
  timestamp: string;
  type: 'alert' | 'system' | 'detection';
}

interface ActivityTimelineProps {
  events: TimelineEvent[];
}

export default function ActivityTimeline({ events }: ActivityTimelineProps) {
  return (
    <div className="timeline-panel fade-in-up" style={{ animationDelay: '0.4s' }}>
      <div className="timeline-header">
        <h2>Activity Timeline</h2>
      </div>
      <div className="timeline-list">
        {events.length === 0 && (
          <div className="timeline-item">
            <div className="timeline-content">
              <div className="timeline-text" style={{ color: 'var(--text-muted)' }}>No recent activity.</div>
            </div>
          </div>
        )}
        {events.map((event) => (
          <div key={event.id} className="timeline-item">
            <div className={`timeline-dot type-${event.type}`}></div>
            <div className="timeline-content">
              <div className="timeline-text">{event.message}</div>
              <div className="timeline-time">{event.timestamp}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
