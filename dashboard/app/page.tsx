'use client';

import { useState, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import VideoFeed, { StreamStats } from './components/VideoFeed';
import StatCard from './components/StatCard';
import AlertsList, { Alert } from './components/AlertsList';
import ActivityTimeline, { TimelineEvent } from './components/ActivityTimeline';

const WS_URL = 'ws://localhost:8765';
const MAX_FEED_ITEMS = 6;

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

export default function Dashboard() {
  const [activeDetections, setActiveDetections] = useState(0);
  const [totalDetections, setTotalDetections] = useState(0);
  const [uptime, setUptime] = useState(0);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);

  // Monotonic sequence prevents React key collisions when multiple violations fire in the same millisecond.
  const alertSeqRef = useRef(0);

  const handleStatsUpdate = useCallback((stats: StreamStats) => {
    setActiveDetections(stats.activeDetections);
    setTotalDetections(stats.totalDetections);
    setUptime(stats.uptime);
  }, []);

  const handleNewAlert = useCallback((alertData: {
    id?: string;
    message: string;
    severity: 'high' | 'medium' | 'low' | 'info';
    timestamp?: string;
  }) => {
    const now = new Date();
    const timeStr = alertData.timestamp || now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    alertSeqRef.current += 1;
    const baseId = alertData.id || `${now.getTime()}-${alertSeqRef.current}`;

    const nextAlert: Alert = {
      id: `alert-${baseId}`,
      message: alertData.message,
      location: 'Main Camera',
      severity: alertData.severity,
      timestamp: timeStr,
    };
    setAlerts((prev) => [nextAlert, ...prev].slice(0, MAX_FEED_ITEMS));

    const nextEvent: TimelineEvent = {
      id: `evt-${baseId}`,
      message: alertData.message,
      timestamp: timeStr,
      type: alertData.severity === 'info' ? 'system' : 'alert',
    };
    setTimelineEvents((prev) => [nextEvent, ...prev].slice(0, MAX_FEED_ITEMS));
  }, []);

  return (
    <div className="dashboard-layout">
      <Sidebar />

      <main className="main-content">
        <header className="header">
          <div>
            <h1 className="glow-text">Site Overview</h1>
            <p className="text-secondary" style={{ fontSize: '0.9rem', marginTop: '4px' }}>
              Live security metrics &amp; AI monitoring
            </p>
          </div>
          <div className="status-badge live">
            <span className="pulse-dot"></span>
            System Active
          </div>
        </header>

        <section className="grid-3x1">
          <StatCard
            icon="🎯"
            label="Active AI Detections"
            value={activeDetections}
            trend={{ value: activeDetections > 0 ? 100 : 0, direction: activeDetections > 0 ? 'up' : 'down' }}
            accentColor="var(--accent-cyan)"
          />
          <StatCard
            icon="📹"
            label="Total Detections (Session)"
            value={totalDetections}
            accentColor="var(--accent-blue)"
          />
          <StatCard
            icon="⏱️"
            label="System Uptime"
            value={formatUptime(uptime)}
            accentColor="var(--accent-emerald)"
          />
        </section>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '24px', marginTop: '24px' }}>
          <div className="main-feed-area">
            <VideoFeed
              id="cam-1"
              name="Main Security Camera"
              location="Office / Workspace"
              resolution="720p HD"
              wsUrl={WS_URL}
              onStatsUpdate={handleStatsUpdate}
              onNewAlert={handleNewAlert}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <AlertsList alerts={alerts} />
            <ActivityTimeline events={timelineEvents} />
          </div>
        </div>
      </main>
    </div>
  );
}
