'use client';

import { useState } from 'react';

const NAV_ITEMS = [
  { id: 'live-feeds', label: 'Live Feeds', icon: '📹' },
  { id: 'alerts', label: 'Alerts & Events', icon: '🔔' },
  { id: 'analytics', label: 'Analytics', icon: '📊' },
  { id: 'site-map', label: 'Site Map', icon: '🗺️' },
  { id: 'settings', label: 'Settings', icon: '⚙️' },
] as const;

export default function Sidebar() {
  const [activeItem, setActiveItem] = useState<string>('live-feeds');

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo-container">
          <span className="logo-icon">🛡️</span>
          <span className="logo-text glow-effect">SiteSafe AI</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <ul>
          {NAV_ITEMS.map((item) => (
            <li key={item.id}>
              <button
                className={`sidebar-nav-item ${activeItem === item.id ? 'active' : ''}`}
                onClick={() => setActiveItem(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="sidebar-footer">
        <div className="system-status">
          <span className="status-indicator online"></span>
          <span className="status-text">All Systems Online</span>
        </div>
        <div className="user-profile">
          <div className="user-avatar">👤</div>
          <div className="user-info">
            <span className="user-name">Admin</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
