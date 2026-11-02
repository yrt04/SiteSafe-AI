'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
}

interface FrameMessage {
  type: 'frame';
  camera_id: string;
  data: string;
  timestamp: string;
  detections: Detection[];
  fps: number;
  detection_count: number;
  total_detections: number;
  is_recording: boolean;
  uptime_seconds: number;
}

export interface StreamStats {
  fps: number;
  activeDetections: number;
  totalDetections: number;
  isRecording: boolean;
  uptime: number;
}

interface VideoFeedProps {
  id: string;
  name: string;
  location: string;
  resolution: string;
  wsUrl?: string;
  onExpand?: (id: string) => void;
  onStatsUpdate?: (stats: StreamStats) => void;
  onNewAlert?: (alert: {
    message: string;
    severity: 'high' | 'medium' | 'low' | 'info';
  }) => void;
}

type FeedStatus = 'live' | 'recording' | 'idle' | 'offline';

const STATUS_LABELS: Record<FeedStatus, string> = {
  live: 'LIVE',
  recording: 'REC',
  idle: 'IDLE',
  offline: 'OFFLINE',
};

const RECONNECT_DELAY_MS = 3000;

export default function VideoFeed({
  id,
  name,
  location,
  resolution,
  wsUrl,
  onExpand,
  onStatsUpdate,
  onNewAlert,
}: VideoFeedProps) {
  const [clockText, setClockText] = useState<string>('');
  const [connected, setConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveFps, setLiveFps] = useState(0);
  const [liveDetections, setLiveDetections] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [frameSrc, setFrameSrc] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const syncClock = () => setClockText(new Date().toLocaleTimeString());
    syncClock();
    const interval = setInterval(syncClock, 1000);
    return () => clearInterval(interval);
  }, []);

  const connectWebSocket = useCallback(() => {
    if (!wsUrl) return;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      onNewAlert?.({ message: 'Dashboard connected to server', severity: 'info' });
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'feed_status') {
          setIsStreaming(msg.streaming);
          if (!msg.streaming) {
            setFrameSrc(null);
            setLiveFps(0);
            setLiveDetections(0);
          }
          if (msg.message) {
            onNewAlert?.({ message: msg.message, severity: 'info' });
          }
        } else if (msg.type === 'frame' && msg.camera_id === id) {
          const frameMsg = msg as FrameMessage;
          setFrameSrc(`data:image/jpeg;base64,${frameMsg.data}`);
          setLiveFps(frameMsg.fps);
          setLiveDetections(frameMsg.detection_count);
          setIsRecording(frameMsg.is_recording);
          setIsStreaming(true);

          onStatsUpdate?.({
            fps: frameMsg.fps,
            activeDetections: frameMsg.detection_count,
            totalDetections: frameMsg.total_detections,
            isRecording: frameMsg.is_recording,
            uptime: frameMsg.uptime_seconds,
          });

        } else if (msg.type === 'recording_status') {
          setIsRecording(msg.recording);
          if (msg.message) {
            onNewAlert?.({ message: msg.message, severity: msg.recording ? 'high' : 'info' });
          }
        } else if (msg.type === 'snapshot_saved') {
          if (msg.message) {
            onNewAlert?.({ message: msg.message, severity: 'info' });
          }
          // Trigger browser download directly from the base64 payload without needing a static file route.
          if (msg.success && msg.data) {
            const link = document.createElement('a');
            link.href = `data:image/jpeg;base64,${msg.data}`;
            link.download = msg.filename;
            link.click();
          }
        }
      } catch {
        return;
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setIsStreaming(false);
      setFrameSrc(null);
      setLiveFps(0);
      setLiveDetections(0);
      // Schedule automatic reconnect if the inference server restarts or drops.
      reconnectTimerRef.current = setTimeout(connectWebSocket, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => ws.close();
  }, [wsUrl, id, onStatsUpdate, onNewAlert]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connectWebSocket]);

  const sendCommand = (type: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type }));
    }
  };

  const toggleFeed = () => {
    sendCommand(isStreaming ? 'stop_feed' : 'start_feed');
  };

  const toggleRecording = () => {
    if (isStreaming) {
      sendCommand(isRecording ? 'stop_recording' : 'start_recording');
    }
  };

  const takeSnapshot = () => {
    if (isStreaming) {
      sendCommand('snapshot');
    }
  };

  const currentStatus: FeedStatus = !connected
    ? 'offline'
    : !isStreaming
      ? 'idle'
      : isRecording
        ? 'recording'
        : 'live';

  return (
    <div className="video-feed-card" style={{ height: '100%', width: '100%' }}>
      <div
        className="video-container"
        style={{ aspectRatio: '16/9', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
      >
        {frameSrc && isStreaming && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={frameSrc} alt={`${name} live feed`} className="video-frame-img" />
        )}

        {connected && !isStreaming && (
          <div style={{ zIndex: 5, textAlign: 'center' }}>
            <button
              onClick={toggleFeed}
              style={{
                padding: '12px 24px',
                fontSize: '1rem',
                fontWeight: '600',
                background: 'var(--accent-cyan)',
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 4px 12px rgba(6, 182, 212, 0.4)',
              }}
            >
              ▶ Start Camera Feed
            </button>
          </div>
        )}

        {!connected && (
          <div className="video-connecting">
            <div className="connecting-spinner" />
            <span>Connecting to server…</span>
          </div>
        )}

        {liveDetections > 0 && isStreaming && (
          <div className="badge-ai-detection">🎯 AI: {liveDetections}</div>
        )}

        <div className={`badge-status status-${currentStatus}`}>
          <span className={`status-pulse-dot ${currentStatus}`}></span>
          {STATUS_LABELS[currentStatus]}
        </div>

        {isStreaming && liveFps > 0 && (
          <div className="badge-fps">{liveFps.toFixed(0)} FPS</div>
        )}

        <div className="video-timestamp">{clockText}</div>

        {isStreaming && (
          <div className="video-controls">
            <button onClick={toggleFeed} title="Stop Feed">⏹</button>
            <button onClick={() => onExpand?.(id)} title="Fullscreen">⛶</button>
            <button
              onClick={toggleRecording}
              title={isRecording ? 'Stop Recording' : 'Start Recording'}
              style={{ color: isRecording ? 'var(--accent-red)' : '' }}
            >
              ⏺
            </button>
            <button onClick={takeSnapshot} title="Snapshot">📸</button>
          </div>
        )}

        <div className="video-info-overlay">
          <div className="video-name">{name}</div>
          <div className="video-meta">
            <span>{location}</span>
            <span className="separator">•</span>
            <span>{resolution}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
