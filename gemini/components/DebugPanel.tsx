'use client';

import React from 'react';

interface DebugPanelProps {
  isVisible: boolean;
  statusMessage: string;
  framesSentCount: number;
  lastProcessedFrameId: number;
  numKeyframes: number;
  slamStatus: string;
  relocRequiredMsg: string | null;
  points: number[][];
  isIncrementalUpdate: boolean;
  sessionId: string | null;
  isConnected: boolean;
  isCameraReady: boolean;
  lastMessage: any;
}

export default function DebugPanel({
  isVisible,
  statusMessage,
  framesSentCount,
  lastProcessedFrameId,
  numKeyframes,
  slamStatus,
  relocRequiredMsg,
  points,
  isIncrementalUpdate,
  sessionId,
  isConnected,
  isCameraReady,
  lastMessage
}: DebugPanelProps) {
  if (!isVisible) {
    return null;
  }

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      gap: '10px', 
      padding: '10px',
      borderTop: '2px solid #ccc',
      background: '#f9f9f9',
      fontSize: '0.9em',
      flexShrink: 0
    }}>
      <h3 style={{ margin: '0 0 10px 0', fontSize: '1.1em', color: '#333' }}>
        Debug Panel
      </h3>
      
      {/* Status Information */}
      <div style={{ 
        padding: '10px', 
        border: '1px solid #ccc', 
        background: '#f0f0f0', 
        borderRadius: '4px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#333' }}>
          System Status
        </div>
        <div style={{ marginBottom: '4px' }}>
          <strong>Status:</strong> {statusMessage}
        </div>
        <div style={{ marginBottom: '4px' }}>
          <strong>Frames (Processed/Sent):</strong> {lastProcessedFrameId === -1 ? 'N/A' : lastProcessedFrameId} / {framesSentCount}
        </div>
        <div style={{ marginBottom: '4px' }}>
          <strong>Keyframes:</strong> {numKeyframes}
        </div>
        <div style={{ marginBottom: '4px' }}>
          <strong>SLAM Mode:</strong> {slamStatus}
        </div>
        <div style={{ marginBottom: '4px' }}>
          <strong>Points:</strong> {points.length}
        </div>
        <div style={{ marginBottom: '4px' }}>
          <strong>Update Type:</strong> {isIncrementalUpdate ? 'Incremental' : 'Full'}
        </div>
        {relocRequiredMsg && (
          <div style={{ 
            color: 'orange', 
            fontWeight: 'bold', 
            marginTop: '8px',
            padding: '8px',
            background: '#fff3cd',
            border: '1px solid #ffeaa7',
            borderRadius: '4px'
          }}>
            <strong>Backend Hint:</strong> {relocRequiredMsg}
          </div>
        )}
      </div>

      {/* Connection Information */}
      <div style={{ 
        display: 'flex', 
        gap: '20px',
        padding: '10px', 
        border: '1px solid #007bff', 
        background: '#e7f3ff', 
        borderRadius: '4px'
      }}>
        <div style={{ fontWeight: 'bold', color: '#333', minWidth: '120px' }}>
          Connection Info:
        </div>
        <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
          <div>
            <strong>Session ID:</strong> {sessionId || 'None'}
          </div>
          <div>
            <strong>WebSocket:</strong> 
            <span style={{ color: isConnected ? 'green' : 'red', fontWeight: 'bold', marginLeft: '4px' }}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div>
            <strong>Camera:</strong> 
            <span style={{ color: isCameraReady ? 'green' : 'red', fontWeight: 'bold', marginLeft: '4px' }}>
              {isCameraReady ? 'Ready' : 'Not Ready'}
            </span>
          </div>
        </div>
      </div>

      {/* Raw Message Display */}
      {lastMessage && (
        <div style={{ 
          border: '1px solid #6c757d', 
          borderRadius: '4px',
          background: '#fff'
        }}>
          <div style={{ 
            padding: '8px', 
            background: '#6c757d', 
            color: 'white', 
            fontWeight: 'bold',
            borderRadius: '4px 4px 0 0'
          }}>
            Last WebSocket Message
          </div>
          <div style={{ 
            padding: '10px',
            maxHeight: '200px',
            overflowY: 'auto',
            fontSize: '0.8em',
            fontFamily: 'monospace',
            wordBreak: 'break-all',
            background: '#f8f9fa'
          }}>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
              {typeof lastMessage === 'string' 
                ? lastMessage 
                : JSON.stringify(lastMessage, null, 2)
              }
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
