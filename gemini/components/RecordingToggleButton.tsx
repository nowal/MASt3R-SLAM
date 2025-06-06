'use client';

import React from 'react';

interface RecordingToggleButtonProps {
  isRecording: boolean;
  onToggle: () => void;
}

export default function RecordingToggleButton({ isRecording, onToggle }: RecordingToggleButtonProps) {
  return (
    <button
      onClick={onToggle}
      style={{
        position: 'absolute',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '70px',
        height: '70px',
        border: isRecording ? 'none' : '3px solid rgba(255, 255, 255, 0.8)',
        borderRadius: isRecording ? '12px' : '50%',
        backgroundColor: isRecording ? '#ff3b30' : 'rgba(255, 255, 255, 0.9)',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10,
        outline: 'none',
        WebkitTapHighlightColor: 'transparent',
      }}
      onMouseDown={(e) => {
        e.currentTarget.style.transform = 'translateX(-50%) scale(0.95)';
      }}
      onMouseUp={(e) => {
        e.currentTarget.style.transform = 'translateX(-50%) scale(1)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateX(-50%) scale(1)';
      }}
    >
      {isRecording && (
        <div
          style={{
            width: '24px',
            height: '24px',
            backgroundColor: 'white',
            borderRadius: '3px',
            animation: 'pulse 1.5s infinite',
          }}
        />
      )}
      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.7;
          }
        }
      `}</style>
    </button>
  );
}
