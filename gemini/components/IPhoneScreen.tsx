'use client';

import React from 'react';
import CameraPermission from './CameraPermission';
import ModelOverlay from './ModelOverlay';
import RecordingToggleButton from './RecordingToggleButton';

interface IPhoneScreenProps {
  onCameraReady: (stream: MediaStream) => void;
  onCameraError: (error: string) => void;
  isRecording: boolean;
  onRecordingToggle: () => void;
  pointsData: number[][];
  colorsData: number[][];
  posesData: { position: number[], orientation: number[] }[];
  isIncrementalUpdate: boolean;
}

export default function IPhoneScreen({
  onCameraReady,
  onCameraError,
  isRecording,
  onRecordingToggle,
  pointsData,
  colorsData,
  posesData,
  isIncrementalUpdate
}: IPhoneScreenProps) {
  return (
    <div
      style={{
        position: 'relative',
        width: '390px',
        height: '844px',
        backgroundColor: '#000',
        borderRadius: '30px',
        overflow: 'hidden',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
        border: '8px solid #1c1c1e',
        margin: '0 auto',
      }}
    >
      {/* Camera Feed - Full Background */}
      <CameraPermission
        onCameraReady={onCameraReady}
        onError={onCameraError}
        width={390}
        height={844}
      />
      
      {/* Model Overlay - Top 25% */}
      <ModelOverlay
        pointsData={pointsData}
        colorsData={colorsData}
        posesData={posesData}
        isIncrementalUpdate={isIncrementalUpdate}
      />
      
      {/* Recording Toggle Button - Bottom Center */}
      <RecordingToggleButton
        isRecording={isRecording}
        onToggle={onRecordingToggle}
      />
    </div>
  );
}
