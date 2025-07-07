'use client';

import React, { useRef, useEffect, forwardRef } from 'react';
import { PointCloudViewerWithMessageHandler } from './PointCloudViewer';

interface ModelOverlayProps {
  pointsData: number[][];
  colorsData: number[][];
  posesData: { position: number[], orientation: number[] }[];
  isIncrementalUpdate: boolean;
  pointCloudViewerRef?: React.RefObject<{ handleMessage: (message: any) => void }>;
}

const ModelOverlay = forwardRef<{ handleMessage: (message: any) => void }, ModelOverlayProps>(({
  pointsData, 
  colorsData, 
  posesData, 
  isIncrementalUpdate,
  pointCloudViewerRef
}, ref) => {
  const internalViewerRef = useRef<{ handleMessage: (message: any) => void }>(null);
  
  // Use external ref if provided, otherwise use internal ref
  const viewerRef = pointCloudViewerRef || internalViewerRef;

  // Forward ref to parent component
  React.useImperativeHandle(ref, () => ({
    handleMessage: (message: any) => {
      if (viewerRef.current) {
        viewerRef.current.handleMessage(message);
      }
    }
  }));

  // Handle point cloud received callback
  const handlePointCloudReceived = (pointCount: number) => {
    console.log(`ModelOverlay: Received point cloud with ${pointCount} points`);
  };

  // Handle frame rendered callback
  const handleFrameRendered = (frameId: number) => {
    console.log(`ModelOverlay: Frame ${frameId} rendered`);
  };

  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '25%',
        backgroundColor: 'rgba(0, 0, 0, 0.3)',
        borderRadius: '0 0 20px 20px',
        overflow: 'hidden',
        zIndex: 5,
      }}
    >
      <PointCloudViewerWithMessageHandler
        ref={viewerRef}
        onPointCloudReceived={handlePointCloudReceived}
        onFrameRendered={handleFrameRendered}
      />
      
      {/* Fallback message when no point cloud data */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'rgba(255, 255, 255, 0.7)',
          fontSize: '14px',
          fontWeight: '500',
          textAlign: 'center',
          pointerEvents: 'none',
          zIndex: 10,
        }}
      >
        3D Point Cloud Preview
        <br />
        <span style={{ fontSize: '12px', opacity: 0.8 }}>
          Waiting for SLAM data...
        </span>
      </div>
    </div>
  );
});

ModelOverlay.displayName = 'ModelOverlay';

export default ModelOverlay;
