'use client';

import React, { useEffect, useRef, useState } from 'react';

interface CameraPermissionProps {
  onCameraReady: (stream: MediaStream) => void;
  onError: (error: string) => void;
  width?: number;
  height?: number;
}

export default function CameraPermission({ 
  onCameraReady, 
  onError, 
  width = 320, 
  height = 240 
}: CameraPermissionProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isCameraReady, setIsCameraReady] = useState<boolean>(false);
  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    let currentStream: MediaStream | null = null;
    const videoEl = videoRef.current;

    const setupCamera = async () => {
      console.log("CameraPermission: Setting up camera...");
      try {
        currentStream = await navigator.mediaDevices.getUserMedia({ 
          video: {
            width: { ideal: width },
            height: { ideal: height }
          }
        });
        
        setStream(currentStream);
        
        if (videoEl) {
          videoEl.srcObject = currentStream;
          
          const handleMetadataLoaded = () => {
            console.log('CameraPermission: Camera ready.');
            setIsCameraReady(true);
            onCameraReady(currentStream!);
            videoEl.removeEventListener('loadedmetadata', handleMetadataLoaded);
          };
          
          videoEl.addEventListener('loadedmetadata', handleMetadataLoaded);
        } else {
          throw new Error("Video element ref is null");
        }
      } catch (err) {
        console.error('CameraPermission: Error accessing camera:', err);
        const errorMsg = err instanceof Error ? err.message : String(err);
        
        // Provide more specific error messages
        let userFriendlyError = errorMsg;
        if (errorMsg.includes('Permission denied')) {
          userFriendlyError = 'Camera permission denied. Please allow camera access and refresh.';
        } else if (errorMsg.includes('NotFoundError')) {
          userFriendlyError = 'No camera found. Please connect a camera and try again.';
        } else if (errorMsg.includes('NotAllowedError')) {
          userFriendlyError = 'Camera access blocked. Please check browser permissions.';
        }
        
        onError(`Camera Error: ${userFriendlyError}`);
        setIsCameraReady(false);
      }
    };

    setupCamera();

    // Cleanup function
    return () => {
      console.log("CameraPermission: Cleaning up camera...");
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
      }
      if (videoEl) {
        videoEl.srcObject = null;
      }
      setIsCameraReady(false);
      setStream(null);
    };
  }, [onCameraReady, onError, width, height]);

  return (
    <div style={{ 
      position: 'relative', 
      width: `${width}px`, 
      height: `${height}px`, 
      background: '#333' 
    }}>
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted 
        style={{ 
          display: 'block', 
          width: '100%', 
          height: '100%', 
          objectFit: 'cover' 
        }}
      />
      {!isCameraReady && (
        <div style={{ 
          position: 'absolute', 
          top: 0, 
          left: 0, 
          width: '100%', 
          height: '100%', 
          background: 'rgba(0,0,0,0.7)', 
          color: 'white', 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          fontSize: '14px'
        }}>
          Requesting camera access...
        </div>
      )}
    </div>
  );
}
