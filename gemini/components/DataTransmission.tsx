'use client';

import React, { useRef, useCallback, useEffect } from 'react';

interface DataTransmissionProps {
  isActive: boolean;
  videoElement: HTMLVideoElement | null;
  sendMessage: ((message: any) => boolean) | null;
  onFrameSent: (frameId: number) => void;
  onError: (error: string) => void;
  frameInterval?: number;
  targetWidth?: number;
  targetHeight?: number;
  quality?: number;
}

export default function DataTransmission({
  isActive,
  videoElement,
  sendMessage,
  onFrameSent,
  onError,
  frameInterval = 500,
  targetWidth = 512,
  targetHeight = 384,
  quality = 0.85
}: DataTransmissionProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameIdRef = useRef<number>(0);
  const supportsWebPRef = useRef<boolean | null>(null);

  // Check WebP support
  const checkWebPSupport = useCallback((): boolean => {
    if (supportsWebPRef.current !== null) {
      return supportsWebPRef.current;
    }

    try {
      // More comprehensive WebP support check
      const canvas = document.createElement('canvas');
      canvas.width = 2;
      canvas.height = 2;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        supportsWebPRef.current = false;
        return false;
      }
      
      // Draw a simple pattern
      ctx.fillStyle = '#FF0000';
      ctx.fillRect(0, 0, 1, 1);
      ctx.fillStyle = '#00FF00';
      ctx.fillRect(1, 1, 1, 1);
      
      const dataURL = canvas.toDataURL('image/webp', 0.8);
      const isWebPSupported = dataURL.indexOf('data:image/webp') === 0;
      
      supportsWebPRef.current = isWebPSupported;
      console.log('WebP support detected:', isWebPSupported, 'User Agent:', navigator.userAgent);
      
      return isWebPSupported;
    } catch (e) {
      console.warn('WebP support check failed:', e);
      supportsWebPRef.current = false;
      return false;
    }
  }, []);

  // Convert canvas to binary data
  const canvasToBlob = useCallback((canvas: HTMLCanvasElement, format: string, quality: number): Promise<Blob | null> => {
    return new Promise((resolve) => {
      try {
        canvas.toBlob((blob) => {
          resolve(blob);
        }, format, quality);
      } catch (e) {
        console.error('Canvas to blob conversion failed:', e);
        resolve(null);
      }
    });
  }, []);

  // Send frame data using new binary protocol
  const sendFrameData = useCallback(async () => {
    if (!videoElement || !sendMessage || !canvasRef.current) {
      return;
    }

    if (videoElement.readyState < 2) {
      return; // Video not ready
    }

    try {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      if (!context) {
        onError('Failed to get canvas context');
        return;
      }

      // Set canvas size
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      // Draw video frame to canvas
      context.drawImage(videoElement, 0, 0, targetWidth, targetHeight);

      // Determine format and MIME type
      const supportsWebP = checkWebPSupport();
      const format = supportsWebP ? 'webp' : 'jpeg';
      const mimeType = supportsWebP ? 'image/webp' : 'image/jpeg';

      // Convert to blob
      const blob = await canvasToBlob(canvas, mimeType, quality);
      
      if (!blob) {
        onError(`Failed to convert canvas to ${format} blob`);
        return;
      }

      const currentFrameId = frameIdRef.current;
      const timestamp = Date.now();

      // Step 1: Send metadata as JSON
      const metadata = {
        type: 'FRAME_METADATA',
        frameClientIndex: currentFrameId,
        timestamp: timestamp,
        format: format,
        quality: quality,
        width: targetWidth,
        height: targetHeight,
        size: blob.size
      };

      const metadataSuccess = sendMessage(metadata);
      if (!metadataSuccess) {
        onError('Failed to send frame metadata');
        return;
      }

      // Step 2: Send binary data
      // Note: We need to send the binary data directly through WebSocket
      // This requires access to the raw WebSocket, which we'll handle in the WebSocketConnection component
      const arrayBuffer = await blob.arrayBuffer();
      const binaryData = new Uint8Array(arrayBuffer);

      // For now, we'll send the binary data as a special message type
      // The WebSocketConnection component will need to handle this
      const binarySuccess = sendMessage({
        type: 'BINARY_FRAME_DATA',
        data: Array.from(binaryData) // Convert to array for JSON transmission temporarily
      });

      if (binarySuccess) {
        frameIdRef.current += 1;
        onFrameSent(currentFrameId);
        console.log(`Sent frame ${currentFrameId} as ${format} (${blob.size} bytes)`);
      } else {
        onError('Failed to send binary frame data');
      }

    } catch (error) {
      console.error('Error sending frame data:', error);
      onError(`Frame transmission error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [videoElement, sendMessage, onFrameSent, onError, targetWidth, targetHeight, quality, checkWebPSupport, canvasToBlob]);

  // Start/stop frame transmission
  useEffect(() => {
    if (isActive && videoElement && sendMessage) {
      if (!intervalRef.current) {
        console.log('Starting frame transmission...');
        intervalRef.current = setInterval(sendFrameData, frameInterval);
      }
    } else {
      if (intervalRef.current) {
        console.log('Stopping frame transmission...');
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isActive, videoElement, sendMessage, sendFrameData, frameInterval]);

  // Reset frame counter when transmission starts
  useEffect(() => {
    if (isActive) {
      frameIdRef.current = 0;
    }
  }, [isActive]);

  return (
    <>
      {/* Hidden canvas for frame processing */}
      <canvas 
        ref={canvasRef} 
        style={{ display: 'none' }} 
        width={targetWidth} 
        height={targetHeight}
      />
    </>
  );
}
