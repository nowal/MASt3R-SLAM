'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';

// Import our new components
import IPhoneScreen from './components/IPhoneScreen';
import WebSocketConnection from './components/WebSocketConnection';
import DebugPanel from './components/DebugPanel';
import DataTransmission from './components/DataTransmission';

// Constants
const BACKEND_URL_HTTP = 'https://api.takeshapehome.com:8443'; // Your backend URL for HTTP
const BACKEND_URL_WS = 'wss://api.takeshapehome.com:8443';    // Your backend URL for WebSocket
const FRAME_SEND_INTERVAL = 500; // Send a frame every 500ms (2 FPS) - Adjust as needed
const FRAME_WIDTH = 512; // Updated to match SLAM processing
const FRAME_HEIGHT = 384; // Updated to match SLAM processing (512x384)

// --- Main Component ---
function FastApiCameraTester() {
  // Debug toggle
  const [debug, setDebug] = useState<boolean>(false); // Set to false by default for clean UI

  // Recording state - controls frame sending
  const [isRecording, setIsRecording] = useState<boolean>(false);

  // Core state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isCameraReady, setIsCameraReady] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Initializing...');
  const [lastMessage, setLastMessage] = useState<any>(null); // For raw debug display

  // State for display data
  const [framesSentCount, setFramesSentCount] = useState<number>(0);
  const [lastProcessedFrameId, setLastProcessedFrameId] = useState<number>(-1);
  const [numKeyframes, setNumKeyframes] = useState<number>(0);
  const [slamStatus, setSlamStatus] = useState<string>('UNKNOWN');
  const [relocRequiredMsg, setRelocRequiredMsg] = useState<string | null>(null);
  
  // State for visualization data
  const [points, setPoints] = useState<number[][]>([]);
  const [pointColors, setPointColors] = useState<number[][]>([]);
  const [poses, setPoses] = useState<{ position: number[], orientation: number[] }[]>([]);
  const [isIncrementalUpdate, setIsIncrementalUpdate] = useState<boolean>(false);

  // Refs
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const sendMessageRef = useRef<((message: any) => boolean) | null>(null);
  const videoStreamRef = useRef<MediaStream | null>(null);
  const videoElementRef = useRef<HTMLVideoElement | null>(null);

  // 1. Connect to backend
  const connectToBackend = useCallback(async () => {
    setStatusMessage('Connecting to backend...');
    // Reset WebSocket and backend related state (but NOT camera state)
    setSessionId(null); setIsConnected(false);
    setFramesSentCount(0); setLastProcessedFrameId(-1);
    setNumKeyframes(0); setSlamStatus('CONNECTING'); setRelocRequiredMsg(null);
    setPoints([]); setPointColors([]); setPoses([]); // Reset visualization data
    if (intervalRef.current) clearInterval(intervalRef.current);

    console.log("Attempting connection via /connect...");
    try {
      const response = await fetch(`${BACKEND_URL_HTTP}/connect`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
      if (!response.ok) {
          let errorDetail = `HTTP error! Status: ${response.status}`;
          try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (_) {}
          throw new Error(errorDetail);
       }
      const data = await response.json();
      if (data.sessionId) { setSessionId(data.sessionId); setStatusMessage(`Session ID obtained`); }
      else { throw new Error('Session ID not received'); }
    } catch (error) {
        console.error('Failed to connect:', error);
        const errorMsg = error instanceof Error ? error.message : String(error);
        setStatusMessage(`Connect Error: ${errorMsg}. Accepted cert?`);
        setSessionId(null); setSlamStatus('ERROR');
     }
  }, []);

  // Effect to connect on mount
  useEffect(() => {
    connectToBackend();
    return () => { // Cleanup on component unmount
        console.log("Cleaning up component mount effect");
        if (intervalRef.current) clearInterval(intervalRef.current);
     };
  }, [connectToBackend]); // Dependency is stable

  // Handle camera ready - now receives both stream and video element
  const handleCameraReady = useCallback((stream: MediaStream, videoElement: HTMLVideoElement) => {
    console.log("Main: Camera ready, stream and video element received");
    videoStreamRef.current = stream;
    videoElementRef.current = videoElement; // Use the shared video element from CameraPermission
    
    setIsCameraReady(true);
  }, []);

  // Handle camera error
  const handleCameraError = useCallback((error: string) => {
    console.error("Main: Camera error:", error);
    setStatusMessage(error);
    setIsCameraReady(false);
  }, []);

  // Handle recording toggle
  const handleRecordingToggle = useCallback(() => {
    setIsRecording(prev => !prev);
    console.log("Recording toggled:", !isRecording);
  }, [isRecording]);

  // Handle WebSocket connection change
  const handleConnectionChange = useCallback((connected: boolean) => {
    setIsConnected(connected);
    if (connected) {
      setStatusMessage('WebSocket Connected');
      setSlamStatus('INIT'); // Assume starting in INIT
      setRelocRequiredMsg(null);
    }
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((message: any) => {
    setLastMessage(message); // Store raw message

    // Handle different message types from backend
    if (message.type === "debug_log") {
        // Handle debug messages from backend
        console.log(`[Backend ${message.level}] ${message.message}`);
        if (message.level === "error") {
            setStatusMessage(`Backend Error: ${message.message}`);
        } else if (message.level === "success") {
            setStatusMessage(message.message);
        }
    } else if (message.type === "frame_processed") {
        // Handle frame processing results
        if (typeof message.frame_id === 'number') {
            setLastProcessedFrameId(message.frame_id);
            console.log(`Frame ${message.frame_id} processed successfully`);
        }
    } else if (message.type === "frame_error") {
        // Handle frame processing errors
        console.error(`Frame ${message.frame_id} processing failed: ${message.error}`);
        setStatusMessage(`Frame processing error: ${message.error}`);
    } else if (message.type === "slam_update") {
        if (typeof message.processedFrameId === 'number') setLastProcessedFrameId(message.processedFrameId);
        if (typeof message.numKeyframes === 'number') setNumKeyframes(message.numKeyframes);
        if (typeof message.status === 'string') {
             setSlamStatus(message.status);
             if (message.status !== 'RELOC') setRelocRequiredMsg(null); // Clear prompt if not in RELOC
             setStatusMessage(`Mode: ${message.status} | KFs: ${message.numKeyframes}`);
        }
        
        // --- UPDATE VISUALIZATION STATE ---
        // Check if this is an incremental update
        setIsIncrementalUpdate(message.isIncrementalPointCloud === true);
        
        if (Array.isArray(message.latestKfWorldPoints)) setPoints(message.latestKfWorldPoints);
        if (Array.isArray(message.latestKfColors)) setPointColors(message.latestKfColors);
        if (Array.isArray(message.keyframePoses)) setPoses(message.keyframePoses);
        // ---------------------------------
    } else if (message.type === "RELOC_NEEDED") {
         console.warn("Received RELOC_NEEDED message");
         setRelocRequiredMsg(message.message || "Tracking lost. Move camera to known area.");
         setStatusMessage('Relocalization Required!');
         setSlamStatus('RELOC'); // Ensure status reflects RELOC
    } else if (message.type === "RELOC_SUCCESS") {
         console.info("Received RELOC_SUCCESS message");
         setRelocRequiredMsg(null);
         setStatusMessage(message.message || "Tracking re-established!");
         // Mode will change via next slam_update
    } else if (message.type === "error") {
        console.error("Backend Error Message:", message.message);
        setStatusMessage(`Backend Error: ${message.message}`);
        setSlamStatus('ERROR');
    }
  }, []);

  // Handle WebSocket errors
  const handleWebSocketError = useCallback((error: string) => {
    console.error("Main: WebSocket error:", error);
    setStatusMessage(error);
    setSlamStatus('ERROR');
    setRelocRequiredMsg(null);
  }, []);

  // Handle WebSocket send message ready
  const handleSendMessageReady = useCallback((sendMessage: (message: any) => boolean) => {
    sendMessageRef.current = sendMessage;
  }, []);

  // Handle frame sent callback
  const handleFrameSent = useCallback((frameId: number) => {
    setFramesSentCount(prevCount => prevCount + 1);
    console.log(`Frame ${frameId} sent successfully`);
  }, []);

  // Handle data transmission error
  const handleDataTransmissionError = useCallback((error: string) => {
    console.error("Data transmission error:", error);
    setStatusMessage(`Transmission Error: ${error}`);
  }, []);

  // --- JSX Rendering ---
  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      minHeight: '100vh', 
      padding: '20px', 
      fontFamily: 'sans-serif', 
      gap: '20px',
      backgroundColor: '#f5f5f7',
      alignItems: 'center'
    }}>
      
      {/* iPhone Screen - Main Focus */}
      <IPhoneScreen
        onCameraReady={handleCameraReady}
        onCameraError={handleCameraError}
        isRecording={isRecording}
        onRecordingToggle={handleRecordingToggle}
        pointsData={points}
        colorsData={pointColors}
        posesData={poses}
        isIncrementalUpdate={isIncrementalUpdate}
      />

      {/* Debug Toggle Button - Bottom */}
      <button 
        onClick={() => setDebug(!debug)}
        style={{
          padding: '12px 24px',
          background: debug ? '#dc3545' : '#007AFF',
          color: 'white',
          border: 'none',
          borderRadius: '25px',
          cursor: 'pointer',
          fontSize: '16px',
          fontWeight: '600',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          transition: 'all 0.2s ease',
        }}
        onMouseDown={(e) => {
          e.currentTarget.style.transform = 'scale(0.95)';
        }}
        onMouseUp={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
        }}
      >
        {debug ? 'Hide Debug' : 'Show Debug'}
      </button>

      {/* Debug Panel - Bottom (when visible) */}
      <DebugPanel
        isVisible={debug}
        statusMessage={statusMessage}
        framesSentCount={framesSentCount}
        lastProcessedFrameId={lastProcessedFrameId}
        numKeyframes={numKeyframes}
        slamStatus={slamStatus}
        relocRequiredMsg={relocRequiredMsg}
        points={points}
        isIncrementalUpdate={isIncrementalUpdate}
        sessionId={sessionId}
        isConnected={isConnected}
        isCameraReady={isCameraReady}
        lastMessage={lastMessage}
      />

      {/* WebSocket Connection Component */}
      <WebSocketConnection
        sessionId={sessionId}
        backendUrl={BACKEND_URL_WS}
        onConnectionChange={handleConnectionChange}
        onMessage={handleWebSocketMessage}
        onError={handleWebSocketError}
        onSendMessageReady={handleSendMessageReady}
      />

      {/* Data Transmission Component */}
      <DataTransmission
        isActive={isConnected && isCameraReady && isRecording}
        videoElement={videoElementRef.current}
        sendMessage={sendMessageRef.current}
        onFrameSent={handleFrameSent}
        onError={handleDataTransmissionError}
        frameInterval={FRAME_SEND_INTERVAL}
        targetWidth={FRAME_WIDTH}
        targetHeight={FRAME_HEIGHT}
        quality={0.85}
      />
    </div>
  );
}

export default FastApiCameraTester;
