'use client';

import React, { useState, useEffect, useRef, useCallback, useMemo, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Line } from '@react-three/drei';
import * as THREE from 'three'; // Import three.js

// Constants
const BACKEND_URL_HTTP = 'https://api.takeshapehome.com:443'; // Your backend URL for HTTP
const BACKEND_URL_WS = 'wss://api.takeshapehome.com:443';    // Your backend URL for WebSocket
const FRAME_SEND_INTERVAL = 500; // Send a frame every 500ms (2 FPS) - Adjust as needed
const FRAME_WIDTH = 640; // Optional: Specify frame width for canvas
const FRAME_HEIGHT = 480; // Optional: Specify frame height for canvas

// --- 3D Visualization Components ---

interface PointCloudViewerProps {
  pointsData: number[][]; // Array of [x, y, z]
  colorsData: number[][]; // Array of [r, g, b] (0-1 range)
  posesData: { position: number[], orientation: number[] }[]; // Array of camera poses
  isIncremental: boolean; // Whether this is an incremental update
}

function PointCloudViewer({ pointsData, colorsData, posesData, isIncremental }: PointCloudViewerProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const [initialized, setInitialized] = useState(false);
  
  // Store accumulated points and colors
  const accumulatedPointsRef = useRef<number[][]>([]);
  const accumulatedColorsRef = useRef<number[][]>([]);
  
  // Handle incremental updates
  useEffect(() => {
    if (!pointsData || pointsData.length === 0) return;
    
    // If this is an incremental update, add to existing points
    if (isIncremental && initialized) {
      accumulatedPointsRef.current = [...accumulatedPointsRef.current, ...pointsData];
      accumulatedColorsRef.current = [...accumulatedColorsRef.current, ...colorsData];
      console.log(`Added ${pointsData.length} new points to existing ${accumulatedPointsRef.current.length - pointsData.length} points`);
    } else {
      // Otherwise replace all points
      accumulatedPointsRef.current = [...pointsData];
      accumulatedColorsRef.current = [...colorsData];
      console.log(`Replaced all points with ${pointsData.length} new points`);
    }
  }, [pointsData, colorsData, isIncremental, initialized]);
  
  // Set up the scene when accumulated points change
  useEffect(() => {
    if (!accumulatedPointsRef.current || accumulatedPointsRef.current.length === 0 || !groupRef.current) return;
    
    // Clean up previous instances
    while (groupRef.current.children.length > 0) {
      const child = groupRef.current.children[0];
      groupRef.current.remove(child);
      if (child instanceof THREE.Mesh) {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      }
    }
    
    // Create a small sphere geometry for each point
    const geometry = new THREE.SphereGeometry(0.01, 4, 4);
    
    // Create materials for each point (or reuse by color)
    const materials = new Map();
    const meshes = [];
    
    // Process each point
    for (let i = 0; i < accumulatedPointsRef.current.length; i++) {
      const point = accumulatedPointsRef.current[i];
      const color = accumulatedColorsRef.current && i < accumulatedColorsRef.current.length 
        ? new THREE.Color(
            accumulatedColorsRef.current[i][0], // Already in 0-1 range
            accumulatedColorsRef.current[i][1], 
            accumulatedColorsRef.current[i][2]
          ) 
        : new THREE.Color(1, 1, 1);
      
      // Get or create material for this color
      const colorKey = color.getHexString();
      if (!materials.has(colorKey)) {
        materials.set(colorKey, new THREE.MeshBasicMaterial({ color }));
      }
      const material = materials.get(colorKey);
      
      // Create mesh for this point
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(point[0], point[1], point[2]);
      meshes.push(mesh);
    }
    
    // Add all meshes to the group
    meshes.forEach(mesh => {
      groupRef.current.add(mesh);
    });
    
    setInitialized(true);
    console.log(`Rendered ${meshes.length} points`);
  }, [accumulatedPointsRef.current, accumulatedColorsRef.current]);
  
  // Render camera poses
  useEffect(() => {
    if (!posesData || posesData.length === 0 || !groupRef.current) return;
    
    // Remove existing camera poses
    const posesToRemove: THREE.Object3D[] = [];
    groupRef.current.children.forEach(child => {
      if (child instanceof THREE.Mesh && child.geometry instanceof THREE.BoxGeometry) {
        posesToRemove.push(child);
      }
      if (child instanceof THREE.Line) {
        posesToRemove.push(child);
      }
    });
    posesToRemove.forEach(pose => {
      groupRef.current.remove(pose);
      if (pose instanceof THREE.Mesh) {
        if (pose.geometry) pose.geometry.dispose();
        if (pose.material) {
          if (Array.isArray(pose.material)) {
            pose.material.forEach(m => m.dispose());
          } else {
            pose.material.dispose();
          }
        }
      }
    });
    
    // Create camera frustum visualization
    posesData.forEach((pose, index) => {
      const isLatest = index === posesData.length - 1;
      const color = isLatest ? 0xff0000 : 0x00ffff;
      
      // Create a small box to represent the camera
      const boxGeometry = new THREE.BoxGeometry(0.03, 0.03, 0.06);
      const boxMaterial = new THREE.MeshBasicMaterial({ color });
      const box = new THREE.Mesh(boxGeometry, boxMaterial);
      
      // Position the camera
      box.position.set(
        pose.position[0],
        pose.position[1],
        pose.position[2]
      );
      
      // Set orientation (if available)
      if (pose.orientation) {
        const [w, x, y, z] = pose.orientation;
        box.quaternion.set(x, y, z, w);
      }
      
      groupRef.current.add(box);
    });
    
    // Add trajectory line if we have multiple poses
    if (posesData.length > 1) {
      const points = posesData.map(p => new THREE.Vector3(p.position[0], p.position[1], p.position[2]));
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
      const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffff00 });
      const line = new THREE.Line(lineGeometry, lineMaterial);
      groupRef.current.add(line);
    }
  }, [posesData]);

  return (
    <group ref={groupRef} />
  );
}

// --- Main Component ---
function FastApiCameraTester() {
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
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const webSocketRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameIdCounterRef = useRef<number>(0);

  // 1. Connect to backend
  const connectToBackend = useCallback(async () => {
    setStatusMessage('Connecting...');
    // Reset all relevant state on new connection attempt
    setSessionId(null); setIsConnected(false); setIsCameraReady(false);
    setFramesSentCount(0); frameIdCounterRef.current = 0; setLastProcessedFrameId(-1);
    setNumKeyframes(0); setSlamStatus('CONNECTING'); setRelocRequiredMsg(null);
    setPoints([]); setPointColors([]); setPoses([]); // Reset visualization data
    webSocketRef.current?.close(); if (intervalRef.current) clearInterval(intervalRef.current);

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
        webSocketRef.current?.close();
        if (intervalRef.current) clearInterval(intervalRef.current);
     };
  }, [connectToBackend]); // Dependency is stable

  // 2. Establish WebSocket connection
  useEffect(() => {
    if (!sessionId) return; // Only run if sessionId is valid

    console.log(`WS Effect: Attempting WebSocket connection for ${sessionId}`);
    setStatusMessage(`WebSocket connecting...`);
    const ws = new WebSocket(`${BACKEND_URL_WS}/ws/${sessionId}`);
    webSocketRef.current = ws;

    ws.onopen = () => {
      console.log('WS Effect: WebSocket Connected');
      setIsConnected(true); // Trigger camera setup
      setStatusMessage('WebSocket Connected');
      setSlamStatus('INIT'); // Assume starting in INIT
      setRelocRequiredMsg(null);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setLastMessage(message); // Store raw message

        // Handle different message types from backend
        if (message.type === "slam_update") {
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
      } catch (e) { console.error("WS Effect: Failed to parse message:", event.data, e); setLastMessage(`Non-JSON: ${event.data}`); }
    };

    ws.onerror = (error) => {
        console.error('WS Effect: WebSocket Error:', error);
        setStatusMessage(`WebSocket Error. Check console/backend.`);
        setIsConnected(false); setSlamStatus('ERROR'); setRelocRequiredMsg(null);
     };
    ws.onclose = (event) => {
        console.log(`WS Effect: WebSocket Closed (Code: ${event.code}, Clean: ${event.wasClean})`);
        setIsConnected(false); setIsCameraReady(false); setSlamStatus('CLOSED'); setRelocRequiredMsg(null);
        if (webSocketRef.current === ws) webSocketRef.current = null;
        if (!event.wasClean) { setStatusMessage(`WS Closed unexpectedly. Retrying in 5s...`); setTimeout(connectToBackend, 5000); }
        else { setStatusMessage('WebSocket Closed.'); }
     };

    // Cleanup for this effect instance
    return () => {
      console.log("WS Effect: Cleaning up WebSocket instance for", sessionId);
       if (ws) {
           ws.onclose = null; ws.onerror = null; ws.onmessage = null; ws.onopen = null;
           if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) { ws.close(); console.log("WS Effect: Closing WS"); }
       }
       if (webSocketRef.current === ws) webSocketRef.current = null;
       setIsConnected(false);
    };
  }, [sessionId, connectToBackend]); // Depend on sessionId and stable connect function

  // 3. Setup Camera
  useEffect(() => {
    if (!isConnected) { if(isCameraReady) setIsCameraReady(false); return; }
    let stream: MediaStream | null = null; let videoEl = videoRef.current;
    const setupCamera = async () => {
      console.log("Camera Effect: Setting up camera...");
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoEl) {
          videoEl.srcObject = stream;
          const handleMetadataLoaded = () => { console.log('Camera Effect: Camera ready.'); setIsCameraReady(true); videoEl?.removeEventListener('loadedmetadata', handleMetadataLoaded); };
          videoEl.addEventListener('loadedmetadata', handleMetadataLoaded);
        } else { throw new Error("Video element ref null"); }
      } catch (err) { console.error('Camera Effect: Error accessing camera:', err); const errorMsg = err instanceof Error ? err.message : String(err); setStatusMessage(`Camera Error: ${errorMsg}`); setIsCameraReady(false); }
    };
    setupCamera();
    return () => { console.log("Camera Effect: Cleaning up camera..."); stream?.getTracks().forEach(track => track.stop()); if(videoEl) videoEl.srcObject = null; setIsCameraReady(false); };
  }, [isConnected]);

  // 4. Send frames periodically
  useEffect(() => {
    if (isConnected && isCameraReady && webSocketRef.current?.readyState === WebSocket.OPEN) {
      if (!intervalRef.current) {
        console.log("Interval Effect: Starting frame send interval...");
        intervalRef.current = setInterval(() => {
          const video = videoRef.current; const canvas = canvasRef.current; const ws = webSocketRef.current;
          if (video && canvas && ws && ws.readyState === WebSocket.OPEN) {
            const context = canvas.getContext('2d');
            if (context) {
              canvas.width = FRAME_WIDTH; canvas.height = FRAME_HEIGHT;
              context.drawImage(video, 0, 0, canvas.width, canvas.height);
              const frameDataUrl = canvas.toDataURL('image/jpeg', 0.7);
              const currentFrameClientIndex = frameIdCounterRef.current;
              const message = { type: 'FRAME', payload: frameDataUrl, timestamp: Date.now(), frameClientIndex: currentFrameClientIndex };
              ws.send(JSON.stringify(message));
              frameIdCounterRef.current += 1; setFramesSentCount(prevCount => prevCount + 1);
            }
          } else { if (intervalRef.current) clearInterval(intervalRef.current); intervalRef.current = null; }
        }, FRAME_SEND_INTERVAL);
      }
    } else { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }
    return () => { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } };
  }, [isConnected, isCameraReady]);


  // --- JSX Rendering ---
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: '10px', fontFamily: 'sans-serif', gap: '10px' }}>
      <h2>FastAPI Camera + SLAM Viewer</h2>

      {/* Status Bar */}
      <div style={{ padding: '10px', border: '1px solid #ccc', background: '#f0f0f0', fontSize: '0.9em', flexShrink: 0 }}>
        <strong>Status:</strong> {statusMessage} |{' '}
        <strong>Frames (Processed/Sent):</strong> {lastProcessedFrameId === -1 ? 'N/A' : lastProcessedFrameId} / {framesSentCount} |{' '}
        <strong>Keyframes:</strong> {numKeyframes} |{' '}
        <strong>Mode:</strong> {slamStatus} |{' '}
        <strong>Points:</strong> {points.length} |{' '}
        <strong>Update Type:</strong> {isIncrementalUpdate ? 'Incremental' : 'Full'}
        {relocRequiredMsg && <div style={{color: 'orange', fontWeight: 'bold', marginTop:'5px'}}> Backend Hint: {relocRequiredMsg}</div>}
      </div>

      {/* Layout: 3D View takes most space, Video feed smaller */}
      <div style={{ flexGrow: 1, display: 'flex', gap: '10px', overflow: 'hidden' }}>

          {/* 3D Canvas */}
          <div style={{ flex: '1', border: '1px solid #ccc', position: 'relative', background:'#272727' }}>
              {/* Only render Canvas if WebSocket is connected to avoid initial errors */}
              {isConnected && (
                  <Canvas camera={{ position: [0, 0.5, -2], fov: 75 }}>
                      <color attach="background" args={['#272727']} />
                      <ambientLight intensity={0.8} />
                      <pointLight position={[5, 5, -5]} intensity={1} />
                      <axesHelper args={[0.5]} />

                      {/* New point cloud renderer */}
                      <Suspense fallback={null}>
                        {points.length > 0 && (
                          <PointCloudViewer 
                            pointsData={points} 
                            colorsData={pointColors} 
                            posesData={poses} 
                            isIncremental={isIncrementalUpdate}
                          />
                        )}
                      </Suspense>

                      <OrbitControls enableZoom={true} makeDefault />
                  </Canvas>
              )}
              {!isConnected && <div style={{position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.8)', color:'white', display:'flex', justifyContent:'center', alignItems:'center'}}>Connecting... Please Wait...</div>}
          </div>

          {/* Video Feed & Debug */}
          <div style={{ width: '320px', flexShrink: 0, display:'flex', flexDirection:'column', gap: '10px', overflow: 'hidden' }}>
              <div style={{ position: 'relative', width: `320px`, height: `240px`, border: '1px solid black', background: '#333' }}>
                  <video ref={videoRef} autoPlay playsInline muted style={{ display: 'block', width: '100%', height: '100%', objectFit: 'cover' }}/>
                  {!isCameraReady && isConnected && ( <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(0,0,0,0.7)', color: 'white', display: 'flex', justifyContent: 'center', alignItems: 'center' }}> Waiting for camera... </div> )}
              </div>
              {/* Raw Message Display (Optional) */}
              <div style={{ border: '1px dashed blue', padding: '5px', fontSize: '0.9em', flexShrink:0 }}>
                  <div><strong>[Debug] Session ID:</strong> {sessionId || 'None'}</div>
                  <div><strong>[Debug] WS Connected:</strong> {isConnected ? 'Yes' : 'No'}</div>
                  <div><strong>[Debug] Camera Ready:</strong> {isCameraReady ? 'Yes' : 'No'}</div>
              </div>
              {lastMessage && (
                  <div style={{ wordBreak: 'break-all', background: '#eee', padding:'5px', fontSize:'0.8em', flexGrow: 1, overflowY: 'auto' }}>
                      <strong>Last Raw Msg:</strong> <pre>{JSON.stringify(lastMessage, null, 2)}</pre>
                  </div>
              )}
          </div>
      </div>

      {/* Hidden canvas */}
      <canvas ref={canvasRef} style={{ display: 'none' }} width={FRAME_WIDTH} height={FRAME_HEIGHT}></canvas>
    </div>
  );
}

export default FastApiCameraTester;
