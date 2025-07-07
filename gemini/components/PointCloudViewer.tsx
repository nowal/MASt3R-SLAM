'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface PointCloudViewerProps {
  onFrameRendered?: (frameId: number) => void;
  onPointCloudReceived?: (pointCount: number) => void;
}

interface PointCloudUpdate {
  type: string;
  keyframe_idx: number;
  total_keyframes: number;
  transmission_count: number;
  data_type: string;
  compression: boolean;
  point_count: number;
  has_colors: boolean;
  data_size: number;
  timestamp: number;
}

const PointCloudViewer: React.FC<PointCloudViewerProps> = ({
  onFrameRendered,
  onPointCloudReceived,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointsGeometryRef = useRef<THREE.BufferGeometry | null>(null);
  const pointsMaterialRef = useRef<THREE.PointsMaterial | null>(null);
  const pointsObjectRef = useRef<THREE.Points | null>(null);
  
  // State for point cloud data
  const [pointCloudStats, setPointCloudStats] = useState({
    totalPoints: 0,
    totalKeyframes: 0,
    lastUpdate: null as number | null,
    transmissionCount: 0
  });
  
  // Pending point cloud metadata (waiting for binary data)
  const pendingMetadataRef = useRef<PointCloudUpdate | null>(null);

  const initializeScene = useCallback(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    // Initialize scene, camera, and renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101010);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, currentMount.clientWidth / currentMount.clientHeight, 0.1, 1000);
    camera.position.set(0, 2, 5);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    renderer.setClearColor(0x000000, 0); // Transparent background
    rendererRef.current = renderer;
    currentMount.appendChild(renderer.domElement);

    // Initialize controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Initialize point cloud object
    const geometry = new THREE.BufferGeometry();
    pointsGeometryRef.current = geometry;
    const material = new THREE.PointsMaterial({ 
      size: 0.02, 
      vertexColors: true,
      sizeAttenuation: true
    });
    pointsMaterialRef.current = material;
    const points = new THREE.Points(geometry, material);
    scene.add(points);
    pointsObjectRef.current = points;

    const animate = () => {
      requestAnimationFrame(animate);
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    console.log('PointCloudViewer: Scene initialized');
  }, []);

  const handlePointCloudMetadata = useCallback((metadata: PointCloudUpdate) => {
    console.log('PointCloudViewer: Received point cloud metadata:', metadata);
    pendingMetadataRef.current = metadata;
    
    // Update stats
    setPointCloudStats(prev => ({
      ...prev,
      totalKeyframes: metadata.total_keyframes,
      transmissionCount: metadata.transmission_count,
      lastUpdate: metadata.timestamp
    }));
  }, []);

  const decompressDraco = useCallback(async (compressedData: ArrayBuffer, hasColors: boolean): Promise<{ positions: Float32Array, colors?: Uint8Array }> => {
    // For now, we'll implement a simple fallback that assumes raw binary data
    // In a real implementation, you would use the Draco decoder library
    console.log('PointCloudViewer: Decompressing Draco data (fallback to raw binary)');
    
    try {
      // Parse the binary format: [4 bytes: points_size][points_data][4 bytes: colors_size][colors_data]
      const dataView = new DataView(compressedData);
      let offset = 0;
      
      // Read points size
      const pointsSize = dataView.getUint32(offset, true); // little endian
      offset += 4;
      
      // Read points data
      const pointsData = new Float32Array(compressedData.slice(offset, offset + pointsSize));
      offset += pointsSize;
      
      let colorsData: Uint8Array | undefined;
      if (hasColors && offset < compressedData.byteLength) {
        // Read colors size
        const colorsSize = dataView.getUint32(offset, true);
        offset += 4;
        
        // Read colors data
        colorsData = new Uint8Array(compressedData.slice(offset, offset + colorsSize));
      }
      
      return { positions: pointsData, colors: colorsData };
    } catch (error) {
      console.error('PointCloudViewer: Failed to decompress Draco data:', error);
      throw error;
    }
  }, []);

  const parseRawBinary = useCallback((binaryData: ArrayBuffer, hasColors: boolean): { positions: Float32Array, colors?: Uint8Array } => {
    console.log('PointCloudViewer: Parsing raw binary data');
    
    try {
      // Parse the binary format: [4 bytes: points_size][points_data][4 bytes: colors_size][colors_data]
      const dataView = new DataView(binaryData);
      let offset = 0;
      
      // Read points size
      const pointsSize = dataView.getUint32(offset, true); // little endian
      offset += 4;
      
      // Read points data
      const pointsData = new Float32Array(binaryData.slice(offset, offset + pointsSize));
      offset += pointsSize;
      
      let colorsData: Uint8Array | undefined;
      if (hasColors && offset < binaryData.byteLength) {
        // Read colors size
        const colorsSize = dataView.getUint32(offset, true);
        offset += 4;
        
        // Read colors data
        colorsData = new Uint8Array(binaryData.slice(offset, offset + colorsSize));
      }
      
      return { positions: pointsData, colors: colorsData };
    } catch (error) {
      console.error('PointCloudViewer: Failed to parse raw binary data:', error);
      throw error;
    }
  }, []);

  const updatePointCloudGeometry = useCallback((positions: Float32Array, colors?: Uint8Array) => {
    if (!pointsGeometryRef.current) {
      console.error('PointCloudViewer: No geometry reference available');
      return;
    }

    try {
      console.log(`PointCloudViewer: Updating geometry with ${positions.length / 3} points`);
      
      // Set position attribute
      pointsGeometryRef.current.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      
      // Set color attribute if available
      if (colors) {
        // Convert Uint8Array colors to Float32Array (0-1 range)
        const colorFloat32 = new Float32Array(colors.length);
        for (let i = 0; i < colors.length; i++) {
          colorFloat32[i] = colors[i] / 255.0;
        }
        pointsGeometryRef.current.setAttribute('color', new THREE.BufferAttribute(colorFloat32, 3));
      } else {
        // Use default white color
        const defaultColors = new Float32Array(positions.length); // Same length as positions
        defaultColors.fill(1.0); // White
        pointsGeometryRef.current.setAttribute('color', new THREE.BufferAttribute(defaultColors, 3));
      }
      
      // Update geometry
      pointsGeometryRef.current.attributes.position.needsUpdate = true;
      pointsGeometryRef.current.attributes.color.needsUpdate = true;
      pointsGeometryRef.current.computeBoundingSphere();
      
      // Update stats
      const pointCount = positions.length / 3;
      setPointCloudStats(prev => ({
        ...prev,
        totalPoints: pointCount
      }));
      
      // Notify parent component
      if (onPointCloudReceived) {
        onPointCloudReceived(pointCount);
      }
      
      console.log(`PointCloudViewer: Successfully updated point cloud with ${pointCount} points`);
    } catch (error) {
      console.error('PointCloudViewer: Failed to update geometry:', error);
    }
  }, [onPointCloudReceived]);

  const handleBinaryPointCloudData = useCallback(async (binaryData: ArrayBuffer) => {
    if (!pendingMetadataRef.current) {
      console.error('PointCloudViewer: Received binary data without metadata');
      return;
    }

    const metadata = pendingMetadataRef.current;
    pendingMetadataRef.current = null; // Clear pending metadata

    console.log(`PointCloudViewer: Processing binary point cloud data (${binaryData.byteLength} bytes)`);

    try {
      let positions: Float32Array;
      let colors: Uint8Array | undefined;

      if (metadata.compression && metadata.data_type === 'draco_compressed') {
        // Use Draco decompression
        const decompressed = await decompressDraco(binaryData, metadata.has_colors);
        positions = decompressed.positions;
        colors = decompressed.colors;
      } else {
        // Parse raw binary data
        const parsed = parseRawBinary(binaryData, metadata.has_colors);
        positions = parsed.positions;
        colors = parsed.colors;
      }

      // Update the Three.js geometry
      updatePointCloudGeometry(positions, colors);

      // Notify parent component about frame rendering
      if (onFrameRendered) {
        onFrameRendered(metadata.keyframe_idx);
      }

    } catch (error) {
      console.error('PointCloudViewer: Failed to process binary point cloud data:', error);
    }
  }, [decompressDraco, parseRawBinary, updatePointCloudGeometry, onFrameRendered]);

  // Expose methods for parent component to call
  const handleMessage = useCallback((message: any) => {
    if (message.type === 'POINT_CLOUD_UPDATE') {
      handlePointCloudMetadata(message);
    } else if (message.type === 'BINARY_POINT_CLOUD_DATA') {
      handleBinaryPointCloudData(message.data);
    }
  }, [handlePointCloudMetadata, handleBinaryPointCloudData]);

  // Initialize scene on mount
  useEffect(() => {
    initializeScene();

    return () => {
      const currentMount = mountRef.current;
      if (currentMount && rendererRef.current) {
        currentMount.removeChild(rendererRef.current.domElement);
      }
    };
  }, [initializeScene]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const currentMount = mountRef.current;
      if (!currentMount || !rendererRef.current || !cameraRef.current) return;

      const width = currentMount.clientWidth;
      const height = currentMount.clientHeight;

      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Expose handleMessage to parent component via ref
  React.useImperativeHandle(React.forwardRef(() => null), () => ({
    handleMessage
  }));

  return (
    <div ref={mountRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Point cloud stats overlay */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        background: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '8px',
        borderRadius: '4px',
        fontSize: '12px',
        fontFamily: 'monospace',
        zIndex: 1000,
        pointerEvents: 'none'
      }}>
        <div>Points: {pointCloudStats.totalPoints.toLocaleString()}</div>
        <div>Keyframes: {pointCloudStats.totalKeyframes}</div>
        <div>Updates: {pointCloudStats.transmissionCount}</div>
        {pointCloudStats.lastUpdate && (
          <div>Last: {new Date(pointCloudStats.lastUpdate).toLocaleTimeString()}</div>
        )}
      </div>
    </div>
  );
};

// Create a version that can be used with message handling
export const PointCloudViewerWithMessageHandler = React.forwardRef<
  { handleMessage: (message: any) => void },
  PointCloudViewerProps
>((props, ref) => {
  const viewerRef = useRef<{ handleMessage: (message: any) => void }>(null);

  React.useImperativeHandle(ref, () => ({
    handleMessage: (message: any) => {
      if (viewerRef.current) {
        viewerRef.current.handleMessage(message);
      }
    }
  }));

  return <PointCloudViewer {...props} ref={viewerRef} />;
});

PointCloudViewerWithMessageHandler.displayName = 'PointCloudViewerWithMessageHandler';

export default PointCloudViewer;
