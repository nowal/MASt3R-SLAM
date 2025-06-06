'use client';

import React, { useRef, useEffect, useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface ModelOverlayProps {
  pointsData: number[][];
  colorsData: number[][];
  posesData: { position: number[], orientation: number[] }[];
  isIncrementalUpdate: boolean;
}

function PointCloudViewer({ pointsData, colorsData, posesData, isIncremental }: {
  pointsData: number[][];
  colorsData: number[][];
  posesData: { position: number[], orientation: number[] }[];
  isIncremental: boolean;
}) {
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
    } else {
      // Otherwise replace all points
      accumulatedPointsRef.current = [...pointsData];
      accumulatedColorsRef.current = [...colorsData];
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
            accumulatedColorsRef.current[i][0],
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

  return <group ref={groupRef} />;
}

export default function ModelOverlay({ 
  pointsData, 
  colorsData, 
  posesData, 
  isIncrementalUpdate 
}: ModelOverlayProps) {
  const hasData = pointsData && pointsData.length > 0;

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
      {hasData ? (
        <Canvas 
          camera={{ position: [0, 0.5, -2], fov: 75 }}
          style={{ background: 'transparent' }}
        >
          <ambientLight intensity={0.8} />
          <pointLight position={[5, 5, -5]} intensity={1} />
          <axesHelper args={[0.3]} />
          
          <Suspense fallback={null}>
            <PointCloudViewer 
              pointsData={pointsData} 
              colorsData={colorsData} 
              posesData={posesData} 
              isIncremental={isIncrementalUpdate}
            />
          </Suspense>
          
          <OrbitControls enableZoom={true} makeDefault />
        </Canvas>
      ) : (
        <div
          style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'rgba(255, 255, 255, 0.7)',
            fontSize: '14px',
            fontWeight: '500',
          }}
        >
          3D Model Preview
        </div>
      )}
    </div>
  );
}
