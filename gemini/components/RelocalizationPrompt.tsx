'use client';

import React from 'react';

interface RelocalizationPromptProps {
  message: string | null;
}

const RelocalizationPrompt: React.FC<RelocalizationPromptProps> = ({ message }) => {
  if (!message) {
    return null;
  }

  return (
    <div style={{
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      color: 'white',
      padding: '20px',
      borderRadius: '10px',
      textAlign: 'center',
      zIndex: 1000,
    }}>
      <h3>Relocalization Required</h3>
      <p>{message}</p>
    </div>
  );
};

export default RelocalizationPrompt;
