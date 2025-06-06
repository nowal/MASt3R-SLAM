'use client';

import React, { useEffect, useRef, useCallback } from 'react';

interface WebSocketConnectionProps {
  sessionId: string | null;
  backendUrl: string;
  onConnectionChange: (connected: boolean) => void;
  onMessage: (message: any) => void;
  onError: (error: string) => void;
  reconnectDelay?: number;
  onSendMessageReady?: (sendMessage: (message: any) => boolean) => void;
}

export default function WebSocketConnection({
  sessionId,
  backendUrl,
  onConnectionChange,
  onMessage,
  onError,
  reconnectDelay = 5000,
  onSendMessageReady
}: WebSocketConnectionProps) {
  const webSocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isIntentionalCloseRef = useRef<boolean>(false);

  const connectToBackend = useCallback(async () => {
    if (!sessionId) return;

    console.log(`WebSocketConnection: Attempting WebSocket connection for ${sessionId}`);
    
    // Clean up any existing connection
    if (webSocketRef.current) {
      isIntentionalCloseRef.current = true;
      webSocketRef.current.close();
      webSocketRef.current = null;
    }

    // Clear any pending reconnection
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    try {
      const ws = new WebSocket(`${backendUrl}/ws/${sessionId}`);
      webSocketRef.current = ws;
      isIntentionalCloseRef.current = false;

      ws.onopen = () => {
        console.log('WebSocketConnection: WebSocket Connected');
        onConnectionChange(true);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          onMessage(message);
        } catch (e) {
          console.error("WebSocketConnection: Failed to parse message:", event.data, e);
          onMessage(`Non-JSON: ${event.data}`);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocketConnection: WebSocket Error:', error);
        onError('WebSocket connection error. Check console and backend.');
        onConnectionChange(false);
      };

      ws.onclose = (event) => {
        console.log(`WebSocketConnection: WebSocket Closed (Code: ${event.code}, Clean: ${event.wasClean})`);
        onConnectionChange(false);
        
        if (webSocketRef.current === ws) {
          webSocketRef.current = null;
        }

        if (!event.wasClean && !isIntentionalCloseRef.current) {
          onError(`WebSocket closed unexpectedly. Retrying in ${reconnectDelay / 1000}s...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connectToBackend();
          }, reconnectDelay);
        } else if (isIntentionalCloseRef.current) {
          console.log('WebSocketConnection: Intentional close, not reconnecting');
        } else {
          onError('WebSocket connection closed.');
        }
      };

    } catch (error) {
      console.error('WebSocketConnection: Failed to create WebSocket:', error);
      const errorMsg = error instanceof Error ? error.message : String(error);
      onError(`WebSocket creation failed: ${errorMsg}`);
      onConnectionChange(false);
    }
  }, [sessionId, backendUrl, onConnectionChange, onMessage, onError, reconnectDelay]);

  // Send message through WebSocket (handles both JSON and binary)
  const sendMessage = useCallback((message: any) => {
    if (webSocketRef.current?.readyState === WebSocket.OPEN) {
      try {
        // Handle binary frame data specially
        if (message.type === 'BINARY_FRAME_DATA' && message.data) {
          // Convert array back to Uint8Array and send as binary
          const binaryData = new Uint8Array(message.data);
          webSocketRef.current.send(binaryData);
          return true;
        } else {
          // Send as JSON for all other message types
          webSocketRef.current.send(JSON.stringify(message));
          return true;
        }
      } catch (error) {
        console.error('WebSocketConnection: Failed to send message:', error);
        onError('Failed to send message through WebSocket');
        return false;
      }
    } else {
      console.warn('WebSocketConnection: Cannot send message, WebSocket not open');
      return false;
    }
  }, [onError]);

  // Expose sendMessage function to parent component
  useEffect(() => {
    if (onSendMessageReady) {
      onSendMessageReady(sendMessage);
    }
  }, [sendMessage, onSendMessageReady]);

  // Effect to establish connection when sessionId changes
  useEffect(() => {
    if (!sessionId) {
      // Clean up when no sessionId
      if (webSocketRef.current) {
        isIntentionalCloseRef.current = true;
        webSocketRef.current.close();
        webSocketRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      onConnectionChange(false);
      return;
    }

    connectToBackend();

    // Cleanup function
    return () => {
      console.log("WebSocketConnection: Cleaning up WebSocket connection");
      isIntentionalCloseRef.current = true;
      
      if (webSocketRef.current) {
        webSocketRef.current.onclose = null;
        webSocketRef.current.onerror = null;
        webSocketRef.current.onmessage = null;
        webSocketRef.current.onopen = null;
        
        if (webSocketRef.current.readyState === WebSocket.OPEN || 
            webSocketRef.current.readyState === WebSocket.CONNECTING) {
          webSocketRef.current.close();
        }
        webSocketRef.current = null;
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      onConnectionChange(false);
    };
  }, [sessionId, connectToBackend, onConnectionChange]);

  // This component doesn't render anything visible
  return null;
}
