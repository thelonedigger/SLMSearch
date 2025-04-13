// Status update handling and WebSocket communication with fallback mechanism

export type StatusType = 'pending' | 'in-progress' | 'completed' | 'error' | 'canceled' | 'canceling';

export interface StatusUpdate {
  id: string;
  title: string;
  status: StatusType;
  details?: string;
  timestamp: number;
  progress?: number;
  operation: string;
  estimatedTimeRemaining?: number;
}

export interface StatusServiceOptions {
  onStatusUpdate?: (update: StatusUpdate) => void;
  onConnectionChange?: (connected: boolean) => void;
  onError?: (error: any) => void;
}

class StatusService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  // Increased maxReconnectAttempts from 5 to 10 as per update 1
  private maxReconnectAttempts = 10;
  private reconnectDelay = 2000;
  private isConnected = false;
  // Original comment: "private statusUpdates: StatusUpdate[] = [];" is preserved below.
  private statusUpdates: StatusUpdate[] = [];
  public options: StatusServiceOptions = {};

  // Added fallback polling mechanism from update 1
  private usePollingFallback = false;
  private pollingInterval: number | null = null;
  private pollIntervalMs = 5000; // Poll every 5 seconds

  // Property for periodic connection check (added in update 1)
  private connectionCheckInterval: number | null = null;

  constructor(options: StatusServiceOptions = {}) {
    this.options = options;
  }

  // Updated getWebSocketUrl from update 1:
  private getWebSocketUrl(): string {
    // Check if a specific backend URL is configured
    const configuredBackendUrl = process.env.REACT_APP_BACKEND_URL;
    if (configuredBackendUrl) {
      // Replace http/https with ws/wss
      return configuredBackendUrl.replace(/^http/, 'ws') + '/ws/status';
    }
    
    // For development, try to detect the backend port
    if (process.env.NODE_ENV === 'development') {
      // If frontend is running on port 3000, assume backend is on 8000
      if (window.location.port === '3000') {
        return `ws://${window.location.hostname}:8000/ws/status`;
      }
      return `ws://${window.location.hostname}:${window.location.port}/ws/status`;
    }
    
    // For production
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsPort = window.location.port || '80';
    return `${wsProtocol}//${window.location.hostname}:${wsPort}/ws/status`;
  }

  public connect(url?: string): void {
    // Don't try to connect if we're using polling fallback
    if (this.usePollingFallback) {
      console.log('Using polling fallback instead of WebSocket');
      return;
    }
    
    const wsUrl = url || this.getWebSocketUrl();
    
    // Modification: check if already connected to avoid duplicate connections (update 1)
    if (this.socket && this.isConnected && this.socket.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    // If we have an existing socket in a non-open state, clean it up
    if (this.socket) {
      this.disconnect();
    }

    console.log(`Connecting to WebSocket at ${wsUrl}`);
    
    try {
      this.socket = new WebSocket(wsUrl);
      
      // Add a connection timeout (update 1)
      const connectionTimeout = setTimeout(() => {
        if (this.socket && this.socket.readyState !== WebSocket.OPEN) {
          console.error('WebSocket connection timed out');
          this.socket.close();
        }
      }, 5000); // 5 second timeout
      
      this.socket.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.options.onConnectionChange?.(true);
        
        // Start periodic connection check (added in update 1)
        this.startConnectionCheck();
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Received WebSocket message:', data);
          
          if (data.type === 'status_update') {
            const update = data.data as StatusUpdate;
            this.handleStatusUpdate(update);
          }
        } catch (error) {
          console.error('Error processing WebSocket message:', error);
          this.options.onError?.(error);
        }
      };

      this.socket.onclose = (event) => {
        clearTimeout(connectionTimeout);
        console.log(`WebSocket disconnected with code ${event.code}, reason: ${event.reason}`);
        this.isConnected = false;
        this.options.onConnectionChange?.(false);
        this.stopConnectionCheck();
        
        // If we've had too many failed attempts, switch to polling fallback (update 1)
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          console.warn('Max reconnect attempts reached. Switching to polling fallback.');
          this.setupPolling();
          return;
        }
        
        this.attemptReconnect(wsUrl);
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.options.onError?.(error);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      this.options.onError?.(error);
      this.isConnected = false;
      this.options.onConnectionChange?.(false);
      this.attemptReconnect(wsUrl);
    }
  }

  // Set up polling as a fallback (added in update 1)
  private setupPolling(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
    }
    
    console.log(`Setting up polling fallback mechanism (${this.pollIntervalMs}ms interval)`);
    
    this.usePollingFallback = true;
    this.pollingInterval = window.setInterval(() => {
      this.pollOperationStatus();
    }, this.pollIntervalMs);
    
    // Immediately poll once
    this.pollOperationStatus();
  }
  
  // Poll for status updates (added in update 1)
  private async pollOperationStatus(): Promise<void> {
    try {
      // Call the REST API endpoint to get operation status
      const response = await fetch('/operations/status');
      
      if (response.ok) {
        const data = await response.json();
        
        // Process each operation status
        if (data && data.operations) {
          for (const operation of data.operations) {
            // Create a compatible status update
            const statusUpdate: StatusUpdate = {
              id: operation.id,
              title: operation.title,
              status: operation.status,
              details: operation.details,
              timestamp: operation.timestamp,
              progress: operation.progress,
              operation: operation.operation,
              estimatedTimeRemaining: operation.estimated_time_remaining
            };
            
            // Handle the status update
            this.handleStatusUpdate(statusUpdate);
          }
        }
      } else {
        console.error('Failed to poll operation status:', response.statusText);
      }
    } catch (error) {
      console.error('Error polling operation status:', error);
    }
  }

  // Begin periodic connection check methods (added in update 1)
  private startConnectionCheck(): void {
    // Clear any existing interval
    this.stopConnectionCheck();
    
    // Check connection every 10 seconds
    const CHECK_INTERVAL_MS = 10000;
    this.connectionCheckInterval = window.setInterval(() => {
      if (this.socket?.readyState !== WebSocket.OPEN) {
        console.log('Connection check failed, socket not open');
        this.isConnected = false;
        this.options.onConnectionChange?.(false);
        this.reconnect();
      }
    }, CHECK_INTERVAL_MS);
  }

  private stopConnectionCheck(): void {
    if (this.connectionCheckInterval !== null) {
      clearInterval(this.connectionCheckInterval);
      this.connectionCheckInterval = null;
    }
  }
  
  // Added in update 1: reconnect that calls disconnect and then connect.
  private reconnect(): void {
    this.disconnect();
    this.connect();
  }

  // Modified in update 1: attemptReconnect now uses exponential backoff and switches to polling fallback after max attempts.
  private attemptReconnect(url: string): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      // Use exponential backoff for reconnect delays
      const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);
      const cappedDelay = Math.min(delay, 30000); // Cap at 30 seconds
      
      setTimeout(() => {
        this.connect(url);
      }, cappedDelay);
    } else {
      console.error('Max reconnect attempts reached. Switching to polling fallback.');
      this.setupPolling();
      return;
    }
  }

  public disconnect(): void {
    // Stop connection check interval
    this.stopConnectionCheck();
    
    // Stop polling if active (update 1)
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    
    if (this.socket) {
      // Only close if not already closed (added in update 1)
      if (this.socket.readyState === WebSocket.OPEN ||
          this.socket.readyState === WebSocket.CONNECTING) {
        try {
          this.socket.close();
        } catch (e) {
          console.error('Error closing WebSocket:', e);
        }
      }
      this.socket = null;
      this.isConnected = false;
      this.options.onConnectionChange?.(false);
    }
  }

  private handleStatusUpdate(update: StatusUpdate): void {
    console.log('Processing status update:', update);
    
    // Store the status update
    const existingIndex = this.statusUpdates.findIndex(u => u.id === update.id);
    
    if (existingIndex >= 0) {
      // Update existing status
      this.statusUpdates[existingIndex] = update;
    } else {
      // Add new status
      this.statusUpdates.push(update);
    }
    
    // Clean up completed/error statuses after keeping them for a while
    this.cleanupOldStatuses();
    
    // Notify listeners
    this.options.onStatusUpdate?.(update);
  }

  private cleanupOldStatuses(): void {
    const now = Date.now();
    const ONE_HOUR = 60 * 60 * 1000;
    
    // Keep completed statuses for an hour, then remove them
    this.statusUpdates = this.statusUpdates.filter(update => {
      if (update.status === 'completed' || update.status === 'error' || update.status === 'canceled') {
        return now - update.timestamp < ONE_HOUR;
      }
      return true;
    });
  }

  public getStatusUpdates(): StatusUpdate[] {
    return [...this.statusUpdates].sort((a, b) => b.timestamp - a.timestamp);
  }

  public cancelOperation(operationId: string): void {
    // Added in update 1: When using polling fallback, use fetch API to cancel operation.
    if (this.usePollingFallback) {
      fetch(`/operations/cancel/${operationId}`, {
        method: 'POST',
      }).catch(error => {
        console.error('Error sending cancel request:', error);
        this.options.onError?.(error);
      });
      return;
    }

    if (!this.isConnected || !this.socket) {
      console.error('Cannot cancel operation: WebSocket not connected');
      return;
    }

    // Check that the socket is open before sending (added in update 1)
    if (this.socket.readyState !== WebSocket.OPEN) {
      console.error('Cannot cancel operation: WebSocket not open');
      return;
    }

    try {
      console.log(`Sending cancel request for operation ${operationId}`);
      this.socket.send(JSON.stringify({
        type: 'cancel_operation',
        operation_id: operationId
      }));
    } catch (error) {
      console.error('Error sending cancel request:', error);
      this.options.onError?.(error);
    }
  }

  public checkStatus(): boolean {
    // In update 1, check that the socket is connected and open or fallback is enabled
    return (this.isConnected && this.socket?.readyState === WebSocket.OPEN) || this.usePollingFallback;
  }
}

// Singleton instance
export const statusService = new StatusService();
export default statusService;
