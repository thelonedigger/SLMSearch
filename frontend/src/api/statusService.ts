// Status update handling and WebSocket communication

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

  constructor(options: StatusServiceOptions = {}) {
    this.options = options;
  }

  // Added in update 1: getWebSocketUrl method to generate the correct WebSocket URL.
  private getWebSocketUrl(): string {
    // For development, explicitly use the backend URL
    if (process.env.NODE_ENV === 'development') {
      return 'ws://localhost:8000/ws/status';
    }
    
    // For production, derive from window location
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.hostname;
    const wsPort = '8000'; // Always use the backend port
    return `${wsProtocol}//${wsHost}:${wsPort}/ws/status`;
  }

  public connect(url?: string): void {
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
      
      this.socket.onopen = () => {
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
        console.log(`WebSocket disconnected with code ${event.code}, reason: ${event.reason}`);
        this.isConnected = false;
        this.options.onConnectionChange?.(false);
        this.stopConnectionCheck();
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

  // Added in update 1: Begin periodic connection check methods
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
  // Added property for connection check interval
  private connectionCheckInterval: number | null = null;

  // Added in update 1: reconnect that calls disconnect and then connect.
  private reconnect(): void {
    this.disconnect();
    this.connect();
  }

  // Modified in update 1: attemptReconnect now uses exponential backoff and logs updated messages.
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
      console.error('Max reconnect attempts reached. WebSocket connection failed.');
      
      // After max attempts, try one final reconnect after a longer delay
      setTimeout(() => {
        this.reconnectAttempts = 0; // Reset counter
        this.connect(url);
      }, 60000); // Try again after a minute
    }
  }
  // End of update 1 modifications

  public disconnect(): void {
    // Stop connection check interval
    this.stopConnectionCheck();
    
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
    // In update 1, check that the socket is both connected and open
    return this.isConnected && this.socket?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
export const statusService = new StatusService();
export default statusService;
