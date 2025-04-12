// Status update handling and WebSocket communication

export type StatusType = 'pending' | 'in-progress' | 'completed' | 'error' | 'canceled';

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
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000;
  private isConnected = false;
  private statusUpdates: StatusUpdate[] = [];
  public options: StatusServiceOptions = {};

  constructor(options: StatusServiceOptions = {}) {
    this.options = options;
  }

  public connect(url: string = `ws://${window.location.host}/ws/status`): void {
    if (this.socket) {
      this.disconnect();
    }

    try {
      this.socket = new WebSocket(url);

      this.socket.onopen = () => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.options.onConnectionChange?.(true);
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'status_update') {
            const update = data.data as StatusUpdate;
            this.handleStatusUpdate(update);
          }
        } catch (error) {
          console.error('Error processing WebSocket message:', error);
          this.options.onError?.(error);
        }
      };

      this.socket.onclose = () => {
        console.log('WebSocket disconnected');
        this.isConnected = false;
        this.options.onConnectionChange?.(false);
        this.attemptReconnect(url);
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.options.onError?.(error);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      this.options.onError?.(error);
      this.attemptReconnect(url);
    }
  }

  private attemptReconnect(url: string): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.connect(url);
      }, this.reconnectDelay);
    } else {
      console.error('Max reconnect attempts reached. WebSocket connection failed.');
    }
  }

  public disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }

  private handleStatusUpdate(update: StatusUpdate): void {
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

    try {
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
    return this.isConnected;
  }
}

// Singleton instance
export const statusService = new StatusService();
export default statusService;