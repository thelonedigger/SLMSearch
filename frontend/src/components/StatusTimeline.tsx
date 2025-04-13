// In frontend/src/components/StatusTimeline.tsx
// Update the StatusTimeline component to better handle and display updates:

import React, { useEffect, useState } from 'react';
import { StatusUpdate } from '../api/StatusService';

interface StatusTimelineProps {
  statusUpdates: StatusUpdate[];
}

const StatusTimeline: React.FC<StatusTimelineProps> = ({ statusUpdates }) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Auto-expand the latest in-progress item
  useEffect(() => {
    const inProgressItems = statusUpdates.filter(update => 
      update.status === 'in-progress' || update.status === 'pending'
    );
    
    if (inProgressItems.length > 0) {
      // Sort by timestamp (most recent first) and expand the most recent
      const latestItem = inProgressItems.sort((a, b) => b.timestamp - a.timestamp)[0];
      setExpandedId(latestItem.id);
    } else if (statusUpdates.length > 0) {
      // If no in-progress items, expand the most recent item of any type
      const latestItem = statusUpdates.sort((a, b) => b.timestamp - a.timestamp)[0];
      setExpandedId(latestItem.id);
    }
  }, [statusUpdates]);

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const getStatusLabel = (status: string): string => {
    switch (status) {
      case 'in-progress': return 'In Progress';
      case 'completed': return 'Completed';
      case 'error': return 'Error';
      case 'canceled': return 'Canceled';
      case 'pending': return 'Pending';
      case 'canceling': return 'Canceling';
      default: return status.charAt(0).toUpperCase() + status.slice(1);
    }
  };

  // Format relative time (e.g., "2 minutes ago")
  const getRelativeTime = (timestamp: number): string => {
    const now = Date.now();
    const diffMs = now - timestamp;
    
    const seconds = Math.floor(diffMs / 1000);
    if (seconds < 60) return `${seconds} seconds ago`;
    
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    
    const days = Math.floor(hours / 24);
    return `${days} day${days !== 1 ? 's' : ''} ago`;
  };

  return (
    <div className="status-timeline">
      <h2>Processing Status</h2>
      {statusUpdates.length === 0 ? (
        <div className="status-empty">No operations in progress</div>
      ) : (
        <ul className="timeline-list">
          {statusUpdates
            .sort((a, b) => b.timestamp - a.timestamp) // Sort by newest first
            .map((update) => (
              <li 
                key={update.id} 
                className={`timeline-item timeline-item-${update.status}`}
                onClick={() => toggleExpand(update.id)}
              >
                <div className="timeline-header">
                  <div className="timeline-header-left">
                    <span className={`status-indicator status-${update.status}`}></span>
                    <span className="timeline-title">{update.title}</span>
                    <span className="timeline-status">{getStatusLabel(update.status)}</span>
                  </div>
                  <span className="timeline-timestamp" title={new Date(update.timestamp).toLocaleString()}>
                    {getRelativeTime(update.timestamp)}
                  </span>
                </div>
                {(expandedId === update.id || update.status === 'in-progress') && (
                  <div className="timeline-details">
                    {update.details && <p>{update.details}</p>}
                    {update.progress !== undefined && update.progress >= 0 && (
                      <div className="timeline-progress">
                        <div className="progress-bar">
                          <div 
                            className="progress-fill"
                            style={{ width: `${update.progress}%` }}
                          ></div>
                        </div>
                        <span className="progress-text">{Math.round(update.progress)}%</span>
                      </div>
                    )}
                    {update.status === 'in-progress' && update.operation === 'build_index' && (
                      <button 
                        className="btn btn-small btn-warning" 
                        onClick={(e) => {
                          e.stopPropagation();
                          // Import and use statusService here
                          const statusService = require('../api/statusService').default;
                          statusService.cancelOperation(update.id);
                        }}
                      >
                        Cancel
                      </button>
                    )}
                  </div>
                )}
              </li>
            ))}
        </ul>
      )}
    </div>
  );
};

export default StatusTimeline;