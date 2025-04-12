import React, { useEffect, useState } from 'react';
import { StatusUpdate } from '../api/StatusService.ts';

interface StatusTimelineProps {
  statusUpdates: StatusUpdate[];
}

const StatusTimeline: React.FC<StatusTimelineProps> = ({ statusUpdates }) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Auto-expand the latest in-progress item
  useEffect(() => {
    const inProgressItems = statusUpdates.filter(update => update.status === 'in-progress');
    if (inProgressItems.length > 0) {
      setExpandedId(inProgressItems[inProgressItems.length - 1].id);
    }
  }, [statusUpdates]);

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="status-timeline">
      <h2>Processing Status</h2>
      {statusUpdates.length === 0 ? (
        <div className="status-empty">No operations in progress</div>
      ) : (
        <ul className="timeline-list">
          {statusUpdates.map((update) => (
            <li 
              key={update.id} 
              className={`timeline-item timeline-item-${update.status}`}
              onClick={() => toggleExpand(update.id)}
            >
              <div className="timeline-header">
                <span className={`status-indicator status-${update.status}`}></span>
                <span className="timeline-title">{update.title}</span>
                <span className="timeline-timestamp">{new Date(update.timestamp).toLocaleTimeString()}</span>
              </div>
              {expandedId === update.id && update.details && (
                <div className="timeline-details">
                  <p>{update.details}</p>
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