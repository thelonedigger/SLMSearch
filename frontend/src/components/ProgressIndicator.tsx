import React from 'react';

interface ProgressIndicatorProps {
  progress: number;
  label?: string;
  showPercentage?: boolean;
  className?: string;
  size?: 'small' | 'medium' | 'large';
  isIndeterminate?: boolean;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  progress,
  label,
  showPercentage = true,
  className = '',
  size = 'medium',
  isIndeterminate = false
}) => {
  // Normalize progress value
  const normalizedProgress = Math.min(Math.max(progress, 0), 100);
  
  return (
    <div className={`progress-container progress-${size} ${className}`}>
      {label && <div className="progress-label">{label}</div>}
      <div className={`progress-bar ${isIndeterminate ? 'indeterminate' : ''}`}>
        <div 
          className="progress-fill"
          style={{ width: isIndeterminate ? '100%' : `${normalizedProgress}%` }}
        ></div>
      </div>
      {showPercentage && !isIndeterminate && (
        <div className="progress-percentage">{Math.round(normalizedProgress)}%</div>
      )}
    </div>
  );
};

export default ProgressIndicator;