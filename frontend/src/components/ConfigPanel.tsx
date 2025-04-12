import React, { useState, useEffect } from 'react';
import { getDatasetInfo, DatasetInfo, cancelOperation } from '../api/api';
import ProgressIndicator from './ProgressIndicator';
import statusService from '../api/StatusService';

interface ConfigPanelProps {
  onConfigChange?: (config: ConfigOptions) => void;
}

export interface ConfigOptions {
  modelName: string;
  useGPU: boolean;
  topK: number;
  dataDir: string;
}

const ConfigPanel: React.FC<ConfigPanelProps> = ({ onConfigChange }) => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  const [config, setConfig] = useState<ConfigOptions>({
    modelName: 'all-MiniLM-L6-v2',
    useGPU: false,
    topK: 10,
    dataDir: './processed_data'
  });
  
  // New state for operations in progress
  const [indexOperationId, setIndexOperationId] = useState<string | null>(null);
  const [indexProgress, setIndexProgress] = useState<number>(0);
  const [isIndexing, setIsIndexing] = useState<boolean>(false);
  
  // Fetch dataset info on component mount
  useEffect(() => {
    const fetchDatasetInfo = async () => {
      try {
        const info = await getDatasetInfo();
        setDatasetInfo(info);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load dataset information');
        setDatasetInfo(null);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchDatasetInfo();
  }, []);
  
  // Monitor status updates related to indexing
  useEffect(() => {
    interface StatusUpdate {
      id: string;
      operation: string;
      status: string;
      progress?: number;
    }
    
    const handleStatusUpdate = (updates: StatusUpdate[]) => {
      // Find indexing operations
      const indexOperation = updates.find(
        update => update.operation === 'build_index' && 
        (update.status === 'in-progress' || update.status === 'pending')
      );
      
      if (indexOperation) {
        setIndexOperationId(indexOperation.id);
        setIndexProgress(indexOperation.progress || 0);
        setIsIndexing(true);
      } else {
        setIsIndexing(false);
      }
    };
    
    // Listen for updates
    const updatesListener = statusService.options.onStatusUpdate;
    statusService.options.onStatusUpdate = (update) => {
      updatesListener?.(update);
      handleStatusUpdate(statusService.getStatusUpdates());
    };
    
    // Initial check
    handleStatusUpdate(statusService.getStatusUpdates());
    
    return () => {
      // Restore original listener
      statusService.options.onStatusUpdate = updatesListener;
    };
  }, []);
  
  // Handle config changes and propagate to parent component
  const handleConfigChange = (key: keyof ConfigOptions, value: string | boolean | number) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    
    if (onConfigChange) {
      onConfigChange(newConfig);
    }
  };
  
  // Handle cancellation of the indexing operation
  const handleCancelIndexing = async () => {
    if (indexOperationId) {
      try {
        await cancelOperation(indexOperationId);
        setIsIndexing(false);
      } catch (error) {
        console.error("Failed to cancel indexing operation:", error);
      }
    }
  };
  
  return (
    <div className="card">
      <div className="card-header">
        <h1>Configuration</h1>
      </div>
      <div className="card-content">
        {isLoading ? (
          <div className="loading"></div>
        ) : error ? (
          <div className="alert alert-error">{error}</div>
        ) : (
          <>
            {/* Show progress indicator for indexing */}
            {isIndexing && (
              <div className="indexing-status">
                <h3>Building Search Index</h3>
                <ProgressIndicator 
                  progress={indexProgress} 
                  label="Indexing in progress" 
                  isIndeterminate={indexProgress === 0}
                />
                <button 
                  className="btn btn-secondary btn-sm" 
                  onClick={handleCancelIndexing}
                >
                  Cancel Indexing
                </button>
              </div>
            )}
            
            <div className="stats-highlight">
              <h2>Dataset Information</h2>
              <div className="stats-container">
                <div className="stats-label">Collection Size</div>
                <div className="stats-value">{datasetInfo?.collection_size.toLocaleString()}</div>
              </div>
              <div className="stats-container">
                <div className="stats-label">Training Queries</div>
                <div className="stats-value">{datasetInfo?.train_queries.toLocaleString()}</div>
              </div>
              <div className="stats-container">
                <div className="stats-label">Validation Queries</div>
                <div className="stats-value">{datasetInfo?.val_queries.toLocaleString()}</div>
              </div>
              {datasetInfo?.has_test_split && (
                <div className="stats-container">
                  <div className="stats-label">Test Queries</div>
                  <div className="stats-value">{datasetInfo.test_queries.toLocaleString()}</div>
                </div>
              )}
              <div className="stats-container">
                <div className="stats-label">Available Splits</div>
                <div className="stats-value">
                  {datasetInfo?.available_splits.join(', ')}
                </div>
              </div>
            </div>
            
            <h2 className="section-divider">Model Configuration</h2>
            
            <div className="form-group">
              <label htmlFor="model-name" className="form-label">Embedding Model</label>
              <select
                id="model-name"
                className="form-select"
                value={config.modelName}
                onChange={(e) => handleConfigChange('modelName', e.target.value)}
              >
                <option value="all-MiniLM-L6-v2">MiniLM-L6-v2</option>
                <option value="all-mpnet-base-v2">MPNet Base v2</option>
                <option value="all-MiniLM-L12-v2">MiniLM-L12-v2</option>
              </select>
            </div>
            
            <div className="form-group">
              <label className="form-label">
                <input
                  type="checkbox"
                  className="form-checkbox"
                  checked={config.useGPU}
                  onChange={(e) => handleConfigChange('useGPU', e.target.checked)}
                />
                Use GPU (if available)
              </label>
            </div>
            
            <div className="form-group">
              <label htmlFor="default-top-k" className="form-label">Default Top K Results</label>
              <input
                id="default-top-k"
                type="number"
                className="form-input"
                min="1"
                max="1000"
                value={config.topK}
                onChange={(e) => handleConfigChange('topK', parseInt(e.target.value) || 10)}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="data-dir" className="form-label">Data Directory</label>
              <input
                id="data-dir"
                type="text"
                className="form-input"
                value={config.dataDir}
                onChange={(e) => handleConfigChange('dataDir', e.target.value)}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ConfigPanel;