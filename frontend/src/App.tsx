import { useState, useEffect } from 'react';
import SearchInterface from './components/SearchInterface';
import ConfigPanel from './components/ConfigPanel';
import EvaluationPanel from './components/EvaluationPanel';
import StatusTimeline from './components/StatusTimeline';
import { getDatasetInfo, DatasetInfo } from './api/api';
import { ConfigOptions } from './components/ConfigPanel';
import statusService, { StatusUpdate } from './api/StatusService';

function App() {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<ConfigOptions>({
    modelName: 'all-MiniLM-L6-v2',
    useGPU: false,
    topK: 10,
    dataDir: './processed_data'
  });
  // New state for status updates
  const [statusUpdates, setStatusUpdates] = useState<StatusUpdate[]>([]);
  const [isSocketConnected, setIsSocketConnected] = useState<boolean>(false);

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

  // Connect to WebSocket for status updates
  useEffect(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.hostname;
    const wsPort = process.env.NODE_ENV === 'development' ? '8000' : window.location.port;
    const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws/status`;
    
    statusService.connect(wsUrl);
    
    // Configure status service to update our state
    statusService.options = {
      onStatusUpdate: (update) => {
        setStatusUpdates(statusService.getStatusUpdates());
      },
      onConnectionChange: (connected) => {
        setIsSocketConnected(connected);
      },
      onError: (error) => {
        console.error('Status service error:', error);
      }
    };
    
    // Cleanup on unmount
    return () => {
      statusService.disconnect();
    };
  }, []);

  // Handle config changes
  const handleConfigChange = (newConfig: ConfigOptions) => {
    setConfig(newConfig);
  };

  return (
    <div className="container">
      <div className="app-header">
        <h1>Semantic Document Retrieval System</h1>
        <p>MiniLM-based document retrieval with reinforcement learning optimization</p>
        {isSocketConnected && <span className="connection-status connected">Backend Connected</span>}
        {!isSocketConnected && <span className="connection-status disconnected">Backend Disconnected</span>}
      </div>
      
      {isLoading ? (
        <div className="loading"></div>
      ) : error ? (
        <div className="alert alert-error">
          {error}
          <p>Make sure the backend server is running at http://localhost:8000</p>
        </div>
      ) : (
        <div className="app-layout">
          <div className="left-column">
            <ConfigPanel onConfigChange={handleConfigChange} />
            
            {/* Display status timeline if there are updates */}
            {statusUpdates.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h1>Processing Status</h1>
                </div>
                <div className="card-content">
                  <StatusTimeline statusUpdates={statusUpdates} />
                </div>
              </div>
            )}
            
            <EvaluationPanel 
              availableSplits={datasetInfo?.available_splits || ['val']} 
            />
          </div>
          <div className="right-column">
            <SearchInterface defaultTopK={config.topK} />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;