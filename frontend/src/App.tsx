import { useState, useEffect } from 'react';
import SearchInterface from './components/SearchInterface';
import ConfigPanel from './components/ConfigPanel';
import EvaluationPanel from './components/EvaluationPanel';
import { getDatasetInfo, DatasetInfo } from './api/api';
import { ConfigOptions } from './components/ConfigPanel';

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

  // Handle config changes
  const handleConfigChange = (newConfig: ConfigOptions) => {
    setConfig(newConfig);
  };

  return (
    <div className="container">
      <div className="app-header">
        <h1>Semantic Document Retrieval System</h1>
        <p>MiniLM-based document retrieval with reinforcement learning optimization</p>
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