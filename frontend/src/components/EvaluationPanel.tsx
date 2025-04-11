import React, { useState } from 'react';
import { 
  runEvaluation, 
  runGPTEvaluation, 
  runGPTRankingEvaluation,
  Metrics,
  EvaluationRequest,
  GPTEvaluationRequest
} from '../api/api';

interface EvaluationPanelProps {
  availableSplits: string[];
}

const EvaluationPanel: React.FC<EvaluationPanelProps> = ({ availableSplits }) => {
  // Standard evaluation state
  const [evalRequest, setEvalRequest] = useState<EvaluationRequest>({
    split: availableSplits[0] || 'val',
    top_k: 100,
    num_samples: 50
  });
  
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  
  // GPT evaluation state
  const [gptRequest, setGptRequest] = useState<GPTEvaluationRequest>({
    num_samples: 10,
  });
  
  const [gptResponse, setGptResponse] = useState<string | null>(null);
  const [isGptEvaluating, setIsGptEvaluating] = useState<boolean>(false);
  const [gptError, setGptError] = useState<string | null>(null);
  
  // GPT ranking evaluation state
  const [rankingNumQueries, setRankingNumQueries] = useState<number>(10);
  const [rankingPassagesPerQuery, setRankingPassagesPerQuery] = useState<number>(5);
  const [rankingResponse, setRankingResponse] = useState<string | null>(null);
  const [isRankingEvaluating, setIsRankingEvaluating] = useState<boolean>(false);
  const [rankingError, setRankingError] = useState<string | null>(null);
  
  // Handle standard evaluation
  const handleEvaluate = async () => {
    setIsEvaluating(true);
    setEvalError(null);
    setMetrics(null);
    
    try {
      const response = await runEvaluation(evalRequest);
      setMetrics(response.metrics);
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : 'An error occurred during evaluation');
    } finally {
      setIsEvaluating(false);
    }
  };
  
  // Handle GPT evaluation
  const handleGptEvaluate = async () => {
    setIsGptEvaluating(true);
    setGptError(null);
    setGptResponse(null);
    
    try {
      const response = await runGPTEvaluation(gptRequest);
      setGptResponse(response.message);
    } catch (err) {
      setGptError(err instanceof Error ? err.message : 'An error occurred during GPT evaluation');
    } finally {
      setIsGptEvaluating(false);
    }
  };
  
  // Handle GPT ranking evaluation
  const handleRankingEvaluate = async () => {
    setIsRankingEvaluating(true);
    setRankingError(null);
    setRankingResponse(null);
    
    try {
      const response = await runGPTRankingEvaluation(
        rankingNumQueries,
        rankingPassagesPerQuery
      );
      setRankingResponse(response.message);
    } catch (err) {
      setRankingError(err instanceof Error ? err.message : 'An error occurred during GPT ranking evaluation');
    } finally {
      setIsRankingEvaluating(false);
    }
  };
  
  return (
    <div className="card">
      <div className="card-header">
        <h1>Evaluation</h1>
      </div>
      <div className="card-content">
        <div className="section-divider">
          <h2>Standard Evaluation</h2>
        </div>
        
        <div className="form-group">
          <label htmlFor="eval-split" className="form-label">Dataset Split</label>
          <select
            id="eval-split"
            className="form-select"
            value={evalRequest.split}
            onChange={(e) => setEvalRequest({ ...evalRequest, split: e.target.value })}
          >
            {availableSplits.map(split => (
              <option key={split} value={split}>{split}</option>
            ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="eval-top-k" className="form-label">Top K Results</label>
          <input
            id="eval-top-k"
            type="number"
            className="form-input"
            min="1"
            max="1000"
            value={evalRequest.top_k}
            onChange={(e) => setEvalRequest({ ...evalRequest, top_k: parseInt(e.target.value) || 100 })}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="eval-samples" className="form-label">Number of Samples (0 for all)</label>
          <input
            id="eval-samples"
            type="number"
            className="form-input"
            min="0"
            value={evalRequest.num_samples || 0}
            onChange={(e) => {
              const value = parseInt(e.target.value);
              setEvalRequest({ 
                ...evalRequest, 
                num_samples: isNaN(value) ? 0 : value <= 0 ? undefined : value 
              });
            }}
          />
        </div>
        
        <button
          className="btn btn-primary"
          onClick={handleEvaluate}
          disabled={isEvaluating}
        >
          {isEvaluating ? 'Evaluating...' : 'Run Evaluation'}
        </button>
        
        {evalError && (
          <div className="alert alert-error">
            {evalError}
          </div>
        )}
        
        {isEvaluating && (
          <div className="loading"></div>
        )}
        
        {metrics && (
          <div className="metrics-container">
            <h3>Evaluation Results</h3>
            <div className="metrics-grid">
              {Object.entries(metrics).map(([key, value]) => (
                <div key={key} className="metric-item">
                  <div className="metric-name">{key}</div>
                  <div className="metric-value">{typeof value === 'number' ? value.toFixed(4) : value}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        <div className="section-divider">
          <h2>GPT-4o Evaluation</h2>
        </div>
        
        <div className="form-group">
          <label htmlFor="gpt-samples" className="form-label">Number of Samples</label>
          <input
            id="gpt-samples"
            type="number"
            className="form-input"
            min="1"
            max="100"
            value={gptRequest.num_samples}
            onChange={(e) => setGptRequest({ ...gptRequest, num_samples: parseInt(e.target.value) || 10 })}
          />
        </div>
        
        <button
          className="btn btn-primary"
          onClick={handleGptEvaluate}
          disabled={isGptEvaluating}
        >
          {isGptEvaluating ? 'Evaluating with GPT-4o...' : 'Run GPT-4o Evaluation'}
        </button>
        
        {gptError && (
          <div className="alert alert-error">
            {gptError}
          </div>
        )}
        
        {isGptEvaluating && (
          <div className="loading"></div>
        )}
        
        {gptResponse && (
          <div className="alert alert-success">
            {gptResponse}
          </div>
        )}
        
        <div className="section-divider">
          <h2>GPT-4o Ranking Evaluation</h2>
        </div>
        
        <div className="form-group">
          <label htmlFor="ranking-queries" className="form-label">Number of Queries</label>
          <input
            id="ranking-queries"
            type="number"
            className="form-input"
            min="1"
            max="100"
            value={rankingNumQueries}
            onChange={(e) => setRankingNumQueries(parseInt(e.target.value) || 10)}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="passages-per-query" className="form-label">Passages Per Query</label>
          <input
            id="passages-per-query"
            type="number"
            className="form-input"
            min="2"
            max="20"
            value={rankingPassagesPerQuery}
            onChange={(e) => setRankingPassagesPerQuery(parseInt(e.target.value) || 5)}
          />
        </div>
        
        <button
          className="btn btn-primary"
          onClick={handleRankingEvaluate}
          disabled={isRankingEvaluating}
        >
          {isRankingEvaluating ? 'Evaluating Rankings...' : 'Run Ranking Evaluation'}
        </button>
        
        {rankingError && (
          <div className="alert alert-error">
            {rankingError}
          </div>
        )}
        
        {isRankingEvaluating && (
          <div className="loading"></div>
        )}
        
        {rankingResponse && (
          <div className="alert alert-success">
            {rankingResponse}
          </div>
        )}
      </div>
    </div>
  );
};

export default EvaluationPanel;