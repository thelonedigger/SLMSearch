import React, { useState } from 'react';
import { search, SearchResult, SearchResponse } from '../api/api';

interface SearchInterfaceProps {
  defaultTopK?: number;
}

const SearchInterface: React.FC<SearchInterfaceProps> = ({ defaultTopK = 10 }) => {
  const [query, setQuery] = useState<string>('');
  const [topK, setTopK] = useState<number>(defaultTopK);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response: SearchResponse = await search({
        query: query.trim(),
        top_k: topK
      });
      
      setResults(response.results);
      setExecutionTime(response.execution_time);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while searching');
      setResults([]);
      setExecutionTime(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="card">
      <div className="card-header">
        <h1>Semantic Document Search</h1>
      </div>
      <div className="card-content">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="search-query" className="form-label">Search Query</label>
            <input
              id="search-query"
              type="text"
              className="form-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your query"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="top-k" className="form-label">Number of Results</label>
            <select
              id="top-k"
              className="form-select"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
          
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isLoading}
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </form>
        
        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}
        
        {isLoading && (
          <div className="loading"></div>
        )}
        
        {!isLoading && results.length > 0 && (
          <div className="search-results-container">
            <div className="section-divider">
              <h2>Search Results</h2>
              <div className="stats-container">
                <div className="stats-label">Execution Time</div>
                <div className="stats-value">{executionTime?.toFixed(3)}s</div>
              </div>
            </div>
            
            <div className="search-results">
              {results.map((result, index) => (
                <div key={result.passage_id} className="search-result">
                  <div className="search-result-header">
                    <span>#{index + 1} - ID: {result.passage_id}</span>
                    <span className="search-result-score">Score: {result.score.toFixed(4)}</span>
                  </div>
                  <div className="search-result-content">
                    {result.passage_text}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {!isLoading && !error && results.length === 0 && query.trim() !== '' && (
          <div className="alert alert-info">
            No results found for your query.
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchInterface;