// API interface for semantic document retrieval system

// Backend API base URL (update this based on your environment)
const API_BASE_URL = "http://localhost:8000";

// Types for API requests and responses
export interface SearchRequest {
  query: string;
  top_k: number;
}

export interface SearchResult {
  passage_id: string;
  score: number;
  passage_text: string;
}

export interface SearchResponse {
  results: SearchResult[];
  execution_time: number;
}

export interface EvaluationRequest {
  split: string;
  top_k: number;
  num_samples?: number;
}

export interface Metrics {
  [key: string]: number;
}

export interface EvaluationResponse {
  metrics: Metrics;
}

export interface DatasetInfo {
  available_splits: string[];
  has_test_split: boolean;
  is_custom_split: boolean;
  collection_size: number;
  train_queries: number;
  val_queries: number;
  test_queries: number;
}

export interface GPTEvaluationRequest {
  num_samples: number;
  data_dir?: string;
}

export interface GPTEvaluationResponse {
  success: boolean;
  message: string;
  results_path: string;
}

// New interfaces for status endpoints
export interface StatusResponse {
  id: string;
  status: string;
  message: string;
  timestamp: number;
  progress?: number;
}

export interface StatusListResponse {
  operations: StatusResponse[];
}

export interface CancelOperationResponse {
  success: boolean;
  message: string;
}

// API functions

// Get dataset information
export async function getDatasetInfo(): Promise<DatasetInfo> {
  const response = await fetch(`${API_BASE_URL}/dataset-info`);
  
  if (!response.ok) {
    throw new Error(`Failed to get dataset info: ${response.statusText}`);
  }
  
  return await response.json();
}

// Search for passages
export async function search(request: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }
  
  return await response.json();
}

// Run evaluation
export async function runEvaluation(request: EvaluationRequest): Promise<EvaluationResponse> {
  const response = await fetch(`${API_BASE_URL}/evaluate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    throw new Error(`Evaluation failed: ${response.statusText}`);
  }
  
  return await response.json();
}

// Run GPT-4o single passage evaluation
export async function runGPTEvaluation(request: GPTEvaluationRequest): Promise<GPTEvaluationResponse> {
  const response = await fetch(`${API_BASE_URL}/gpt-evaluate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    throw new Error(`GPT evaluation failed: ${response.statusText}`);
  }
  
  return await response.json();
}

// Run GPT-4o ranking evaluation
export async function runGPTRankingEvaluation(
  numQueries: number = 10,
  passagesPerQuery: number = 5,
  dataDir?: string
): Promise<GPTEvaluationResponse> {
  // Build URL with query parameters
  let url = `${API_BASE_URL}/gpt-ranking-evaluate?num_queries=${numQueries}&passages_per_query=${passagesPerQuery}`;
  
  if (dataDir) {
    url += `&data_dir=${encodeURIComponent(dataDir)}`;
  }
  
  const response = await fetch(url, {
    method: "POST",
  });
  
  if (!response.ok) {
    throw new Error(`GPT ranking evaluation failed: ${response.statusText}`);
  }
  
  return await response.json();
}

// New API functions for status and cancellation

// Get status of all operations
export async function getOperationStatuses(): Promise<StatusListResponse> {
  const response = await fetch(`${API_BASE_URL}/operations/status`);
  
  if (!response.ok) {
    throw new Error(`Failed to get operation statuses: ${response.statusText}`);
  }
  
  return await response.json();
}

// Get status of a specific operation
export async function getOperationStatus(operationId: string): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE_URL}/operations/status/${operationId}`);
  
  if (!response.ok) {
    throw new Error(`Failed to get operation status: ${response.statusText}`);
  }
  
  return await response.json();
}

// Cancel an operation
export async function cancelOperation(operationId: string): Promise<CancelOperationResponse> {
  const response = await fetch(`${API_BASE_URL}/operations/cancel/${operationId}`, {
    method: "POST",
  });
  
  if (!response.ok) {
    throw new Error(`Failed to cancel operation: ${response.statusText}`);
  }
  
  return await response.json();
}