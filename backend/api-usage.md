# FastAPI Integration for Semantic Document Retrieval

This document describes how to use the FastAPI integration for the Semantic Document Retrieval system.

## Setup

1. Install the dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have processed the MS MARCO dataset using the `DataPreprocessing/splitting.py` script.

3. If you want to use GPT-4o evaluation, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the API

Start the FastAPI server with:
```bash
uvicorn main:app --reload
```

This will start the server at http://localhost:8000

## API Endpoints

### 1. Search

**Endpoint**: `POST /search`

**Request**:
```json
{
  "query": "your search query",
  "top_k": 10
}
```

**Response**:
```json
{
  "results": [
    {
      "passage_id": "passage-123",
      "score": 0.95,
      "passage_text": "This is the retrieved passage content..."
    },
    ...
  ],
  "execution_time": 0.125
}
```

### 2. Evaluate Pipeline

**Endpoint**: `POST /evaluate`

**Request**:
```json
{
  "split": "val",
  "top_k": 100,
  "num_samples": 50
}
```

**Response**:
```json
{
  "metrics": {
    "mrr": 0.35,
    "ndcg": 0.42,
    "precision_at_1": 0.25,
    ...
  }
}
```

### 3. GPT-4o Evaluation

**Endpoint**: `POST /gpt-evaluate`

**Request**:
```json
{
  "num_samples": 10,
  "data_dir": "./processed_data"
}
```

**Response**:
```json
{
  "success": true,
  "message": "GPT-4o evaluation started with 10 samples. Results will be saved to gpt4o_validation_results.csv",
  "results_path": "gpt4o_validation_results.csv"
}
```

### 4. GPT-4o Ranking Evaluation

**Endpoint**: `POST /gpt-ranking-evaluate`

**Query Parameters**:
- `num_queries`: Number of queries to evaluate (default: 10)
- `passages_per_query`: Number of passages per query to rank (default: 5)
- `data_dir`: Data directory (default: "./processed_data")

**Response**:
```json
{
  "success": true,
  "message": "GPT-4o ranking evaluation started with 10 queries and 5 passages per query. Results will be saved to gpt4o_ranking_validation.csv",
  "results_path": "gpt4o_ranking_validation.csv"
}
```

## API Documentation

FastAPI automatically generates interactive API documentation. You can access it at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc