from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import json
import time
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Import from existing modules
from datapipeline.data_handler import load_preprocessed_data
from datapipeline.embedding_engine import create_embedding_engine
from datapipeline.retrieval_pipeline import RetrievalPipeline, run_evaluation
from gptevaluator.gpt4o_validation import validate_gpt4o_evaluation, validate_ranking_capability

# Create FastAPI app
app = FastAPI(
    title="Semantic Document Retrieval API",
    description="API for semantic document retrieval with reinforcement learning optimization",
    version="0.1.0"
)

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get default top_k from environment or use 10 as default
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", 10))

# Define request and response models
class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K

class SearchResult(BaseModel):
    passage_id: str
    score: float
    passage_text: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    execution_time: float

class EvaluationRequest(BaseModel):
    split: str = "val"
    top_k: int = 100
    num_samples: Optional[int] = None

class EvaluationResponse(BaseModel):
    metrics: Dict[str, float]

class GPTEvaluationRequest(BaseModel):
    num_samples: int = 10
    data_dir: str = os.environ.get("DATA_DIR", "./processed_data")

class GPTEvaluationResponse(BaseModel):
    success: bool
    message: str
    results_path: str

# Global variables for storing the pipeline
pipeline = None
dataset = None
embedding_engine = None

# Dependency to get the initialized pipeline
async def get_pipeline():
    global pipeline, dataset, embedding_engine
    
    if pipeline is None:
        try:
            # Get configuration from environment variables
            data_dir = os.environ.get("DATA_DIR", "./processed_data")
            model_name = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
            use_gpu = os.environ.get("USE_GPU", "False").lower() == "true"
            save_dir = os.environ.get("SAVE_DIR", "./saved_pipeline")
            
            # Check if data_dir exists
            if not os.path.exists(data_dir):
                raise HTTPException(status_code=500, detail=f"Data directory {data_dir} not found. Please run data preprocessing first.")
            
            # Load dataset
            dataset = load_preprocessed_data(data_dir)
            
            # Create embedding engine
            embedding_engine = create_embedding_engine(model_name=model_name, use_gpu=use_gpu)
            
            # Create retrieval pipeline
            pipeline = RetrievalPipeline(dataset, embedding_engine)
            
            # Build or load index
            if os.path.exists(os.path.join(save_dir, "pipeline_state.pkl")):
                pipeline.load(save_dir, use_gpu_index=use_gpu)
            else:
                # Build index if not found
                pipeline.build_index(batch_size=32)
                
                # Save the pipeline
                os.makedirs(save_dir, exist_ok=True)
                pipeline.save(save_dir)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize pipeline: {str(e)}")
    
    return pipeline

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Welcome to the Semantic Document Retrieval API",
        "documentation": "/docs",
        "redoc": "/redoc"
    }

@app.get("/dataset-info")
async def dataset_info(pipeline=Depends(get_pipeline)):
    """
    Get information about the dataset, including available splits
    """
    available_splits = pipeline.get_available_splits()
    has_test_split = 'test' in available_splits
    
    return {
        "available_splits": available_splits,
        "has_test_split": has_test_split,
        "is_custom_split": has_test_split,  # If test split exists, it's a custom split
        "collection_size": len(pipeline.dataset.collection),
        "train_queries": len(pipeline.dataset.train_queries),
        "val_queries": len(pipeline.dataset.val_queries),
        "test_queries": len(pipeline.dataset.test_queries) if has_test_split else 0,
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, pipeline=Depends(get_pipeline)):
    """
    Search for relevant passages given a query
    """
    try:
        # Record start time
        start_time = time.time()
        
        # Perform search
        results = pipeline.retrieve(request.query, top_k=request.top_k)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Format results
        formatted_results = [
            SearchResult(
                passage_id=pid,
                score=float(score),
                passage_text=text
            )
            for pid, score, text in results
        ]
        
        return SearchResponse(results=formatted_results, execution_time=execution_time)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest, pipeline=Depends(get_pipeline)):
    """
    Evaluate the retrieval pipeline on a dataset split
    """
    try:
        # Check if the requested split is available
        available_splits = pipeline.get_available_splits()
        if request.split not in available_splits:
            raise HTTPException(
                status_code=400, 
                detail=f"Split '{request.split}' not available. Available splits: {available_splits}"
            )
        
        # Run evaluation
        metrics = pipeline.evaluate(
            split=request.split,
            top_k=request.top_k,
            num_samples=request.num_samples
        )
        
        return EvaluationResponse(metrics=metrics)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/gpt-evaluate", response_model=GPTEvaluationResponse)
async def gpt_evaluate(request: GPTEvaluationRequest, background_tasks: BackgroundTasks):
    """
    Run GPT-4o evaluation for single passage evaluation
    """
    try:
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY environment variable not set.")
        
        # Run GPT-4o evaluation in background to not block the response
        background_tasks.add_task(
            validate_gpt4o_evaluation,
            data_dir=request.data_dir,
            num_samples=request.num_samples
        )
        
        return GPTEvaluationResponse(
            success=True,
            message=f"GPT-4o evaluation started with {request.num_samples} samples. Results will be saved to gpt4o_validation_results.csv",
            results_path="gpt4o_validation_results.csv"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-4o evaluation failed: {str(e)}")

@app.post("/gpt-ranking-evaluate", response_model=GPTEvaluationResponse)
async def gpt_ranking_evaluate(
    num_queries: int = 10,
    passages_per_query: int = 5,
    data_dir: str = os.environ.get("DATA_DIR", "./processed_data"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Run GPT-4o evaluation for passage ranking
    """
    try:
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY environment variable not set.")
        
        # Run GPT-4o ranking evaluation in background
        background_tasks.add_task(
            validate_ranking_capability,
            data_dir=data_dir,
            num_queries=num_queries,
            passages_per_query=passages_per_query
        )
        
        return GPTEvaluationResponse(
            success=True,
            message=f"GPT-4o ranking evaluation started with {num_queries} queries and {passages_per_query} passages per query. Results will be saved to gpt4o_ranking_validation.csv",
            results_path="gpt4o_ranking_validation.csv"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-4o ranking evaluation failed: {str(e)}")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)