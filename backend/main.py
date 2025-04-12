from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import json
import time
import traceback
import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backend_debug.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

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
logger.info("CORS middleware configured")

# Get default top_k from environment or use 10 as default
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", 10))
logger.info(f"Default TOP_K set to {DEFAULT_TOP_K}")

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
            
            logger.info(f"Initializing pipeline with config: data_dir={data_dir}, model_name={model_name}, use_gpu={use_gpu}, save_dir={save_dir}")
            
            # Check if data_dir exists
            if not os.path.exists(data_dir):
                logger.error(f"Data directory {data_dir} not found!")
                raise HTTPException(status_code=500, detail=f"Data directory {data_dir} not found. Please run data preprocessing first.")
            
            # Check if data_dir contains required files
            required_files = ['collection.pkl', 'train_queries.pkl', 'train_qrels.pkl', 'val_queries.pkl', 'val_qrels.pkl']
            missing_files = [file for file in required_files if not os.path.exists(os.path.join(data_dir, file))]
            if missing_files:
                logger.error(f"Missing required files in data directory: {missing_files}")
                raise HTTPException(status_code=500, detail=f"Missing required files in data directory: {missing_files}")
            
            # Load dataset
            logger.info("Loading dataset...")
            try:
                dataset = load_preprocessed_data(data_dir)
                logger.info(f"Dataset loaded successfully with {len(dataset.collection)} passages")
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")
            
            # Create embedding engine
            logger.info(f"Creating embedding engine with model {model_name}...")
            try:
                embedding_engine = create_embedding_engine(model_name=model_name, use_gpu=use_gpu)
                logger.info("Embedding engine created successfully")
            except Exception as e:
                logger.error(f"Error creating embedding engine: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed to create embedding engine: {str(e)}")
            
            # Create retrieval pipeline
            logger.info("Creating retrieval pipeline...")
            try:
                pipeline = RetrievalPipeline(dataset, embedding_engine)
                logger.info("Retrieval pipeline created successfully")
            except Exception as e:
                logger.error(f"Error creating retrieval pipeline: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed to create retrieval pipeline: {str(e)}")
            
            # Create save_dir if it doesn't exist
            if not os.path.exists(save_dir):
                logger.info(f"Creating save directory: {save_dir}")
                os.makedirs(save_dir, exist_ok=True)
            
            # Build or load index
            if os.path.exists(os.path.join(save_dir, "pipeline_state.pkl")):
                logger.info(f"Found existing pipeline state at {os.path.join(save_dir, 'pipeline_state.pkl')}")
                try:
                    pipeline.load(save_dir, use_gpu_index=use_gpu)
                    logger.info("Pipeline loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading pipeline: {str(e)}")
                    logger.error(traceback.format_exc())
                    # If loading fails, we'll build a new index
                    logger.info("Failed to load pipeline, will build new index")
                    try:
                        pipeline.build_index(batch_size=32)
                        pipeline.save(save_dir)
                        logger.info("New index built and saved successfully")
                    except Exception as build_error:
                        logger.error(f"Error building index: {str(build_error)}")
                        logger.error(traceback.format_exc())
                        raise HTTPException(status_code=500, detail=f"Failed to build index: {str(build_error)}")
            else:
                # Build index if not found
                logger.info("No existing pipeline state found, building new index...")
                try:
                    pipeline.build_index(batch_size=32)
                    logger.info("Index built successfully")
                    
                    # Save the pipeline
                    logger.info(f"Saving pipeline to {save_dir}...")
                    pipeline.save(save_dir)
                    logger.info("Pipeline saved successfully")
                except Exception as e:
                    logger.error(f"Error building or saving index: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise HTTPException(status_code=500, detail=f"Failed to build or save index: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unhandled exception in pipeline initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to initialize pipeline: {str(e)}")
    
    return pipeline

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    logger.info("Root endpoint accessed")
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
    logger.info("Dataset info endpoint accessed")
    try:
        available_splits = pipeline.get_available_splits()
        has_test_split = 'test' in available_splits
        
        info = {
            "available_splits": available_splits,
            "has_test_split": has_test_split,
            "is_custom_split": has_test_split,  # If test split exists, it's a custom split
            "collection_size": len(pipeline.dataset.collection),
            "train_queries": len(pipeline.dataset.train_queries),
            "val_queries": len(pipeline.dataset.val_queries),
            "test_queries": len(pipeline.dataset.test_queries) if has_test_split else 0,
        }
        logger.info(f"Dataset info response: {info}")
        return info
    except Exception as e:
        logger.error(f"Error in dataset-info endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, pipeline=Depends(get_pipeline)):
    """
    Search for relevant passages given a query
    """
    logger.info(f"Search endpoint accessed with query: {request.query}, top_k: {request.top_k}")
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
        
        logger.info(f"Search completed in {execution_time:.3f}s with {len(formatted_results)} results")
        return SearchResponse(results=formatted_results, execution_time=execution_time)
    
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest, pipeline=Depends(get_pipeline)):
    """
    Evaluate the retrieval pipeline on a dataset split
    """
    logger.info(f"Evaluate endpoint accessed with split: {request.split}, top_k: {request.top_k}, num_samples: {request.num_samples}")
    try:
        # Check if the requested split is available
        available_splits = pipeline.get_available_splits()
        if request.split not in available_splits:
            logger.error(f"Split '{request.split}' not available. Available splits: {available_splits}")
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
        
        logger.info(f"Evaluation completed with metrics: {metrics}")
        return EvaluationResponse(metrics=metrics)
    
    except ValueError as e:
        logger.error(f"ValueError in evaluate endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in evaluate endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/gpt-evaluate", response_model=GPTEvaluationResponse)
async def gpt_evaluate(request: GPTEvaluationRequest, background_tasks: BackgroundTasks):
    """
    Run GPT-4o evaluation for single passage evaluation
    """
    logger.info(f"GPT evaluation endpoint accessed with num_samples: {request.num_samples}")
    try:
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY environment variable not set.")
        
        # Run GPT-4o evaluation in background to not block the response
        background_tasks.add_task(
            validate_gpt4o_evaluation,
            data_dir=request.data_dir,
            num_samples=request.num_samples
        )
        
        logger.info(f"GPT evaluation started with {request.num_samples} samples")
        return GPTEvaluationResponse(
            success=True,
            message=f"GPT-4o evaluation started with {request.num_samples} samples. Results will be saved to gpt4o_validation_results.csv",
            results_path="gpt4o_validation_results.csv"
        )
    
    except Exception as e:
        logger.error(f"Error in gpt-evaluate endpoint: {str(e)}")
        logger.error(traceback.format_exc())
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
    logger.info(f"GPT ranking evaluation endpoint accessed with num_queries: {num_queries}, passages_per_query: {passages_per_query}")
    try:
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY environment variable not set.")
        
        # Run GPT-4o ranking evaluation in background
        background_tasks.add_task(
            validate_ranking_capability,
            data_dir=data_dir,
            num_queries=num_queries,
            passages_per_query=passages_per_query
        )
        
        logger.info(f"GPT ranking evaluation started with {num_queries} queries and {passages_per_query} passages per query")
        return GPTEvaluationResponse(
            success=True,
            message=f"GPT-4o ranking evaluation started with {num_queries} queries and {passages_per_query} passages per query. Results will be saved to gpt4o_ranking_validation.csv",
            results_path="gpt4o_ranking_validation.csv"
        )
    
    except Exception as e:
        logger.error(f"Error in gpt-ranking-evaluate endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"GPT-4o ranking evaluation failed: {str(e)}")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)