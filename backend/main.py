from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import json
import time
import traceback
from starlette.websockets import WebSocketState  # <<== Added per update 1
import logging
import uuid
import asyncio
from typing import List, Dict, Optional, Any, Set
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

# New models for operation status
class OperationStatus(BaseModel):
    id: str
    operation: str
    status: str
    title: str
    details: Optional[str] = None
    progress: Optional[float] = None
    timestamp: int
    estimated_time_remaining: Optional[int] = None

class OperationStatusList(BaseModel):
    operations: List[OperationStatus]

class CancelOperationResponse(BaseModel):
    success: bool
    message: str

# Global variables for storing the pipeline
pipeline = None
dataset = None
embedding_engine = None

# Active WebSocket connections
active_connections: Set[WebSocket] = set()

# Operation tracking
operations: Dict[str, OperationStatus] = {}

# Helper function to broadcast status updates to all WebSocket clients
async def broadcast_status_update(status_update: OperationStatus):
    # Update operation status in memory
    operations[status_update.id] = status_update
    
    # Prepare the message
    message = {
        "type": "status_update",
        "data": status_update.dict()
    }
    
    # Send to all active connections with better error handling
    disconnected_connections = []
    for connection in active_connections:
        try:
            # Check if connection is still open before sending
            if hasattr(connection, 'client_state') and connection.client_state != WebSocketState.CONNECTED:
                logger.warning(f"Client not in CONNECTED state, marking for removal")
                disconnected_connections.append(connection)
                continue
            await connection.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send status update to a client: {str(e)}")
            logger.error(traceback.format_exc())
            # Mark connection for removal
            disconnected_connections.append(connection)
    
    # Remove disconnected clients
    for connection in disconnected_connections:
        try:
            # Use discard instead of remove to avoid errors if connection is not in set
            active_connections.discard(connection)
            logger.info(f"Removed disconnected client. Active connections: {len(active_connections)}")
        except Exception as e:
            logger.error(f"Error removing disconnected client: {str(e)}")
            logger.error(traceback.format_exc())

# Helper to add a new operation and broadcast its status
async def create_operation(operation_type: str, title: str, details: Optional[str] = None):
    operation_id = str(uuid.uuid4())
    
    # Create initial status
    status = OperationStatus(
        id=operation_id,
        operation=operation_type,
        status="pending",
        title=title,
        details=details,
        progress=0,
        timestamp=int(time.time() * 1000),
        estimated_time_remaining=None
    )
    
    # Broadcast the update
    await broadcast_status_update(status)
    
    return operation_id

# Helper to update operation status
async def update_operation_status(
    operation_id: str, 
    status: str, 
    progress: Optional[float] = None,
    details: Optional[str] = None,
    estimated_time_remaining: Optional[int] = None
):
    if operation_id not in operations:
        logger.warning(f"Attempted to update non-existent operation: {operation_id}")
        return
    
    # Get the existing operation
    operation = operations[operation_id]
    
    # Update fields
    operation.status = status
    if progress is not None:
        operation.progress = progress
    if details is not None:
        operation.details = details
    operation.timestamp = int(time.time() * 1000)
    if estimated_time_remaining is not None:
        operation.estimated_time_remaining = estimated_time_remaining
    
    # Broadcast the update
    await broadcast_status_update(operation)

# Handle cancellation status
cancellation_requests: Dict[str, bool] = {}

def is_operation_cancelled(operation_id: str) -> bool:
    return cancellation_requests.get(operation_id, False)

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
            
            # Create an operation for loading dataset
            op_id = await create_operation(
                "load_dataset", 
                "Loading Dataset", 
                f"Loading dataset from {data_dir}"
            )
            
            # Load dataset
            logger.info("Loading dataset...")
            try:
                dataset = load_preprocessed_data(data_dir)
                await update_operation_status(
                    op_id, "completed", 100, 
                    f"Dataset loaded successfully with {len(dataset.collection)} passages"
                )
                logger.info(f"Dataset loaded successfully with {len(dataset.collection)} passages")
            except Exception as e:
                await update_operation_status(op_id, "error", 0, f"Failed to load dataset: {str(e)}")
                logger.error(f"Error loading dataset: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")
            
            # Create an operation for creating embedding engine
            op_id = await create_operation(
                "create_embedding_engine", 
                "Creating Embedding Engine", 
                f"Initializing model: {model_name}"
            )
            
            # Create embedding engine
            logger.info(f"Creating embedding engine with model {model_name}...")
            try:
                embedding_engine = create_embedding_engine(model_name=model_name, use_gpu=use_gpu)
                await update_operation_status(op_id, "completed", 100, "Embedding engine created successfully")
                logger.info("Embedding engine created successfully")
            except Exception as e:
                await update_operation_status(op_id, "error", 0, f"Failed to create embedding engine: {str(e)}")
                logger.error(f"Error creating embedding engine: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed to create embedding engine: {str(e)}")
            
            # Create an operation for creating pipeline
            op_id = await create_operation(
                "create_pipeline", 
                "Creating Retrieval Pipeline", 
                "Initializing retrieval pipeline"
            )
            
            # Create retrieval pipeline
            logger.info("Creating retrieval pipeline...")
            try:
                # Pass the status update function to the pipeline
                async def status_callback(status, progress, details=None, est_time_remaining=None):
                    await update_operation_status(
                        op_id, status, progress, details, est_time_remaining
                    )
                
                pipeline = RetrievalPipeline(
                    dataset, 
                    embedding_engine,
                    status_callback=status_callback,
                    cancellation_check=lambda: is_operation_cancelled(op_id)
                )
                await update_operation_status(op_id, "completed", 100, "Retrieval pipeline created successfully")
                logger.info("Retrieval pipeline created successfully")
            except Exception as e:
                await update_operation_status(op_id, "error", 0, f"Failed to create retrieval pipeline: {str(e)}")
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
                
                # Create an operation for loading index
                op_id = await create_operation(
                    "load_index", 
                    "Loading Search Index", 
                    f"Loading index from {save_dir}"
                )
                
                try:
                    # Update pipeline with new operation ID for load
                    pipeline.current_operation_id = op_id
                    
                    # Load pipeline with progress tracking
                    pipeline.load(
                        save_dir, 
                        use_gpu_index=use_gpu
                    )
                    await update_operation_status(op_id, "completed", 100, "Index loaded successfully")
                    logger.info("Pipeline loaded successfully")
                except Exception as e:
                    await update_operation_status(op_id, "error", 0, f"Error loading index: {str(e)}")
                    logger.error(f"Error loading pipeline: {str(e)}")
                    logger.error(traceback.format_exc())
                    # If loading fails, we'll build a new index
                    logger.info("Failed to load pipeline, will build new index")
                    
                    # Create an operation for building index
                    build_op_id = await create_operation(
                        "build_index", 
                        "Building Search Index", 
                        "Creating new search index"
                    )
                    
                    try:
                        # Update pipeline with new operation ID for build
                        pipeline.current_operation_id = build_op_id
                        
                        # Build index with progress tracking
                        pipeline.build_index(batch_size=32)
                        pipeline.save(save_dir)
                        await update_operation_status(build_op_id, "completed", 100, "New index built and saved successfully")
                        logger.info("New index built and saved successfully")
                    except Exception as build_error:
                        await update_operation_status(build_op_id, "error", 0, f"Failed to build index: {str(build_error)}")
                        logger.error(f"Error building index: {str(build_error)}")
                        logger.error(traceback.format_exc())
                        raise HTTPException(status_code=500, detail=f"Failed to build index: {str(build_error)}")
            else:
                # Build index if not found
                logger.info("No existing pipeline state found, building new index...")
                
                # Create an operation for building index
                op_id = await create_operation(
                    "build_index", 
                    "Building Search Index", 
                    "Creating new search index"
                )
                
                try:
                    # Update pipeline with new operation ID for build
                    pipeline.current_operation_id = op_id
                    
                    # Build index with progress tracking
                    pipeline.build_index(batch_size=32)
                    await update_operation_status(op_id, "completed", 100, "Index built successfully")
                    logger.info("Index built successfully")
                    
                    # Save the pipeline
                    save_op_id = await create_operation(
                        "save_pipeline", 
                        "Saving Pipeline", 
                        f"Saving pipeline to {save_dir}"
                    )
                    pipeline.current_operation_id = save_op_id
                    
                    logger.info(f"Saving pipeline to {save_dir}...")
                    pipeline.save(save_dir)
                    await update_operation_status(save_op_id, "completed", 100, "Pipeline saved successfully")
                    logger.info("Pipeline saved successfully")
                except Exception as e:
                    await update_operation_status(op_id, "error", 0, f"Failed to build or save index: {str(e)}")
                    logger.error(f"Error building or saving index: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise HTTPException(status_code=500, detail=f"Failed to build or save index: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unhandled exception in pipeline initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to initialize pipeline: {str(e)}")
    
    return pipeline

# WebSocket endpoint for real-time status updates
@app.websocket("/ws/status")
async def websocket_status_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(active_connections)}")
        
        # Send all current operations as initial state
        for op_id, status in operations.items():
            try:
                await websocket.send_json({
                    "type": "status_update",
                    "data": status.dict()
                })
                # Add a small delay to avoid overwhelming the connection
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error sending initial status: {str(e)}")
                logger.error(traceback.format_exc())
                break
        
        # Keep the connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                # Process message...
            except WebSocketDisconnect as e:
                logger.info(f"WebSocket client disconnected during receive: code={getattr(e, 'code', 'unknown')}")
                break
            except Exception as e:
                logger.error(f"WebSocket receive error: {str(e)}")
                logger.error(traceback.format_exc())
                break
                
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket client disconnected: code={getattr(e, 'code', 'unknown')}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Use discard instead of remove to avoid errors
        active_connections.discard(websocket)
        logger.info(f"WebSocket connection closed. Active connections: {len(active_connections)}")

# New endpoints for operation status
@app.get("/operations/status", response_model=OperationStatusList)
async def get_operations_status():
    """Get status of all current operations"""
    return OperationStatusList(operations=list(operations.values()))

@app.get("/operations/status/{operation_id}", response_model=OperationStatus)
async def get_operation_status(operation_id: str):
    """Get status of a specific operation"""
    if operation_id not in operations:
        raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")
    return operations[operation_id]

@app.post("/operations/cancel/{operation_id}", response_model=CancelOperationResponse)
async def cancel_operation_endpoint(operation_id: str):
    """Cancel an operation"""
    if operation_id not in operations:
        raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")
    
    operation = operations[operation_id]
    if operation.status in ["completed", "error", "canceled"]:
        return CancelOperationResponse(
            success=False,
            message=f"Operation {operation_id} already in final state: {operation.status}"
        )
    
    # Mark for cancellation
    cancellation_requests[operation_id] = True
    
    # Update status
    await update_operation_status(
        operation_id, 
        "canceling", 
        details="Cancellation requested"
    )
    
    logger.info(f"Cancellation requested for operation {operation_id}")
    
    return CancelOperationResponse(
        success=True,
        message=f"Cancellation requested for operation {operation_id}"
    )

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
        # Create an operation for search
        op_id = await create_operation(
            "search", 
            "Searching Documents", 
            f"Query: {request.query[:30]}{'...' if len(request.query) > 30 else ''}"
        )
        
        # Record start time
        start_time = time.time()
        
        # Update pipeline with operation ID
        pipeline.current_operation_id = op_id
        
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
        
        # Update operation status
        await update_operation_status(
            op_id, 
            "completed", 
            100, 
            f"Search completed with {len(formatted_results)} results"
        )
        
        logger.info(f"Search completed in {execution_time:.3f}s with {len(formatted_results)} results")
        return SearchResponse(results=formatted_results, execution_time=execution_time)
    
    except Exception as e:
        if 'op_id' in locals():
            await update_operation_status(op_id, "error", 0, f"Search failed: {str(e)}")
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
        
        # Create an operation for evaluation
        op_id = await create_operation(
            "evaluate", 
            f"Evaluating on {request.split} split", 
            f"Evaluating top {request.top_k} results" + 
            (f" with {request.num_samples} samples" if request.num_samples else " with all samples")
        )
        
        # Update pipeline with operation ID
        pipeline.current_operation_id = op_id
        
        # Run evaluation
        metrics = pipeline.evaluate(
            split=request.split,
            top_k=request.top_k,
            num_samples=request.num_samples
        )
        
        # Update operation status
        await update_operation_status(
            op_id, 
            "completed", 
            100, 
            f"Evaluation completed with MRR: {metrics.get('mrr', 0):.4f}"
        )
        
        logger.info(f"Evaluation completed with metrics: {metrics}")
        return EvaluationResponse(metrics=metrics)
    
    except ValueError as e:
        if 'op_id' in locals():
            await update_operation_status(op_id, "error", 0, f"Evaluation failed: {str(e)}")
        logger.error(f"ValueError in evaluate endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if 'op_id' in locals():
            await update_operation_status(op_id, "error", 0, f"Evaluation failed: {str(e)}")
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
        
        # Create operation for GPT evaluation
        op_id = await create_operation(
            "gpt_evaluate", 
            "GPT-4o Evaluation", 
            f"Running evaluation with {request.num_samples} samples"
        )
        
        # Modified evaluation function to report progress
        async def run_with_progress_updates():
            try:
                # Wrap the synchronous evaluation function
                sample_count = 0
                total_samples = request.num_samples
                
                def progress_callback(sample_index):
                    nonlocal sample_count
                    sample_count = sample_index + 1
                    progress = int((sample_count / total_samples) * 100)
                    # Use asyncio to run the coroutine from a non-async context
                    asyncio.create_task(
                        update_operation_status(
                            op_id, 
                            "in-progress", 
                            progress, 
                            f"Processed {sample_count}/{total_samples} samples"
                        )
                    )
                    # Check for cancellation
                    return is_operation_cancelled(op_id)
                
                # Run the evaluation with progress tracking
                results = validate_gpt4o_evaluation(
                    data_dir=request.data_dir,
                    num_samples=request.num_samples,
                    progress_callback=progress_callback
                )
                
                # Update status on completion
                await update_operation_status(
                    op_id, 
                    "completed", 
                    100, 
                    f"GPT-4o evaluation completed. Results saved to gpt4o_validation_results.csv"
                )
                
            except Exception as e:
                logger.error(f"Error in GPT evaluation: {str(e)}")
                await update_operation_status(
                    op_id, 
                    "error", 
                    0, 
                    f"GPT evaluation failed: {str(e)}"
                )
        
        # Run the evaluation in background
        background_tasks.add_task(run_with_progress_updates)
        
        logger.info(f"GPT evaluation started with {request.num_samples} samples")
        return GPTEvaluationResponse(
            success=True,
            message=f"GPT-4o evaluation started with {request.num_samples} samples. Results will be saved to gpt4o_validation_results.csv",
            results_path="gpt4o_validation_results.csv"
        )
    
    except Exception as e:
        if 'op_id' in locals():
            await update_operation_status(op_id, "error", 0, f"GPT evaluation failed: {str(e)}")
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
        
        # Create operation for GPT ranking evaluation
        op_id = await create_operation(
            "gpt_ranking_evaluate", 
            "GPT-4o Ranking Evaluation", 
            f"Evaluating rankings for {num_queries} queries with {passages_per_query} passages each"
        )
        
        # Modified ranking evaluation function to report progress
        async def run_with_progress_updates():
            try:
                # Wrap the synchronous evaluation function
                query_count = 0
                total_queries = num_queries
                
                def progress_callback(query_index):
                    nonlocal query_count
                    query_count = query_index + 1
                    progress = int((query_count / total_queries) * 100)
                    # Use asyncio to run the coroutine from a non-async context
                    asyncio.create_task(
                        update_operation_status(
                            op_id, 
                            "in-progress", 
                            progress, 
                            f"Processed {query_count}/{total_queries} queries"
                        )
                    )
                    # Check for cancellation
                    return is_operation_cancelled(op_id)
                
                # Run the evaluation with progress tracking
                results = validate_ranking_capability(
                    data_dir=data_dir,
                    num_queries=num_queries,
                    passages_per_query=passages_per_query,
                    progress_callback=progress_callback
                )
                
                # Update status on completion
                await update_operation_status(
                    op_id, 
                    "completed", 
                    100, 
                    f"GPT-4o ranking evaluation completed. Results saved to gpt4o_ranking_validation.csv"
                )
                
            except Exception as e:
                logger.error(f"Error in GPT ranking evaluation: {str(e)}")
                await update_operation_status(
                    op_id, 
                    "error", 
                    0, 
                    f"GPT ranking evaluation failed: {str(e)}"
                )
        
        # Run the evaluation in background
        background_tasks.add_task(run_with_progress_updates)
        
        logger.info(f"GPT ranking evaluation started with {num_queries} queries and {passages_per_query} passages per query")
        return GPTEvaluationResponse(
            success=True,
            message=f"GPT-4o ranking evaluation started with {num_queries} queries and {passages_per_query} passages per query. Results will be saved to gpt4o_ranking_validation.csv",
            results_path="gpt4o_ranking_validation.csv"
        )
    
    except Exception as e:
        if 'op_id' in locals():
            await update_operation_status(op_id, "error", 0, f"GPT ranking evaluation failed: {str(e)}")
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
