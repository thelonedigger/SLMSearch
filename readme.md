Semantic Document Retrieval with RL Optimization
Overview
This project implements a semantic document retrieval system using MiniLM embeddings and optimizes it with reinforcement learning. It uses GPT-4o to evaluate retrieval quality against the MS MARCO dataset.

Features:

- Semantic Document Retrieval: Uses sentence-transformers MiniLM models to embed documents
- FAISS Vector Search: Fast approximate nearest neighbor search using FAISS
- Automated Evaluation: GPT-4o for automated retrieval quality assessment
- Reinforcement Learning: Parameter optimization for chunking, similarity metrics, and re-ranking
- Comprehensive Metrics: MRR, nDCG, Precision@k for evaluation

Project Structure

```
├── DataPreprocessing/
│   └── splitting.py         # Preprocesses MS MARCO dataset
├── backend/
│   ├── datapipeline/
│   │   ├── data_handler.py       # Dataset management
│   │   ├── embedding_engine.py   # Embeddings generation
│   │   └── retrieval_pipeline.py # End-to-end search pipeline
│   └── gptevaluator/
│       ├── evaluation_prompts.py # GPT-4o prompts
│       ├── gpt4o_validation.py   # Validation framework
│       └── run_validation.py     # Validation runner
```

Licence

MIT License