from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import config
from src.ml_agent import SQLBoostMLAgent

app = FastAPI(
    title="SQLBoost API",
    description="SQL Query Optimization API using Machine Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = None

class QueryRequest(BaseModel):
    query: str

class RewriteRequest(BaseModel):
    query: str
    transformation: Optional[str] = None

class OptimizationResponse(BaseModel):
    original_query: str
    performance_comparison: str
    best_optimization: Optional[Dict[str, Any]]
    all_candidates_evaluated: int
    candidates: List[Dict[str, Any]]
    successful_optimizations: int
    recommendations: List[str]
    agent_insights: Dict[str, Any]

class RewriteResponse(BaseModel):
    original_query: str
    transformations: Dict[str, List[str]]

class StatusResponse(BaseModel):
    knowledge_base_patterns: int
    optimizations_learned: int
    strategies_available: int
    exploration_rate: float
    learning_rate: float
    strategy_effectiveness: Dict[str, Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    global agent
    try:
        db_config = config.get_database_config()
        agent = SQLBoostMLAgent(db_config)
    except Exception as e:
        print(f"Failed to initialize agent: {e}")

@app.get("/")
async def root():
    return {"message": "SQLBoost API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_query(request: QueryRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = agent.optimize_query(request.query)
        return OptimizationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite_query(request: RewriteRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        transformations = {}

        if request.transformation:
            rewritten = agent.query_rewriter.rewrite_query(request.query, request.transformation)
            transformations[request.transformation] = rewritten
        else:
            available_transforms = agent.query_rewriter.get_supported_transformations()
            for transform in available_transforms:
                rewritten = agent.query_rewriter.rewrite_query(request.query, transform)
                if rewritten and (len(rewritten) > 1 or (len(rewritten) == 1 and rewritten[0] != request.query)):
                    transformations[transform] = rewritten

        return RewriteResponse(
            original_query=request.query,
            transformations=transformations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rewrite failed: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        status = agent.get_agent_status()
        return StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@app.get("/transforms")
async def get_transformations():
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        transforms = agent.query_rewriter.get_supported_transformations()
        return {"transformations": transforms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform retrieval failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)