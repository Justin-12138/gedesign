import numpy as np
from typing import List, Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
import httpx
from utils.logger import logger

class EnsembleInput(BaseModel):
    data_path: str = Field(default="../data/example_data.csv", description="data path")
    methods: List[Literal['mrmr', 'lasso', 'rfe']] = Field(default=["mrmr", "lasso", "rfe"], 
                                                          description="feature selection methods")
    ensemble_method: Literal['simple', 'stacking', 'weighted', 'voting'] = Field(
        default="simple", description="ensemble method")
    top_n: int = Field(default=10, description="number of features to select", ge=1, le=100)
    weights: List[float] = Field(default=None, description="weights for weighted average")

class EnsembleOutput(BaseModel):
    selected_features: List[str] = Field(default=[], description="selected features")
    cross_validation_results: dict = Field(default={}, description="cross validation results")

app = FastAPI()

async def get_features_from_method(method: str, data_path: str, top_n: int) -> tuple:
    """调用特征选择方法API获取特征"""
    timeout = 600 if method == 'rfe' else 60  # RFE方法设置600秒超时，其他方法60秒
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            response = await client.post(
                f"http://127.0.0.1:8000/{method}",
                json={"data_path": data_path, "top_n": top_n}
            )
            result = response.json()
            return result["features"], result["scores"]
        except httpx.TimeoutException:
            logger.error(f"Timeout occurred while calling {method} method")
            return [], []

async def perform_cross_validation(features: List[str], data_path: str) -> dict:
    """调用交叉验证API"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600)) as client:  # 交叉验证也设置600秒超时
        try:
            response = await client.post(
                "http://127.0.0.1:8000/cross",
                json={
                    "data_path": data_path,
                    "features": features,
                    "n_fold": 5,
                    "test_size": 0.2
                }
            )
            return response.json()
        except httpx.TimeoutException:
            logger.error("Timeout occurred during cross validation")
            return {}

@app.post("/simple", response_model=EnsembleOutput, tags=["Ensemble"])
async def simple_ensemble(data: EnsembleInput):
    """简单组合：取并集后选择top_n个特征"""
    all_features = set()
    for method in data.methods:
        features, _ = await get_features_from_method(method, data.data_path, data.top_n)
        all_features.update(features)
    
    selected_features = list(all_features)[:data.top_n]
    cv_results = await perform_cross_validation(selected_features, data.data_path)
    
    logger.info("Simple ensemble done!")
    return EnsembleOutput(selected_features=selected_features, 
                         cross_validation_results=cv_results)

@app.post("/stacking", response_model=EnsembleOutput, tags=["Ensemble"])
async def stacking_ensemble(data: EnsembleInput):
    """堆叠法：根据每个特征在不同方法中的排名计算最终排名"""
    feature_ranks = {}
    
    for method in data.methods:
        features, _ = await get_features_from_method(method, data.data_path, data.top_n)
        for rank, feature in enumerate(features):
            if feature not in feature_ranks:
                feature_ranks[feature] = []
            feature_ranks[feature].append(rank)
    
    # 计算每个特征的平均排名
    avg_ranks = {f: np.mean(ranks) for f, ranks in feature_ranks.items()}
    selected_features = [f for f, _ in sorted(avg_ranks.items(), 
                                            key=lambda x: x[1])[:data.top_n]]
    
    cv_results = await perform_cross_validation(selected_features, data.data_path)
    
    logger.info("Stacking ensemble done!")
    return EnsembleOutput(selected_features=selected_features, 
                         cross_validation_results=cv_results)

@app.post("/weighted", response_model=EnsembleOutput, tags=["Ensemble"])
async def weighted_ensemble(data: EnsembleInput):
    """加权平均：根据预定义权重组合不同方法的结果"""
    if not data.weights:
        data.weights = [1.0/len(data.methods)] * len(data.methods)
    
    feature_scores = {}
    
    for method, weight in zip(data.methods, data.weights):
        features, scores = await get_features_from_method(method, data.data_path, data.top_n)
        for feature, score in zip(features, scores):
            if feature not in feature_scores:
                feature_scores[feature] = 0
            feature_scores[feature] += score * weight
    
    selected_features = [f for f, _ in sorted(feature_scores.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:data.top_n]]
    
    cv_results = await perform_cross_validation(selected_features, data.data_path)
    
    logger.info("Weighted ensemble done!")
    return EnsembleOutput(selected_features=selected_features, 
                         cross_validation_results=cv_results)

@app.post("/voting", response_model=EnsembleOutput, tags=["Ensemble"])
async def voting_ensemble(data: EnsembleInput):
    """投票法：选择在多个方法中都出现的特征"""
    feature_votes = {}
    
    for method in data.methods:
        features, _ = await get_features_from_method(method, data.data_path, data.top_n)
        for feature in features:
            if feature not in feature_votes:
                feature_votes[feature] = 0
            feature_votes[feature] += 1
    
    # 选择获得票数最多的特征
    selected_features = [f for f, _ in sorted(feature_votes.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:data.top_n]]
    
    cv_results = await perform_cross_validation(selected_features, data.data_path)
    
    logger.info("Voting ensemble done!")
    return EnsembleOutput(selected_features=selected_features, 
                         cross_validation_results=cv_results)

if __name__ == "__main__":
    import uvicorn
    logger.info("Ensemble server started successfully!")
    uvicorn.run("ensemble:app", host="127.0.0.1", port=8001, reload=True)