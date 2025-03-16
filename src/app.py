import numpy as np
import pandas as pd
import uvicorn
import os
import gseapy as gp
from feature_engine.selection import MRMR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from utils.logger import logger
import requests
import networkx as nx
import io
import base64

# 添加这两行导入matplotlib
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

# 导入模型定义
from models import *

# 创建FastAPI应用
app = FastAPI(title="特征选择与分析系统", 
              description="提供特征选择、特征集成、交叉验证、富集分析和PPI网络分析功能")

# 添加CORS中间件支持前端跨域请求
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 数据预览接口
@app.post("/preview", response_model=DataPreviewOutput, tags=["Data Preview"])
async def preview_data(data: DataPreviewInput):
    """预览CSV数据文件的基本信息和样本"""
    try:
        # 读取CSV文件
        df = pd.read_csv(data.data_path)
        
        # 获取基本信息
        n_samples, n_cols = df.shape
        n_features = n_cols - 1  # 假设最后一列是目标变量
        
        # 获取特征名称
        feature_names = list(df.columns[:-1])
        
        # 判断是否需要截断列
        is_truncated = n_cols > 100
        
        # 准备预览数据
        if is_truncated:
            # 如果列数超过100，只取前3列和最后一列（目标变量）
            preview_df = df.iloc[:3, list(range(3)) + [n_cols-1]]
            # 添加省略列
            preview_df.insert(3, "...", ["..."] * 3)
        else:
            # 否则取所有列的前3行
            preview_df = df.iloc[:3]
        
        # 转换为字典列表
        preview_data = preview_df.to_dict(orient='records')
        
        logger.info(f"Data preview completed for {data.data_path}")
        return DataPreviewOutput(
            n_samples=n_samples,
            n_features=n_features,
            feature_names=feature_names,
            preview_data=preview_data,
            is_truncated=is_truncated
        )
    
    except Exception as e:
        logger.error(f"Error previewing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"无法预览数据: {str(e)}")

# 2. 特征选择接口
@app.post("/mrmr", response_model=Output, tags=["Feature Selection"])
async def mrmr(data: Input):
    """使用MRMR方法进行特征选择"""
    try:
        args = data.model_dump()
        # 读取数据
        df = pd.read_csv(args["data_path"])
        feature_names = list(df.columns[:-1])  # 取最后一列之前的特征名
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        # 执行 mRMR
        sel = MRMR(method=args["method"], regression=False)
        sel.fit(X, y)
        sel.variables_ = feature_names  # 赋值特征名
        scores = list(sel.relevance_)
        features = list(sel.variables_)
        # 按 scores 降序排序
        sorted_pairs = sorted(zip(scores, features), reverse=True, key=lambda x: x[0])
        top_scores, top_features = zip(*sorted_pairs[:args["top_n"]])
        logger.info("MRMR feature selection completed")
        return Output(scores=list(top_scores), features=list(top_features))
    except Exception as e:
        logger.error(f"Error in MRMR feature selection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MRMR feature selection failed: {str(e)}")

@app.post("/lasso", response_model=Output, tags=["Feature Selection"])
async def lasso(data: Input):
    """使用LASSO方法进行特征选择"""
    try:
        args = data.model_dump()
        # 读取数据
        df = pd.read_csv(args["data_path"])
        feature_names = list(df.columns[:-1])
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练LASSO模型
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_scaled, y)
        
        # 获取特征重要性分数
        scores = np.abs(lasso.coef_)
        
        # 按分数降序排序
        sorted_pairs = sorted(zip(scores, feature_names), reverse=True, key=lambda x: x[0])
        top_scores, top_features = zip(*sorted_pairs[:args["top_n"]])
        
        logger.info("LASSO feature selection completed")
        return Output(scores=list(top_scores), features=list(top_features))
    except Exception as e:
        logger.error(f"Error in LASSO feature selection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LASSO feature selection failed: {str(e)}")

@app.post("/rfe", response_model=Output, tags=["Feature Selection"])
async def rfe(data: Input):
    """使用RFE方法进行特征选择"""
    try:
        args = data.model_dump()
        # 读取数据
        df = pd.read_csv(args["data_path"])
        feature_names = list(df.columns[:-1])
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用随机森林作为基础估计器
        base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 创建RFE对象
        rfe_selector = RFE(estimator=base_estimator, n_features_to_select=args["top_n"], step=1)
        rfe_selector.fit(X_scaled, y)
        
        # 获取特征重要性分数
        scores = rfe_selector.ranking_
        scores = 1 / scores  # 转换ranking为分数（ranking值越小越重要）
        
        # 按分数降序排序
        sorted_pairs = sorted(zip(scores, feature_names), reverse=True, key=lambda x: x[0])
        top_scores, top_features = zip(*sorted_pairs[:args["top_n"]])
        
        logger.info("RFE feature selection completed")
        return Output(scores=list(top_scores), features=list(top_features))
    except Exception as e:
        logger.error(f"Error in RFE feature selection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RFE feature selection failed: {str(e)}")

# 3. 特征集成接口
# 集成方法辅助函数
async def get_features_from_method(method: str, data_path: str, top_n: int) -> tuple:
    """直接调用特征选择方法获取特征"""
    try:
        if method == 'mrmr':
            result = await mrmr(Input(data_path=data_path, method="MID", top_n=top_n))
        elif method == 'lasso':
            result = await lasso(Input(data_path=data_path, method="MID", top_n=top_n))
        elif method == 'rfe':
            result = await rfe(Input(data_path=data_path, method="MID", top_n=top_n))
        else:
            logger.error(f"Unknown method: {method}")
            return [], []
            
        return result.features, result.scores
    except Exception as e:
        logger.error(f"Error calling {method} method: {str(e)}")
        return [], []

@app.post("/simple", response_model=EnsembleOutput, tags=["Feature Ensemble"])
async def simple_ensemble(data: EnsembleInput):
    """简单组合：取并集后选择top_n个特征"""
    try:
        all_features = set()
        for method in data.methods:
            features, _ = await get_features_from_method(method, data.data_path, data.top_n)
            all_features.update(features)
        
        selected_features = list(all_features)[:data.top_n]
        
        logger.info("Simple ensemble completed")
        return EnsembleOutput(selected_features=selected_features)
    except Exception as e:
        logger.error(f"Error in simple ensemble: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simple ensemble failed: {str(e)}")

@app.post("/stacking", response_model=EnsembleOutput, tags=["Feature Ensemble"])
async def stacking_ensemble(data: EnsembleInput):
    """堆叠法：根据每个特征在不同方法中的排名计算最终排名"""
    try:
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
        
        logger.info("Stacking ensemble completed")
        return EnsembleOutput(selected_features=selected_features)
    except Exception as e:
        logger.error(f"Error in stacking ensemble: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stacking ensemble failed: {str(e)}")

@app.post("/weighted", response_model=EnsembleOutput, tags=["Feature Ensemble"])
async def weighted_ensemble(data: EnsembleInput):
    """加权平均：根据预定义权重组合不同方法的结果"""
    try:
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
        
        logger.info("Weighted ensemble completed")
        return EnsembleOutput(selected_features=selected_features)
    except Exception as e:
        logger.error(f"Error in weighted ensemble: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Weighted ensemble failed: {str(e)}")

@app.post("/voting", response_model=EnsembleOutput, tags=["Feature Ensemble"])
async def voting_ensemble(data: EnsembleInput):
    """投票法：选择在多个方法中都出现的特征"""
    try:
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
        
        logger.info("Voting ensemble completed")
        return EnsembleOutput(selected_features=selected_features)
    except Exception as e:
        logger.error(f"Error in voting ensemble: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voting ensemble failed: {str(e)}")

# 4. 交叉验证接口
@app.post("/cross", response_model=CrossOutput, tags=["Analysis"])
def crossval(cross_input: CrossInput):
    """对特征列表进行交叉验证，评估其性能"""
    try:
        args = cross_input.model_dump()
        df = pd.read_csv(args["data_path"])

        # 只使用选定的特征进行训练
        if not args["features"]:
            logger.info("No features selected")
            raise HTTPException(status_code=400, detail="No features selected for training")

        X = df[args["features"]].values
        y = df.iloc[:, -1].values  # 目标变量

        skf = StratifiedKFold(n_splits=args["n_fold"], shuffle=True, random_state=42)
        accuracies, f1_scores, mcc_scores, roc_aucs = [], [], [], []
        all_conf_matrices = np.zeros((2, 2))

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # 训练分类器（这里使用随机森林）
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]  # 预测概率，用于计算 ROC
            # 计算评估指标
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            conf_matrix = confusion_matrix(y_test, y_pred)

            accuracies.append(acc)
            f1_scores.append(f1)
            mcc_scores.append(mcc)
            roc_aucs.append(roc_auc)
            all_conf_matrices += conf_matrix  # 叠加混淆矩阵

        # 计算平均指标
        avg_acc = float(np.mean(accuracies))
        avg_f1 = float(np.mean(f1_scores))
        avg_mcc = np.mean(mcc_scores)
        avg_roc_auc = roc_aucs
        avg_conf_matrix = all_conf_matrices.astype(int).tolist()  # 转换为整数列表
        logger.info("Cross-validation completed")

        return CrossOutput(
            accuracy=avg_acc,
            f1_score=avg_f1,
            mcc=avg_mcc,
            roc_auc=avg_roc_auc,
            confusion_matrix=avg_conf_matrix
        )
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cross-validation failed: {str(e)}")

# 5. 富集分析接口
@app.post("/enrichment", response_model=EnrichmentOutput, tags=["Analysis"])
async def perform_enrichment_analysis(data: EnrichmentInput):
    """对特征列表进行富集分析，识别生物学通路"""
    try:
        # 去除重复基因
        unique_genes = list(dict.fromkeys(data.gene_list))
        
        results = {}
        
        # 对每个基因集进行富集分析
        for gene_set in data.gene_sets:
            # 执行富集分析
            enr = gp.enrichr(
                gene_list=unique_genes,
                gene_sets=gene_set,
                organism=data.organism,
                outdir=None,  # 不保存到文件
                cutoff=data.cutoff
            )
            
            # 保存结果
            result_list = []
            for _, row in enr.results.iterrows():
                result_list.append(
                    EnrichmentResult(
                        term=row['Term'],
                        overlap=row['Overlap'],
                        p_value=row['P-value'],
                        adjusted_p_value=row['Adjusted P-value'],
                        odds_ratio=row['Odds Ratio'],
                        combined_score=row['Combined Score'],
                        genes=row['Genes']
                    )
                )
            
            results[gene_set] = result_list
        
        logger.info("Enrichment analysis completed")
        return EnrichmentOutput(results=results)
    
    except Exception as e:
        logger.error(f"Error in enrichment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enrichment analysis failed: {str(e)}")

# 6. PPI网络分析接口
@app.post("/ppi-network", response_model=PPINetworkOutput, tags=["Analysis"])
async def ppi_network_analysis(data: PPINetworkInput):
    """构建蛋白质相互作用网络"""
    try:
        # 去除重复基因
        unique_genes = list(dict.fromkeys(data.gene_list))
        
        if not unique_genes:
            logger.error("Empty gene list provided for PPI analysis")
            raise HTTPException(status_code=400, detail="Empty gene list provided for PPI analysis")
        
        logger.info(f"Starting PPI analysis with {len(unique_genes)} genes")
        
        # 调用STRING API
        string_api_url = "https://string-db.org/api/tsv/network"
        params = {
            "identifiers": "%0d".join(unique_genes),
            "species": data.species,
            "network_flavor": data.network_type,
            "required_score": data.required_score
        }
        
        logger.info(f"Calling STRING API with params: {params}")
        
        try:
            response = requests.get(string_api_url, params=params, timeout=30)
            response.raise_for_status()  # 如果响应状态码不是200，将引发异常
        except requests.exceptions.RequestException as e:
            logger.error(f"STRING API request failed: {str(e)}")
            raise HTTPException(status_code=503, 
                               detail=f"Failed to connect to STRING database: {str(e)}")
        
        # 检查响应内容
        if not response.text.strip():
            logger.error("STRING API returned empty response")
            raise HTTPException(status_code=404, 
                               detail="No interactions found for the provided genes")
        
        # 解析返回的数据
        try:
            ppi_data = pd.read_csv(io.StringIO(response.text), sep="\t")
            logger.info(f"Received {len(ppi_data)} interactions from STRING API")
            
            if len(ppi_data) == 0:
                logger.warning("No interactions found in STRING API response")
                raise HTTPException(status_code=404, 
                                  detail="No protein-protein interactions found for the provided genes")
        except Exception as e:
            logger.error(f"Failed to parse STRING API response: {str(e)}")
            raise HTTPException(status_code=500, 
                               detail=f"Failed to parse STRING API response: {str(e)}")
        
        # 构建网络
        G = nx.Graph()
        try:
            for _, row in ppi_data.iterrows():
                G.add_edge(row["preferredName_A"], row["preferredName_B"], 
                          weight=float(row["score"]))
            
            logger.info(f"Network built with {len(G.nodes())} nodes and {len(G.edges())} edges")
            
            if len(G.nodes()) == 0:
                logger.warning("Network has no nodes")
                raise HTTPException(status_code=404, 
                                  detail="Could not build network: no valid nodes found")
        except KeyError as e:
            logger.error(f"Missing expected column in STRING API response: {str(e)}")
            raise HTTPException(status_code=500, 
                               detail=f"STRING API response format changed: missing column {str(e)}")
        except Exception as e:
            logger.error(f"Failed to build network: {str(e)}")
            raise HTTPException(status_code=500, 
                               detail=f"Failed to build network: {str(e)}")
        
        # 准备网络数据
        nodes = []
        edges = []
        try:
            for node in G.nodes():
                nodes.append({
                    "id": node,
                    "label": node,
                    "size": G.degree(node) * 3  # 节点大小与度成正比
                })
            
            for u, v, data in G.edges(data=True):
                edges.append({
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 0.5)
                })
        except Exception as e:
            logger.error(f"Failed to prepare network data: {str(e)}")
            raise HTTPException(status_code=500, 
                               detail=f"Failed to prepare network data: {str(e)}")
        
        # 生成网络图像
        try:
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                   edge_color='gray', node_size=700, font_size=8)
            plt.title("Protein-Protein Interaction Network")
            
            # 将图像转换为base64编码的字符串
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate network image: {str(e)}")
            raise HTTPException(status_code=500, 
                               detail=f"Failed to generate network visualization: {str(e)}")
        
        network_data = {
            "nodes": nodes,
            "edges": edges
        }
        
        logger.info(f"PPI network analysis completed successfully: {len(nodes)} nodes, {len(edges)} edges")
        return PPINetworkOutput(
            network_data=network_data,
            node_count=len(nodes),
            edge_count=len(edges),
            network_image=img_str
        )
    
    except HTTPException:
        # 重新抛出HTTP异常，保持原始状态码和详细信息
        raise
    except Exception as e:
        logger.error(f"Unexpected error in PPI network analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, 
                           detail=f"PPI network analysis failed due to an unexpected error: {str(e)}")

if __name__ == "__main__":
    logger.info("Feature Selection and Analysis API server started")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
