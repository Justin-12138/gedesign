import numpy as np
import pandas as pd
import uvicorn
from feature_engine.selection import MRMR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel, Field
from typing import List, Literal
from fastapi import FastAPI
from utils.logger import logger


class Input(BaseModel):
    data_path: str = Field(default="../data/example_data.csv", description="data path")
    method: Literal['MIQ', 'MID', 'FCQ', 'FCD', 'RFCQ'] = Field(default="MID", description="method")
    top_n: int = Field(default=10, description="top n value", ge=1, le=100)


class Output(BaseModel):
    scores: List[float] = Field(default=[], description="scores")
    features: List[str] = Field(default=[], description="features")


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/mrmr", response_model=Output, tags=["Feature selection"])
async def mrmr(data: Input):
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
    logger.info("MRMR done!")
    return Output(scores=list(top_scores), features=list(top_features))


@app.post(path="/lasso", response_model=Output, tags=["Feature selection"])
async def lasso(data: Input):
    args = data.model_dump()
    # 读取数据
    df = pd.read_csv(args["data_path"])
    feature_names = list(df.columns[:-1])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 使用LASSO进行特征选择
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    
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
    
    logger.info("LASSO feature selection done!")
    return Output(scores=list(top_scores), features=list(top_features))


@app.post(path="/rfe", response_model=Output, tags=["Feature selection"])
async def rfe(data: Input):
    args = data.model_dump()
    # 读取数据
    df = pd.read_csv(args["data_path"])
    feature_names = list(df.columns[:-1])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 使用RFE进行特征选择
    from sklearn.feature_selection import RFE
    from sklearn.preprocessing import StandardScaler
    
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
    
    logger.info("RFE feature selection done!")
    return Output(scores=list(top_scores), features=list(top_features))


class CrossInput(BaseModel):
    data_path: str = Field(default="../data/example_data.csv", description="data path")
    features: List[str] = Field(default=[], description="Selected features")
    n_fold: int = Field(default=5, description="number of folds")
    test_size: float = Field(default=0.2, description="test size")


class CrossOutput(BaseModel):
    accuracy: float = Field(default=0.0, description="Accuracy")
    f1_score: float = Field(default=0.0, description="F1-score")
    mcc: float = Field(default=0.0, description="Matthews Correlation Coefficient")
    roc_auc: List[float] = Field(default=0.0, description="Area under ROC curve")
    confusion_matrix: List[List[int]] = Field(default=[], description="Confusion Matrix")


@app.post("/cross", response_model=CrossOutput, tags=["Cross Validation"])
def crossval(cross_input: CrossInput):
    args = cross_input.model_dump()
    df = pd.read_csv(args["data_path"])

    # 只使用选定的特征进行训练
    if not args["features"]:
        logger.info("No features selected")
        raise ValueError("No features selected for training")

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
    logger.info("Crossval done!")


    return CrossOutput(
        accuracy=avg_acc,
        f1_score=avg_f1,
        mcc=avg_mcc,
        roc_auc=avg_roc_auc,
        confusion_matrix=avg_conf_matrix
    )


if __name__ == "__main__":
    logger.info("server started successfully！")
    uvicorn.run("fs:app", host="127.0.0.1", port=8000, reload=True)
