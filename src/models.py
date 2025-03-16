from pydantic import BaseModel,Field
from typing import List,Literal,Dict,Any

class Input(BaseModel):
    data_path: str = Field(default="../data/example_data.csv", description="data path")
    method: Literal['MIQ', 'MID', 'FCQ', 'FCD', 'RFCQ','lasso','rfe'] = Field(default="MID", description="method")
    top_n: int = Field(default=10, description="top n value", ge=1, le=100)

class Output(BaseModel):
    scores: List[float] = Field(default=[], description="scores")
    features: List[str] = Field(default=[], description="features")

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

# 添加数据预览模型
class DataPreviewInput(BaseModel):
    data_path: str = Field(default="../data/example_data.csv", description="数据文件路径")

class DataPreviewOutput(BaseModel):
    n_samples: int = Field(description="样本数量（行数）")
    n_features: int = Field(description="特征数量（列数）")
    feature_names: List[str] = Field(description="特征名称列表")
    preview_data: List[Dict[str, Any]] = Field(description="前3行数据预览")
    is_truncated: bool = Field(description="是否截断了列（超过100列时）")

# 修改富集分析模型
class EnrichmentInput(BaseModel):
    gene_list: List[str] = Field(..., description="List of genes for enrichment analysis")
    gene_sets: List[str] = Field(
        default=["KEGG_2016", "GO_Biological_Process_2021", "Reactome_2022"],
        description="Gene set databases to use for enrichment"
    )
    organism: str = Field(default="Human", description="Organism for analysis")
    cutoff: float = Field(default=0.05, description="P-value cutoff for significance")
    output_dir: str = Field(default="./data/enrichr", description="Directory to save results")

class EnrichmentResult(BaseModel):
    term: str = Field(..., description="Enriched term name")
    overlap: str = Field(..., description="Overlap ratio (e.g., '3/44')")
    p_value: float = Field(..., description="P-value")
    adjusted_p_value: float = Field(..., description="Adjusted P-value")
    odds_ratio: float = Field(..., description="Odds ratio")
    combined_score: float = Field(..., description="Combined score")
    genes: str = Field(..., description="Genes in this term")

class EnrichmentOutput(BaseModel):
    results: Dict[str, List[EnrichmentResult]] = Field(..., description="Enrichment results for each gene set")

# 添加PPI网络分析模型
class PPINetworkInput(BaseModel):
    gene_list: List[str] = Field(..., description="List of genes for PPI network analysis")
    species: int = Field(default=9606, description="Species ID (default: 9606 for human)")
    required_score: int = Field(default=400, description="Minimum required interaction score (0-1000)")
    network_type: str = Field(default="confidence", description="Network type")

class PPINetworkOutput(BaseModel):
    network_data: Dict[str, Any] = Field(..., description="Network data for visualization")
    node_count: int = Field(..., description="Number of nodes in the network")
    edge_count: int = Field(..., description="Number of edges in the network")
    network_image: str = Field(..., description="Base64 encoded network image")