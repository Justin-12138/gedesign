import gseapy as gp
from gseapy.plot import barplot, dotplot
import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import io

# 特征选择后的基因
gene_list = ['AARS', 'AASS', 'AATK', 'ABCA10', 'ABCA2', 'ABCA6']


### 1️⃣ 富集分析（Enrichment Analysis） ###
def enrichment_analysis(genes):
    """
    进行 KEGG、GO、Reactome 富集分析，并绘制富集结果
    """
    # 选择的基因集库
    gene_sets = ['KEGG_2016', 'GO_Biological_Process_2021', 'Reactome_2022']

    for gs in gene_sets:
        enr = gp.enrichr(gene_list=genes, gene_sets=gs, organism='Human', outdir=None, cutoff=0.05)
        print(f"\n📌 {gs} 富集分析结果：")
        print(enr.results[['Term', 'Overlap', 'Adjusted P-value']].head(10))  # 显示前10个富集通路

        # 绘制富集结果
        plt.figure(figsize=(80, 40))
        sns.barplot(y=enr.results['Term'][:10], x=-enr.results['Adjusted P-value'][:10], palette='viridis')

        plt.xlabel('-log(Adjusted P-value)',fontsize=20)
        plt.ylabel('Enriched Term',fontsize=20)
        plt.title(f'{gs} Top Enriched Terms',fontsize=20)

        # ✅ 让 y 轴标签斜着显示
        plt.yticks(rotation=45,fontsize=20)  # 斜着显示富集通路名称
        plt.xticks(fontsize=20)
        plt.show()


# 执行富集分析
enrichment_analysis(gene_list)



### 2️⃣ KEGG Pathway 分析 ###
def get_kegg_pathway(gene):
    """
    获取基因对应的 KEGG Pathway，并打开通路图
    """
    kegg_url = f"https://www.kegg.jp/dbget-bin/www_bget?hsa:{gene}"
    print(f"🔗 查看 {gene} 的 KEGG Pathway: {kegg_url}")


# 查看 KEGG Pathway
for gene in gene_list:
    get_kegg_pathway(gene)


### 3️⃣ GSEA 分析（如果有 RNA-seq 数据） ###
def gsea_analysis(expression_file, cls_file):
    """
    进行 GSEA 分析（需要 RNA-seq 表达数据）
    """
    gsea_res = gp.gsea(data=expression_file,  # 基因表达数据
                       gene_sets='KEGG_2016',
                       cls=cls_file,  # 类别标签文件
                       method='signal_to_noise',
                       outdir='./gsea_results')

    # 绘制 GSEA 结果
    gsea_res.res2d.head(10).plot(kind='bar', x='Term', y='Adjusted P-value', legend=False,fontsize=20)
    plt.xlabel('Pathway',fontsize=20)
    plt.ylabel('-log(Adjusted P-value)',fontsize=20)
    plt.title('Top GSEA Enriched Pathways',fontsize=20)
    plt.show()


# 假设有 RNA-seq 数据
# gsea_analysis('your_expression_data.gct', 'class_vector.cls')  # 取消注释以运行
import io  # ✅ 添加 io 模块


def ppi_network(genes):
    """
    构建 PPI 蛋白相互作用网络
    """
    string_api_url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": "%0d".join(genes),
        "species": 9606,  # 人类
        "network_flavor": "confidence",
        "required_score": 400
    }

    response = requests.get(string_api_url, params=params)
    if response.status_code != 200:
        print("❌ 获取 STRING PPI 数据失败！")
        return

    # ✅ 使用 io.StringIO 解析 STRING API 返回的文本数据
    ppi_data = pd.read_csv(io.StringIO(response.text), sep="\t")

    G = nx.Graph()
    for _, row in ppi_data.iterrows():
        G.add_edge(row["preferredName_A"], row["preferredName_B"], weight=row["score"])

    # 绘制网络
    plt.figure(figsize=(80, 40))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=20)
    plt.title("Protein-Protein Interaction Network",fontsize=20)
    plt.show()


# 运行 PPI 网络分析
ppi_network(gene_list)
