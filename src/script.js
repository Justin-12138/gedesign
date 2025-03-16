// 全局变量
const API_URL = 'http://127.0.0.1:8000';
let currentFeatures = []; // 当前选择的特征列表
let currentOperation = ''; // 当前操作类型：'feature-selection' 或 'ensemble'

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化事件监听器
    initEventListeners();
    
    // 初始化步骤指示器
    updateStepIndicator(1);
    
    // 初始化MRMR方法下拉菜单
    populateMRMRMethods();
});

// 初始化事件监听器
function initEventListeners() {
    // 数据预览按钮
    document.getElementById('preview-btn').addEventListener('click', previewData);
    
    // 特征选择按钮
    document.getElementById('fs-btn').addEventListener('click', performFeatureSelection);
    
    // 特征集成按钮
    document.getElementById('ensemble-btn').addEventListener('click', performEnsemble);
    
    // 特征选择后的交叉验证按钮
    document.getElementById('fs-cv-btn').addEventListener('click', function() {
        performCrossValidation(currentFeatures);
    });
    
    // 特征集成后的交叉验证按钮
    document.getElementById('ensemble-cv-btn').addEventListener('click', function() {
        performCrossValidation(currentFeatures);
    });
    
    // 特征选择后的富集分析按钮
    document.getElementById('fs-enrichment-btn').addEventListener('click', function() {
        document.getElementById('enrichment-section').style.display = 'block';
        document.getElementById('enrichment-section').scrollIntoView({ behavior: 'smooth' });
        currentOperation = 'feature-selection';
    });
    
    // 特征集成后的富集分析按钮
    document.getElementById('ensemble-enrichment-btn').addEventListener('click', function() {
        document.getElementById('enrichment-section').style.display = 'block';
        document.getElementById('enrichment-section').scrollIntoView({ behavior: 'smooth' });
        currentOperation = 'ensemble';
    });
    
    // 富集分析按钮
    document.getElementById('run-enrichment-btn').addEventListener('click', performEnrichmentAnalysis);
    
    // 特征选择后的PPI网络分析按钮
    document.getElementById('fs-ppi-btn').addEventListener('click', function() {
        document.getElementById('ppi-section').style.display = 'block';
        document.getElementById('ppi-section').scrollIntoView({ behavior: 'smooth' });
        currentOperation = 'feature-selection';
    });
    
    // 特征集成后的PPI网络分析按钮
    document.getElementById('ensemble-ppi-btn').addEventListener('click', function() {
        document.getElementById('ppi-section').style.display = 'block';
        document.getElementById('ppi-section').scrollIntoView({ behavior: 'smooth' });
        currentOperation = 'ensemble';
    });
    
    // PPI网络分析按钮
    document.getElementById('run-ppi-btn').addEventListener('click', performPPINetworkAnalysis);
    
    // 初始隐藏分析部分
    document.getElementById('enrichment-section').style.display = 'none';
    document.getElementById('ppi-section').style.display = 'none';
}

// 更新步骤指示器
function updateStepIndicator(stepNumber) {
    const steps = document.querySelectorAll('.step');
    steps.forEach((step, index) => {
        if (index + 1 <= stepNumber) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
}

// 填充MRMR方法下拉菜单
function populateMRMRMethods() {
    const methodSelect = document.getElementById('fs-method');
    const methods = ['MIQ', 'MID', 'FCQ', 'FCD', 'RFCQ','lasso','rfe'];
    
    methods.forEach(method => {
        const option = document.createElement('option');
        option.value = method;
        option.textContent = method;
        if (method === 'MID') {
            option.selected = true;
        }
        methodSelect.appendChild(option);
    });
}

// 获取数据路径
function getDataPath() {
    return document.getElementById('data-path').value;
}

// 显示加载动画
function showLoading(id) {
    const loadingElement = document.getElementById(id);
    if (loadingElement) {
        loadingElement.style.display = 'block';
    }
}

// 隐藏加载动画
function hideLoading(id) {
    const loadingElement = document.getElementById(id);
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
}

// 显示结果区域
function showResults(id) {
    const resultsElement = document.getElementById(id);
    if (resultsElement) {
        resultsElement.classList.remove('d-none');
    }
}

// 隐藏结果区域
function hideResults(id) {
    const resultsElement = document.getElementById(id);
    if (resultsElement) {
        resultsElement.classList.add('d-none');
    }
}

// 显示通知
function showNotification(message, isError = false) {
    const notification = document.createElement('div');
    notification.className = `notification ${isError ? 'error' : ''}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // 显示通知
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // 3秒后隐藏通知
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// 数据预览
async function previewData() {
    const dataPath = getDataPath();
    
    if (!dataPath) {
        showNotification('请输入数据文件路径', true);
        return;
    }
    
    showLoading('preview-loading');
    hideResults('preview-results');
    
    try {
        const response = await fetch(`${API_URL}/preview`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                data_path: dataPath
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        // 更新预览信息
        document.getElementById('sample-count').textContent = data.n_samples;
        document.getElementById('feature-count').textContent = data.n_features;
        
        // 清空并填充预览表格
        const tableBody = document.getElementById('preview-table-body');
        tableBody.innerHTML = '';
        
        // 创建表头
        const tableHead = document.getElementById('preview-table-head');
        tableHead.innerHTML = '';
        const headerRow = document.createElement('tr');
        
        if (data.preview_data.length > 0) {
            const firstRow = data.preview_data[0];
            for (const key in firstRow) {
                const th = document.createElement('th');
                th.textContent = key;
                headerRow.appendChild(th);
            }
            tableHead.appendChild(headerRow);
            
            // 填充表格内容
            data.preview_data.forEach(row => {
                const tr = document.createElement('tr');
                for (const key in row) {
                    const td = document.createElement('td');
                    td.textContent = row[key];
                    tr.appendChild(td);
                }
                tableBody.appendChild(tr);
            });
        }
        
        showResults('preview-results');
        updateStepIndicator(2); // 更新步骤指示器
        showNotification('数据预览成功');
    } catch (error) {
        console.error('Error:', error);
        showNotification('数据预览失败: ' + error.message, true);
    } finally {
        hideLoading('preview-loading');
    }
}

// 执行特征选择
async function performFeatureSelection() {
    const dataPath = getDataPath();
    const method = document.getElementById('fs-method').value;
    const topN = document.getElementById('fs-top-n').value;
    
    if (!dataPath) {
        showNotification('请输入数据文件路径', true);
        return;
    }
    
    showLoading('fs-loading');
    hideResults('fs-results');
    
    try {
        let endpoint;
        if (method.startsWith('M') || method.startsWith('F') || method.startsWith('R')) {
            endpoint = '/mrmr';
        } else if (method.toLowerCase() === 'lasso') {
            endpoint = '/lasso';
        } else if (method.toLowerCase() === 'rfe') {
            endpoint = '/rfe';
        } else {
            throw new Error('未知的特征选择方法');
        }
        
        const response = await fetch(`${API_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                data_path: dataPath,
                method: method,
                top_n: parseInt(topN)
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        // 更新全局特征列表
        currentFeatures = data.features;
        currentOperation = 'feature-selection';
        
        // 显示特征和分数
        displayFeatures(data.features, data.scores);
        
        showResults('fs-results');
        updateStepIndicator(3); // 更新步骤指示器
        showNotification('特征选择成功');
    } catch (error) {
        console.error('Error:', error);
        showNotification('特征选择失败: ' + error.message, true);
    } finally {
        hideLoading('fs-loading');
    }
}

// 显示特征和分数
function displayFeatures(features, scores) {
    const featuresContainer = document.getElementById('selected-features');
    featuresContainer.innerHTML = '';
    
    features.forEach((feature, index) => {
        const score = scores[index];
        const featureElement = document.createElement('div');
        featureElement.className = 'feature-badge';
        featureElement.textContent = `${feature} (${score.toFixed(4)})`;
        featuresContainer.appendChild(featureElement);
    });
}

// 执行特征集成
async function performEnsemble() {
    const dataPath = getDataPath();
    const ensembleMethod = document.getElementById('ensemble-method').value;
    const topN = document.getElementById('ensemble-top-n').value;
    
    // 获取选中的方法
    const methods = [];
    if (document.getElementById('check-mrmr').checked) methods.push('mrmr');
    if (document.getElementById('check-lasso').checked) methods.push('lasso');
    if (document.getElementById('check-rfe').checked) methods.push('rfe');
    
    if (methods.length === 0) {
        showNotification('请至少选择一种特征选择方法', true);
        return;
    }
    
    if (!dataPath) {
        showNotification('请输入数据文件路径', true);
        return;
    }
    
    showLoading('ensemble-loading');
    hideResults('ensemble-results');
    
    try {
        const response = await fetch(`${API_URL}/${ensembleMethod}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                data_path: dataPath,
                methods: methods,
                ensemble_method: ensembleMethod,
                top_n: parseInt(topN)
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        // 更新全局特征列表
        currentFeatures = data.selected_features;
        currentOperation = 'ensemble';
        
        // 显示集成后的特征
        const featuresContainer = document.getElementById('ensemble-features');
        featuresContainer.innerHTML = '';
        
        data.selected_features.forEach(feature => {
            const badge = document.createElement('div');
            badge.className = 'feature-badge';
            badge.textContent = feature;
            featuresContainer.appendChild(badge);
        });
        
        showResults('ensemble-results');
        updateStepIndicator(3); // 更新步骤指示器
        showNotification('特征集成成功');
    } catch (error) {
        console.error('Error:', error);
        showNotification('特征集成失败: ' + error.message, true);
    } finally {
        hideLoading('ensemble-loading');
    }
}

// 执行交叉验证
async function performCrossValidation(features) {
    if (features.length === 0) {
        showNotification('没有可用的特征进行交叉验证', true);
        return;
    }
    
    const dataPath = getDataPath();
    const nFold = document.getElementById('cv-folds').value;
    const testSize = document.getElementById('cv-test-size').value;
    
    hideResults('cv-results');
    showLoading('cv-loading');
    
    try {
        const response = await fetch(`${API_URL}/cross`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                data_path: dataPath,
                features: features,
                n_fold: parseInt(nFold),
                test_size: parseFloat(testSize)
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        displayCrossValidationResults(data);
        showNotification('交叉验证成功');
    } catch (error) {
        console.error('Error:', error);
        showNotification('交叉验证失败: ' + error.message, true);
    } finally {
        hideLoading('cv-loading');
    }
}

// 显示交叉验证结果
function displayCrossValidationResults(data) {
    // 更新指标
    document.getElementById('cv-accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
    document.getElementById('cv-f1').textContent = (data.f1_score * 100).toFixed(2) + '%';
    document.getElementById('cv-mcc').textContent = data.mcc.toFixed(4);
    
    // 如果 roc_auc 是数组，取平均值
    const rocAuc = Array.isArray(data.roc_auc) ? 
        data.roc_auc.reduce((a, b) => a + b, 0) / data.roc_auc.length : 
        data.roc_auc;
    document.getElementById('cv-roc').textContent = (rocAuc * 100).toFixed(2) + '%';
    
    // 更新混淆矩阵
    if (data.confusion_matrix && data.confusion_matrix.length === 2) {
        document.getElementById('cm-tn').textContent = data.confusion_matrix[0][0];
        document.getElementById('cm-fp').textContent = data.confusion_matrix[0][1];
        document.getElementById('cm-fn').textContent = data.confusion_matrix[1][0];
        document.getElementById('cm-tp').textContent = data.confusion_matrix[1][1];
    }
    
    showResults('cv-results');
}

// 执行富集分析
async function performEnrichmentAnalysis() {
    if (currentFeatures.length === 0) {
        showNotification('没有可用的特征进行富集分析', true);
        return;
    }
    
    hideResults('enrichment-results');
    showLoading('enrichment-loading');
    
    // 获取选中的基因集
    const geneSets = [];
    if (document.getElementById('check-kegg').checked) geneSets.push('KEGG_2016');
    if (document.getElementById('check-go').checked) geneSets.push('GO_Biological_Process_2021');
    if (document.getElementById('check-reactome').checked) geneSets.push('Reactome_2022');
    
    if (geneSets.length === 0) {
        showNotification('请至少选择一个基因集数据库', true);
        hideLoading('enrichment-loading');
        return;
    }
    
    // 获取富集分析参数
    const enrichmentParams = {
        gene_list: currentFeatures,
        gene_sets: geneSets,
        organism: document.getElementById('enrichment-organism').value,
        cutoff: parseFloat(document.getElementById('enrichment-cutoff').value)
    };
    
    try {
        const response = await fetch(`${API_URL}/enrichment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(enrichmentParams),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        // 显示结果
        displayEnrichmentResults(data);
        showNotification('富集分析成功');
    } catch (error) {
        console.error('Error:', error);
        showNotification('富集分析失败: ' + error.message, true);
    } finally {
        hideLoading('enrichment-loading');
    }
}

// 显示富集分析结果
function displayEnrichmentResults(data) {
    // 清空标签页
    const tabsContainer = document.getElementById('enrichment-tabs');
    const tabContentContainer = document.getElementById('enrichment-tab-content');
    tabsContainer.innerHTML = '';
    tabContentContainer.innerHTML = '';
    
    // 为每个基因集创建标签页
    let isFirst = true;
    for (const geneSet in data.results) {
        // 创建标签
        const tabId = `tab-${geneSet.replace(/[^a-zA-Z0-9]/g, '-')}`;
        const tabLink = document.createElement('li');
        tabLink.className = 'nav-item';
        tabLink.innerHTML = `
            <a class="nav-link ${isFirst ? 'active' : ''}" id="${tabId}-tab" data-bs-toggle="tab" 
               href="#${tabId}" role="tab" aria-controls="${tabId}" 
               aria-selected="${isFirst ? 'true' : 'false'}">${geneSet}</a>
        `;
        tabsContainer.appendChild(tabLink);
        
        // 创建标签内容
        const tabContent = document.createElement('div');
        tabContent.className = `tab-pane fade ${isFirst ? 'show active' : ''}`;
        tabContent.id = tabId;
        tabContent.setAttribute('role', 'tabpanel');
        tabContent.setAttribute('aria-labelledby', `${tabId}-tab`);
        
        // 添加表格
        const tableDiv = document.createElement('div');
        tableDiv.innerHTML = `
            <h6>富集通路</h6>
            <div class="table-responsive">
                <table class="table table-sm table-bordered table-hover">
                    <thead>
                        <tr>
                            <th>通路</th>
                            <th>Overlap</th>
                            <th>P值</th>
                            <th>调整后P值</th>
                            <th>富集比</th>
                            <th>基因</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.results[geneSet].map(result => `
                            <tr class="pathway-row">
                                <td>${result.term}</td>
                                <td>${result.overlap}</td>
                                <td>${parseFloat(result.p_value).toExponential(3)}</td>
                                <td>${parseFloat(result.adjusted_p_value).toExponential(3)}</td>
                                <td>${parseFloat(result.odds_ratio).toFixed(2)}</td>
                                <td title="${result.genes}" style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                    ${result.genes}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        
        tabContent.appendChild(tableDiv);
        tabContentContainer.appendChild(tabContent);
        
        isFirst = false;
    }
    
    showResults('enrichment-results');
}

// 执行PPI网络分析
async function performPPINetworkAnalysis() {
    if (currentFeatures.length === 0) {
        showNotification('没有可用的特征进行PPI网络分析', true);
        return;
    }
    
    hideResults('ppi-results');
    showLoading('ppi-loading');
    
    // 获取PPI网络分析参数
    const ppiParams = {
        gene_list: currentFeatures,
        species: parseInt(document.getElementById('ppi-species').value),
        required_score: parseInt(document.getElementById('ppi-score').value),
        network_type: document.getElementById('ppi-network-type').value
    };
    
    try {
        const response = await fetch(`${API_URL}/ppi-network`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(ppiParams),
        });
        
        if (!response.ok) {
            let errorMessage;
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || `HTTP error ${response.status}`;
            } catch (e) {
                errorMessage = await response.text() || `HTTP error ${response.status}`;
            }
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        
        // 显示结果
        displayPPINetworkResults(data);
        showNotification('PPI网络分析成功');
    } catch (error) {
        console.error('Error:', error);
        showNotification(`PPI网络分析失败: ${error.message}`, true);
        
        // 显示一个空的结果区域，告知用户错误
        const ppiResults = document.getElementById('ppi-results');
        ppiResults.classList.remove('d-none');
        ppiResults.innerHTML = `
            <div class="alert alert-danger">
                <h5>PPI网络分析失败</h5>
                <p>${error.message}</p>
                <p>可能的原因：</p>
                <ul>
                    <li>提供的基因列表中没有已知的蛋白质相互作用</li>
                    <li>STRING数据库无法识别提供的基因名</li>
                    <li>网络连接问题</li>
                </ul>
                <p>建议：</p>
                <ul>
                    <li>检查基因名称是否正确</li>
                    <li>尝试降低交互分数阈值</li>
                    <li>尝试使用不同的物种</li>
                </ul>
            </div>
        `;
    } finally {
        hideLoading('ppi-loading');
    }
}

// 显示PPI网络分析结果
function displayPPINetworkResults(data) {
    // 更新节点数和边数
    document.getElementById('ppi-node-count').textContent = data.node_count;
    document.getElementById('ppi-edge-count').textContent = data.edge_count;
    
    // 显示网络图像
    const networkImageContainer = document.getElementById('ppi-network-image');
    networkImageContainer.innerHTML = `
        <img src="data:image/png;base64,${data.network_image}" 
             alt="PPI Network" class="img-fluid network-image">
    `;
    
    showResults('ppi-results');
}
