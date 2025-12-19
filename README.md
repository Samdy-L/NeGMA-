# 社区级虚假信息检测

本项目实现了一个**基于动态社区的虚假信息检测框架**，旨在识别社交网络中的虚假信息传播团伙。该框架将检测视角从微观的节点级提升至中观的社区级，并采用**两阶段策略**（规则预过滤 + 逻辑回归），以确保模型的高可解释性和准确性。

## 📂 项目结构

```text
experiment4_project/
├── data/                   # 数据目录（存储生成的实验结果）
├── plots/                  # 可视化输出（ROC曲线、特征重要性等）
├── src/                    # 源代码
│   ├── features.py         # 特征提取（时序特征与结构特征）
│   ├── filters.py          # 第一阶段：规则预过滤
│   ├── model.py            # 第二阶段：逻辑回归分类器
│   ├── fusion.py           # 两阶段结果融合逻辑
│   ├── evaluation.py       # 后验验证与指标计算
│   ├── visualization.py    # 绘图脚本
│   └── utils.py            # 数据生成与加载工具
├── main.py                 # 实验主程序入口
└── requirements.txt        # Python 依赖库
```

## 🚀 快速开始

### 1. 环境配置

请确保已安装 Python 3.8+。

```bash
# 创建虚拟环境（可选但推荐）
python -m venv venv
# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行实验

执行主脚本以运行完整的实验流程（数据生成 -> 特征提取 -> 检测 -> 评估）：

```bash
python main.py
```

脚本将执行以下操作：
1.  生成模拟动态网络数据（10个快照，200个节点）。
2.  提取8个核心社区特征（包含形成速度与稳定性）。
3.  应用两阶段检测策略。
4.  执行后验验证。
5.  在控制台输出评估指标（Accuracy, F1, AUC）。
6.  将详细结果保存至 `data/experiment_results.csv`。

### 3. 生成可视化图表

生成学术级图表（ROC曲线、特征重要性、社区演化轨迹、雷达图）：

```bash
python src/visualization.py
```

生成的图片将保存在 `plots/` 目录下。

## 🧠 方法论

### 核心特征
模型使用了8个关键特征，重点关注时间演化：
*   **动态特征**：形成速度 (Formation Speed)、成员稳定性 (Member Stability)。
*   **结构特征**：内部密度 (Internal Density)、隔离度 (Isolation Degree)、平均内部度 (Avg Internal Degree)。
*   **统计特征**：社区规模 (Community Size)、权重和 (Weight Sum)、权重方差 (Weight Variance)。

### 两阶段策略
1.  **第一阶段（基于规则）**：使用硬性阈值 ($D>0.7, V \le 2, I<0.3$) 快速筛选高风险社区。
2.  **第二阶段（基于模型）**：使用带 5 折交叉验证的逻辑回归进行精细分类。
3.  **融合逻辑**：规则判定的结果具有更高优先级。

### 后验验证
对检测出的虚假社区，提取其内部连接强度前 10% 的核心节点，检查其历史行为以强化检测的可信度。

## 📊 评估指标

*   **社区发现层面**：模块度 (Modularity)、NMI、时间平滑度 (Temporal Smoothness)。
*   **虚假检测层面**：准确率 (Accuracy)、精确率 (Precision)、召回率 (Recall)、F1-Score、AUC-ROC。

## 📝 说明

本项目属于“社会计算”课程实验内容。
