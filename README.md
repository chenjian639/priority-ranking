# TESS TOI 数据预处理与候选优先级评分

把 NASA TOI 原始表处理为可直接用于机器学习的建模表，并实现误报风险预测与候选优先级评分管线。文档简要说明目录结构、输入/输出、模型与如何复现。

## 快速开始（Quick start）

最小复现步骤（Windows）：

1. 创建并激活虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. 安装依赖：

```powershell
pip install -r requirements.txt
```

3. 运行脚本（按顺序）：

```powershell
python code/preprocess_toi_modeling_data.py --input raw/NASA_TESS_TOI_2026-05-28_FULL.csv --output-dir .
python code/analyze_priority.py
python code/compute_priority.py
```

说明：`preprocess_toi_modeling_data.py` 在需要从原始 CSV 重新生成 `processed/` 时运行；若仓库已有 `processed/`，可直接运行后两个脚本。


## 1. 快速文件结构

```text
TESS_TOI_preprocessing_package/
├─ raw/
│  └─ NASA_TESS_TOI_2026-05-28_FULL.csv
├─ processed/
│  ├─ selected_columns_all_toi.csv
│  ├─ train_modeling_table_cp_kp_vs_fp_fa.csv
│  ├─ candidate_pc_apc_for_risk_prediction.csv
│  ├─ excluded_other_labels.csv
│  ├─ preprocessing_summary.json
│  └─ priority_outputs/
│     ├─ candidates_with_scores.csv
│     ├─ top20_priority.csv
│     ├─ top20_high_risk.csv
│     ├─ top20_breakdown.csv
│     ├─ cv_metrics.txt
│     └─ *.png
├─ code/
│  ├─ preprocess_toi_modeling_data.py
│  ├─ analyze_priority.py
│  └─ compute_priority.py
└─ README.md
```

## 2. 数据与预处理摘要

- 原始数据：`raw/NASA_TESS_TOI_2026-05-28_FULL.csv`（7,931 行，91 列）。
- 预处理脚本：`code/preprocess_toi_modeling_data.py`，会生成 `processed/` 下的表格和 `preprocessing_summary.json`。
- 保留特征（用于建模）：`pl_orbper, pl_rade, st_tmag, st_teff, st_rad`；计算 `missing_feature_count` 与 `missing_feature_names`。
- 标签处理：`CP`/`KP` 作为正类（1），`FP`/`FA` 作为负类（0）；`PC`/`APC` 为待预测候选，不纳入训练集。
- 预处理结果摘要（见 `processed/preprocessing_summary.json`）：训练表 2666 行，候选（PC/APC）5262 行。

## 3. 核心输出文件（说明）

- `processed/train_modeling_table_cp_kp_vs_fp_fa.csv`：训练表（CP/KP vs FP/FA），含目标 `target_confirmed_planet`。
- `processed/candidate_pc_apc_for_risk_prediction.csv`：PC/APC 候选表，留空的目标用于预测。
- `processed/selected_columns_all_toi.csv`：所有 TOI 的精选字段，便于 EDA。
- `processed/excluded_other_labels.csv`：未分类或缺标签记录。
- `processed/preprocessing_summary.json`：机器可读的处理摘要。
- `processed/priority_outputs/`：包含打分后的候选表、Top20 输出、CV 指标与可视化图像。

### 输出表格说明（详细）

- `processed/priority_outputs/candidates_with_scores.csv`：对所有 PC/APC 候选的完整打分表。包含原始特征、`planet_probability`（模型概率）、`science_interest`、`priority_score`（或 `priority_score_calc`，最终惩罚后分数）及 `missing_feature_count` 等。通常按 `priority_score` 降序筛选观测目标。

- `processed/priority_outputs/top20_priority.csv`：按最终优先级（惩罚后分数）选出的前 20 名候选，用于直接生成观测清单。包含标识字段、`planet_probability`、`priority_score` 及关键观测参数。

- `processed/priority_outputs/top20_breakdown.csv`：Top20 的分解表（由 `analyze_priority.py` 生成），每一行包含分项贡献：`contrib_planet`、`contrib_science`、`contrib_brightness`、`contrib_period`，以及 `missing_penalty`、`priority_score_before_penalty`（惩罚前分数）与 `priority_score_calc`（惩罚后分数）。适合用于可视化和分析惩罚影响。

- `processed/priority_outputs/top20_high_risk.csv`：模型认为概率最低的 20 个候选（按 `planet_probability` 升序），用于人工复核可能的误报或数据缺失问题，通常会包含 `missing_feature_names` 以便快速定位问题字段。


## 4. 模型与优先级评分（实现要点）

- 使用特征：`pl_orbper, pl_rade, st_tmag, st_teff, st_rad, missing_feature_count`。
- 预处理：对这些特征使用 `SimpleImputer(strategy='median')` 填充缺失值，然后用 `StandardScaler` 标准化。
- 基线模型：`RandomForestClassifier(n_estimators=200, random_state=42)`；训练脚本位于 `code/analyze_priority.py` 和 `code/compute_priority.py`。
- 行星概率：模型的正类概率由 `predict_proba(... )[:, 1]` 写入字段 `planet_probability`。

- 优先级评分（默认合成策略）：

  priority_score = 0.50 × planet_probability
                 + 0.20 × science_interest
                 + 0.15 × brightness_score
                 + 0.10 × short_period_score
                 - missing_penalty

  说明：
  - `science_interest`：基于与地球的相似性计算得到，具体用半径相对于 1 R_earth 和估算入射通量相对于 1 Earth flux 的对数空间高斯核（见 `code/compute_priority.py` 中 `similarity_to_one` / `similarity` 实现）分别计算半径相似度和入射通量相似度，再按权重合成为科学兴趣分数（代码中默认权重为 0.6×radius_similarity + 0.4×insolation_similarity；核宽度默认 radius sigma=0.25，insolation sigma=0.5）。说明：更接近地球半径与地球入射通量的候选会获得更高的 `science_interest`，便于优先考虑潜在类地或位于宜居带附近的候选。
  - `brightness_score`：由 `st_tmag` 线性归一化得到（星等越小越高分）。
    - 解释：TESS 星等越小表示恒星越亮，观测时信噪比更高，后续地基随访（地面望远镜或高精度光度/径向速度）更容易得到高质量数据，因此优先级更高。
  - `short_period_score`：由 `pl_orbper` 线性归一化得到（周期越短越高分）。
    - 解释：轨道周期短的候选在更短的时间内重复出现凌日事件，更容易在有限的观测窗口内捕捉到后续凌日，有助于快速确认或排除候选；短周期对象对观测安排和资源利用更友好，也能更快地改进轨道精度（减少观测等待时间）。
  - `missing_penalty`：0.05 × (missing_feature_count / max_missing) 用于降低缺失过多样本的优先级。

实现差异说明：
- 当前代码实现把 `science_interest` 纳入了合成，并采用了不同的权重（默认在 `code/compute_priority.py` 中为 planet 0.5、science 0.2、brightness 0.15、period 0.1；`code/analyze_priority.py` 中也使用了类似的权重分解用于可视化）。

如何修改权重：

- 在 `code/compute_priority.py` 中，优先级合成在 `compute_scores` 函数和 `priority_score = ...` 处定义，直接修改相应系数即可。
- 在 `code/analyze_priority.py` 中，用于绘图的贡献分解在 `w = {'planet':0.5, 'science':0.2, 'brightness':0.15, 'period':0.1}`（可按需修改以保持分析一致）。

## 5. 评估与可视化

- 在 `code/analyze_priority.py` 中实现了 5 折分层交叉验证（StratifiedKFold），并输出：AUC、Brier score、Accuracy 到 `processed/priority_outputs/cv_metrics.txt`。
- 生成校准曲线、Top20 贡献图（分项堆积）和半径-入射通量散点图，保存在 `processed/priority_outputs/`。

## 6. 如何复现（快速命令）

在项目根目录运行：

```bash
python code/preprocess_toi_modeling_data.py --input raw/NASA_TESS_TOI_2026-05-28_FULL.csv --output-dir .
python code/analyze_priority.py
python code/compute_priority.py
```

说明：`analyze_priority.py` 会训练并保存模型（`processed/planet_rf_model.joblib`）并生成 CV 指标与图像；`compute_priority.py` 会对 PC/APC 生成 `planet_probability` 与 `priority_score` 并导出 Top20 表。

## 7. 注意事项与建议

- 标识字段（`toi`、`tid`、`toidisplay`）不应作为模型特征。
- 避免把 `rowupdate`、`release_date` 作为直接特征，除非明确需要时间信息并注意数据泄漏。
- 预处理阶段未执行插补/标准化，相关操作在训练 pipeline 中完成以避免数据泄漏。

建议的后续工作：使用 XGBoost/LightGBM 做模型对比、加入 SHAP 分析提升解释性、将 `priority_score` 的权重参数化并做权重扫描以便决策支持。

---


**图表解读**

- **校准曲线（Calibration curve）**: 见 [processed/priority_outputs/calibration_curve.png](processed/priority_outputs/calibration_curve.png)。曲线总体靠近对角线，说明模型输出的 `planet_probability` 与观测到的正类频率相对一致；在极低概率区间略显保守（预测概率稍高于观测频率），在中等概率区间存在小幅偏差。总体校准可接受，但若需要严格概率解释（例如用于贝叶斯决策或阈值敏感的调度），建议在部署前使用 `CalibratedClassifierCV` 或 Isotonic/Platt 标定进行微调。

- **半径 vs 入射通量散点（Radius vs Insolation）**: 见 [processed/priority_outputs/radius_insolation_scatter.png](processed/priority_outputs/radius_insolation_scatter.png)。横轴为行星半径（R_earth，对数尺度），纵轴为入射通量（Earth flux，对数尺度），点颜色表示 `planet_probability`。图中高概率（黄色）点集中在小到中等半径且入射通量为弱到中等值的区域，说明模型倾向于把较接近地球半径且入射通量适中的候选判为更可能是真行星；超大半径或极端入射通量的样本概率普遍较低。这与训练样本中类地/小型候选更具典型特征一致，也支持 `science_interest` 将地球相似性作为加分项的设计决策。

- **Top20 权重分解（Top20 contributions）**: 见 [processed/priority_outputs/top20_contributions.png](processed/priority_outputs/top20_contributions.png)。堆积条形图展示了前 20 名候选的 `priority_score` 分解：蓝色（`planet_probability`）为主导贡献，橙色为 `science_interest`，绿色为 `brightness`，红色为 `period`；红点表示合成前的分数（未减去缺失惩罚）。结论：优先级主要由模型概率驱动（即置信度高的候选优先），`science_interest` 经常是次要但稳定的加分来源，`brightness` 和 `period` 提供辅助调整。若希望提升科学价值导向，可在 `code/compute_priority.py` 中提高 `science_interest` 的系数，并重新运行管线以观察排序变化。



