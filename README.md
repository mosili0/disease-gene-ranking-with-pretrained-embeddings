# Biomedical Foundation Model 小型实践

研究问题：**预训练嵌入在什么情况下有助于疾病相关基因排序，何时简单方法更有效？**

使用冻结的 ESM-2 与 BioBERT，完成 GO 功能相似性、疾病关联补全、药物关联靶点补全/机制文本检索。包含 AAC、TF-IDF、STRING、精确随机期望及固定等权融合对照，不训练或微调大模型。

**这是修订后的评估版本。** 每折留出一个已知正例，用剩余种子对待测正例及全部未标注背景共同评分；种子不参加候选排序，AUC在折内计算后平均。结果不能解释为全基因组筛查、临床致病判断或未知药物靶点发现。

完整结论和实际数字见[实验报告](实验报告.md)，[十分钟汇报提纲](汇报提纲.md)可用于小组分工。各表由结果CSV自动生成，执行 `--steps report` 会更新报告和提纲。

## 1 本机和离线复现

已配置好依赖、权重与修订版缓存的环境：

```text
python main.py --steps all --offline
python -m unittest discover -s tests -v
```

Windows也可运行：

```bat
run.bat --steps all --offline
```

`run.bat` 按顺序使用 `BIOFM_PYTHON` 指定的解释器、项目 `.venv\Scripts\python.exe`、当前PATH里的 `python`。没有绑定个人电脑路径。使用已有conda环境时，先激活环境，或把 `BIOFM_PYTHON` 设为该环境的解释器路径。

`--offline` 禁止数据HTTP请求，模型始终只从本地加载；缺少完整公开缓存时明确报错，不用旧top400数据伪装完整标签。随附小型嵌入缓存可直接复算下游实验；如果输入、模型文件、查询或处理版本改变，缓存签名会失效并需要本地重新推理。

## 2 换机器首次配置

推荐Python **3.10**（本次验证为3.10.21），创建独立环境：

```text
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/bootstrap.py all
.\.venv\Scripts\python.exe main.py --steps all
```

macOS/Linux：

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/bootstrap.py all
.venv/bin/python main.py --steps all
```

下载器只补齐缺失模型文件，使用Hugging Face查询到的固定提交版本下载，并记录文件SHA-256。`models/`和旧机器的`vendor/`不提交到GitHub。标准pip会解析全部依赖，不依赖旧版仅下载三个wheel的方式。若使用已有`vendor/`，入口会优先加载其中的旧机器依赖；换机器应新建环境，不复制这个目录。

`requirements.txt`固定了本次使用的主要数值和NLP包版本；Torch允许兼容稳定版本。本机原环境为Torch开发构建，精确版本保存在`results/environment.json`，不能声称新机器重新推理后逐位一致。权重和输入哈希见`data/emb/v2/summary.json`；使用同一嵌入缓存能复算同一评估。

首次运行需要访问UniProt、Open Targets、ChEMBL、STRING，模型下载需要Hugging Face。网络失败会明确报错，不自动关闭证书校验。缓存齐全后可断网运行。

## 3 分步入口

```text
python main.py --steps data
python main.py --steps embeddings
python main.py --steps tasks --offline
python main.py --steps benchmark --offline
python main.py --steps viz --offline
python main.py --steps report
python main.py --steps data,embeddings,tasks,benchmark,viz,report
```

- `data`：读取/获取UniProt数据，不会附带运行模型。
- `embeddings`：提取并验证嵌入缓存，不会运行评估。
- `tasks`：运行人工演示标签及GO实验，包含999次基因级置换检验。
- `benchmark`：在整个候选池覆盖完整的公开注释上运行相同协议，并报告0.1/0.2/0.3阈值敏感性。
- `viz`：从现有评估结果生成8张图，不重新获取公开标签。
- `report`：从已保存结果生成报告和汇报提纲。

可用 `--force-data` 刷新UniProt、`--force-emb`重新推理、`--force-bench`刷新公开平台；在线刷新选项不能与离线模式混用。公开平台会更新，刷新后数字可能变化，应保留对应manifest。`--device cpu`为默认，`--threads 4`限制CPU线程数。单模型计算可使用 `--models esm --steps embeddings,tasks,benchmark`；完整报告要求两个模型均已计算。

## 4 评估与输入处理

- AUC对同分正例/背景对计0.5，ROC/AP按同分组计算；排名用中秩，Recall@10/MRR按随机打破同分的期望计分。
- 公开疾病数据先逐一解析81个候选基因ID，再用Open Targets `Bs`过滤查询全部候选ID，核对返回count；旧的前400条缓存不被视为完整数据。`GBA`显式映射到`GBA1`。
- 背景是“未标注/低于阈值”，并非实验确认的真阴性。Open Targets association score不是致病概率。
- ESM对全长序列按1022残基分段；BioBERT对清理后的完整功能文本按254个内容token分段。排除特殊token及padding，按内容数量加权汇总，不丢弃长输入尾部，也没有跨窗口注意力。
- TF-IDF仅用同批基因完整文本拟合词表和IDF；AAC使用全长序列。融合是固定等权拼接，没有根据测试标签选权重。
- 原药物机制查询包含靶点/机制提示，只解释为文本检索；新增仅适应证查询去掉药物名、靶点名及直接靶点类别表达。两组文本完整保存，属于审阅后指定的探索性对照。
- GO检验同时置换相似度矩阵两轴的基因身份，保留基因对依赖；5种表示做BH校正。其他任务比较为描述性，不宣称未经检验的“显著优于”。

## 5 文件与可追溯性

```text
main.py / config.py / run.bat / requirements.txt
src/metrics.py             同分处理与排名指标
src/evaluation.py          所有方法共用的逐折评估
src/tasks.py               GO和人工演示标签实验
src/benchmark.py           公开注释、完整候选查询和网络基线
src/baselines.py           AAC、TF-IDF和固定等权融合
src/embeddings.py          全输入覆盖、冻结推理与签名缓存
scripts/bootstrap.py       可迁移模型下载器
scripts/build_report.py    用结果生成报告与汇报提纲
tests/test_evaluation.py    防止评估错误再次发生
data/benchmark/            原始公开响应、ID映射和来源manifest
data/emb/v2/               修订版嵌入、输入签名和权重哈希
results/*.csv              完整指标、阈值敏感性、标签和查询审计
results/*_detail.json       每折种子、候选、得分与名次
results/environment.json   实际运行环境与代码哈希
results/figures/           8张结果图
archive_before_revision_20261103/  本地旧版备份（不提交）
```

人工药物标签实际为21个药物、39条关联；人工疾病标签为小规模演示，缺少逐条文献引证。公开药物记录可能把复合物映射到多个组分；保留这些标记，不把所有组分都称为直接结合靶点。UniProt功能文本与GO/公开关联可能共享来源，公开注释对照不等于完全独立的外部验证。

主要来源：[UniProt](https://www.uniprot.org/help/api)、[Open Targets](https://platform-docs.opentargets.org/data-access/graphql-api)、[ChEMBL](https://www.ebi.ac.uk/chembl/api/data/docs)、[STRING](https://string-db.org/help/api/)、[ESM-2](https://huggingface.co/facebook/esm2_t6_8M_UR50D)、[BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)。

## 6 GitHub提交状态

**尚未发布；当前目录没有配置远程仓库，因此暂无真实仓库链接。** 不填虚构URL。提交成员需在实际课程GitHub仓库上传代码、README、报告、提纲、测试、小型输入/公开标签/嵌入缓存和结果，并把真实仓库链接补入本节及报告。

`.gitignore`已排除模型、vendor、虚拟环境和旧版备份，避免把大文件及本机运行环境上传。上传前在另一环境按第2节验证运行；权重通过来源下载。当前目录中保存的修订材料已可供审阅，发布操作尚未执行。
