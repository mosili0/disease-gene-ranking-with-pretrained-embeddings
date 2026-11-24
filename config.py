# -*- coding: utf-8 -*-
"""
全局配置：路径、预训练模型、基因集合、疾病基因与药物靶点基准数据。

说明
----
* 基因 / 疾病 / 药物的关联基准数据为“小规模人工整理”的数据集：
  - 疾病-基因关联参考 OMIM、ClinGen 与相关综述（常见致病基因）；
  - 药物-靶点关联参考 DrugBank 与药品说明书（主要药理靶点）；
  该基准仅用于课程小型实践的方法演示与评估，不代表完整的临床注释。
* 基因的 GO 注释与文本摘要、蛋白序列在运行时从 UniProt 获取并缓存
  （见 src/data_fetch.py），保证任务输入为真实的公开数据。
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
VENDOR_DIR = os.path.join(ROOT, "vendor")

# ---------------------------------------------------------------------------
# 预训练模型（下载到 models/ 后本地加载，运行时不依赖网络）
# ---------------------------------------------------------------------------
MODEL_IDS = {
    # 蛋白语言模型：ESM-2 (8M 参数)，在 UniRef50 上自监督预训练（Lin et al., Science 2023）
    "esm": "facebook/esm2_t6_8M_UR50D",
    # 生物医学语言模型：BioBERT，在 PubMed/PMC 上预训练（Lee et al., Bioinformatics 2020）
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
}

# ---------------------------------------------------------------------------
# 基因集合（81 个基因，按功能类别分组，用于可视化与结果解读）
# ---------------------------------------------------------------------------
GENE_CATEGORIES = {
    "DNA repair": ["BRCA1", "BRCA2", "ATM", "CHEK2", "PALB2", "RAD51",
                   "BRIP1", "MSH2", "MLH1", "RAD51C", "FANCD2", "TP53"],
    "Cell cycle / apoptosis": ["RB1", "CDKN2A", "CCND1", "CDK4", "CDK6", "MYC",
                               "BCL2", "APC", "CDH1"],
    "RTK signaling": ["EGFR", "ERBB2", "KIT", "PDGFRA", "FLT3", "RET",
                      "ALK", "MET", "FGFR2", "VEGFA", "KDR", "ROS1"],
    "RAS-MAPK-PI3K": ["KRAS", "HRAS", "NRAS", "BRAF", "RAF1", "MAP2K1",
                      "MAPK1", "PIK3CA", "AKT1", "MTOR"],
    "Cytoplasmic kinases": ["ABL1", "SRC", "JAK1", "JAK2", "BTK", "SYK"],
    "Metabolic enzymes": ["HMGCR", "FDPS", "PTGS1", "PTGS2", "VKORC1",
                          "CYP2C9", "CYP2D6", "ACHE", "ATP4A", "SLC5A2"],
    "Neural signaling": ["SLC6A4", "DRD2", "HTR2A", "ADRB2"],
    "Neurodegeneration": ["APP", "PSEN1", "PSEN2", "APOE", "MAPT", "SNCA",
                          "LRRK2", "PRKN", "PINK1", "PARK7", "GBA", "TREM2",
                          "SOD1"],
    "Other targets": ["F10", "MS4A1", "IL6R", "TNF", "F2"],
}

ALL_GENES = [g for genes in GENE_CATEGORIES.values() for g in genes]
GENE_SYMBOL_ALIASES = {"GBA": "GBA1"}  # retain legacy project key; resolve current approved symbol

# ---------------------------------------------------------------------------
# 疾病基因演示标签（混合已知致病/易感/相关基因；其余为未标注背景）
# ---------------------------------------------------------------------------
DISEASES = {
    "breast_cancer": {
        "label": "乳腺癌 (Breast cancer)",
        "genes": ["BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "TP53", "RAD51C", "CDH1"],
        "query": ("hereditary breast and ovarian cancer syndrome DNA damage repair "
                  "homologous recombination tumor suppressor"),
    },
    "colorectal_cancer": {
        "label": "结直肠癌 (Colorectal cancer)",
        "genes": ["APC", "MSH2", "MLH1", "KRAS", "BRAF", "TP53"],
        "query": ("colorectal cancer familial adenomatous polyposis Lynch syndrome "
                  "mismatch repair WNT signaling"),
    },
    "lung_cancer": {
        "label": "肺癌 (Lung cancer)",
        "genes": ["EGFR", "KRAS", "ALK", "MET", "RET", "BRAF", "TP53"],
        "query": ("non-small cell lung cancer lung adenocarcinoma epidermal growth "
                  "factor receptor tyrosine kinase"),
    },
    "parkinson": {
        "label": "帕金森病 (Parkinson disease)",
        "genes": ["SNCA", "LRRK2", "PRKN", "PINK1", "PARK7", "GBA"],
        "query": ("Parkinson disease neurodegeneration dopaminergic neuron "
                  "alpha-synuclein mitochondrial quality control"),
    },
    "alzheimer": {
        "label": "阿尔茨海默病 (Alzheimer disease)",
        "genes": ["APP", "PSEN1", "PSEN2", "APOE", "TREM2", "MAPT"],
        "query": ("Alzheimer disease amyloid beta precursor protein tau "
                  "neurofibrillary tangle neuroinflammation"),
    },
}

# ---------------------------------------------------------------------------
# 药物靶点基准（主要药理靶点；"text query" 用于药物文本查询实验）
# ---------------------------------------------------------------------------
DRUGS = {
    "imatinib": {
        "label": "伊马替尼 (imatinib)",
        "targets": ["ABL1", "KIT", "PDGFRA"],
        "query": ("imatinib tyrosine kinase inhibitor BCR-ABL chronic myeloid "
                  "leukemia gastrointestinal stromal tumor"),
    },
    "gefitinib": {
        "label": "吉非替尼 (gefitinib)",
        "targets": ["EGFR"],
        "query": ("gefitinib epidermal growth factor receptor EGFR tyrosine kinase "
                  "inhibitor non-small cell lung cancer"),
    },
    "trastuzumab": {
        "label": "曲妥珠单抗 (trastuzumab)",
        "targets": ["ERBB2"],
        "query": "trastuzumab HER2 ERBB2 monoclonal antibody breast cancer",
    },
    "venetoclax": {
        "label": "维奈克拉 (venetoclax)",
        "targets": ["BCL2"],
        "query": "venetoclax BCL2 inhibitor apoptosis chronic lymphocytic leukemia",
    },
    "aspirin": {
        "label": "阿司匹林 (aspirin)",
        "targets": ["PTGS1", "PTGS2"],
        "query": ("aspirin acetylsalicylic acid cyclooxygenase COX inhibitor "
                  "antiplatelet analgesic"),
    },
    "atorvastatin": {
        "label": "阿托伐他汀 (atorvastatin)",
        "targets": ["HMGCR"],
        "query": "atorvastatin statin HMG-CoA reductase inhibitor cholesterol lowering",
    },
    "alendronate": {
        "label": "阿仑膦酸钠 (alendronate)",
        "targets": ["FDPS"],
        "query": ("alendronate bisphosphonate farnesyl diphosphate synthase "
                  "inhibitor osteoporosis"),
    },
    "rivaroxaban": {
        "label": "利伐沙班 (rivaroxaban)",
        "targets": ["F10"],
        "query": "rivaroxaban direct factor Xa inhibitor anticoagulant",
    },
    "warfarin": {
        "label": "华法林 (warfarin)",
        "targets": ["VKORC1"],
        "query": "warfarin vitamin K epoxide reductase inhibitor anticoagulant coumarin",
    },
    "omeprazole": {
        "label": "奥美拉唑 (omeprazole)",
        "targets": ["ATP4A"],
        "query": "omeprazole proton pump inhibitor gastric acid secretion peptic ulcer",
    },
    "fluoxetine": {
        "label": "氟西汀 (fluoxetine)",
        "targets": ["SLC6A4"],
        "query": ("fluoxetine selective serotonin reuptake inhibitor SSRI "
                  "antidepressant"),
    },
    "empagliflozin": {
        "label": "恩格列净 (empagliflozin)",
        "targets": ["SLC5A2"],
        "query": ("empagliflozin SGLT2 sodium glucose cotransporter inhibitor "
                  "type 2 diabetes"),
    },
    "donepezil": {
        "label": "多奈哌齐 (donepezil)",
        "targets": ["ACHE"],
        "query": "donepezil acetylcholinesterase inhibitor Alzheimer disease dementia",
    },
    "rituximab": {
        "label": "利妥昔单抗 (rituximab)",
        "targets": ["MS4A1"],
        "query": "rituximab anti-CD20 monoclonal antibody B cell lymphoma",
    },
    "bevacizumab": {
        "label": "贝伐珠单抗 (bevacizumab)",
        "targets": ["VEGFA"],
        "query": "bevacizumab anti-VEGF monoclonal antibody angiogenesis inhibitor",
    },
    "haloperidol": {
        "label": "氟哌啶醇 (haloperidol)",
        "targets": ["DRD2"],
        "query": "haloperidol antipsychotic dopamine D2 receptor antagonist schizophrenia",
    },
    "dasatinib": {
        "label": "达沙替尼 (dasatinib)",
        "targets": ["ABL1", "SRC", "KIT", "PDGFRA"],
        "query": ("dasatinib multi-targeted BCR-ABL SRC family kinase inhibitor "
                  "chronic myeloid leukemia"),
    },
    "sorafenib": {
        "label": "索拉非尼 (sorafenib)",
        "targets": ["BRAF", "RAF1", "KIT", "PDGFRA", "FLT3", "KDR"],
        "query": ("sorafenib multi-kinase inhibitor RAF VEGFR PDGFR hepatocellular "
                  "carcinoma renal cell carcinoma"),
    },
    "crizotinib": {
        "label": "克唑替尼 (crizotinib)",
        "targets": ["ALK", "MET", "ROS1"],
        "query": ("crizotinib ALK MET ROS1 tyrosine kinase inhibitor non-small cell "
                  "lung cancer"),
    },
    "palbociclib": {
        "label": "帕博西尼 (palbociclib)",
        "targets": ["CDK4", "CDK6"],
        "query": ("palbociclib CDK4 CDK6 cyclin dependent kinase inhibitor breast cancer"),
    },
    "sunitinib": {
        "label": "舒尼替尼 (sunitinib)",
        "targets": ["KIT", "PDGFRA", "FLT3", "KDR", "RET"],
        "query": ("sunitinib multi-targeted receptor tyrosine kinase inhibitor VEGFR "
                  "PDGFR renal cell carcinoma"),
    },
}

# ---------------------------------------------------------------------------
# 运行参数
# ---------------------------------------------------------------------------
EMBED_MODELS = ("esm", "biobert")          # 参与实验的预训练模型
EMBED_DIM = {"esm": 320, "biobert": 768}   # 各自的嵌入维度
RANDOM_SEED = 2024

# Indication-only controls intentionally omit drug names, target names/aliases,
# and direct target-class descriptions. They still contain disease knowledge:
# these manually specified, post-review controls are exploratory, not blinded.
DRUG_INDICATION_QUERIES = {
    "imatinib": "chronic myeloid leukemia gastrointestinal stromal tumor treatment",
    "gefitinib": "non-small cell lung cancer treatment",
    "trastuzumab": "breast cancer gastric cancer treatment",
    "venetoclax": "chronic lymphocytic leukemia acute myeloid leukemia treatment",
    "aspirin": "pain fever inflammation prevention of arterial thrombosis",
    "atorvastatin": "hypercholesterolemia prevention of cardiovascular disease",
    "alendronate": "osteoporosis prevention of bone fractures",
    "rivaroxaban": "prevention of stroke and systemic embolism venous thrombosis",
    "warfarin": "prevention of stroke and systemic embolism venous thrombosis",
    "omeprazole": "gastroesophageal reflux disease peptic ulcer treatment",
    "fluoxetine": "major depressive disorder obsessive compulsive disorder treatment",
    "empagliflozin": "type 2 diabetes mellitus heart failure treatment",
    "donepezil": "Alzheimer disease dementia symptomatic treatment",
    "rituximab": "B cell non-Hodgkin lymphoma rheumatoid arthritis treatment",
    "bevacizumab": "colorectal cancer lung cancer treatment",
    "haloperidol": "schizophrenia psychosis treatment",
    "dasatinib": "chronic myeloid leukemia acute lymphoblastic leukemia treatment",
    "sorafenib": "hepatocellular carcinoma renal cell carcinoma treatment",
    "crizotinib": "non-small cell lung cancer treatment",
    "palbociclib": "advanced or metastatic breast cancer treatment",
    "sunitinib": "renal cell carcinoma gastrointestinal stromal tumor treatment",
}


def all_queries():
    out = {"disease_" + k: d["query"] for k, d in DISEASES.items()}
    out.update({"drug_" + k: d["query"] for k, d in DRUGS.items()})
    out.update({"drug_indication_" + k: q for k, q in DRUG_INDICATION_QUERIES.items()})
    return out


PERMUTATIONS = 999
OT_THRESHOLDS = (0.1, 0.2, 0.3)
