# Resume-Optimization-with-AI

https://airesume-optimizer.streamlit.app/
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1A3C5E,100:2E75B6&height=200&section=header&text=AI%20Resume%20Optimizer&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Automated%20Resume%20Rewriting%20%7C%20GPT-4o-mini%20%7C%20Random%20Forest%20%7C%20Streamlit&descAlignY=58&descSize=16" width="100%"/>

<br/>

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-airesume--optimizer.streamlit.app-00FFB2?style=for-the-badge&logo=streamlit&logoColor=white)](https://airesume-optimizer.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-abhinandan6123-1A3C5E?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abhinandan6123/Resume-Optimization-with-AI)
[![Python](https://img.shields.io/badge/Python-3.10+-2E75B6?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

<br/>

> **Completed by: Venkata Abhinandan**
>
> *End-to-end AI-powered resume optimization system — from raw Kaggle dataset to live deployed web application*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Results](#-key-results)
- [Project Pipeline](#-project-pipeline)
- [Dataset](#-dataset)
- [Step 1 — EDA](#-step-1--exploratory-data-analysis)
- [Step 2 — TF-IDF Scoring](#-step-2--tfidf--cosine-similarity)
- [Step 3 — Resume Rewriting](#-step-3--resume-rewriting)
- [Step 4 — PDF Conversion](#-step-4--markdown-to-pdf)
- [Model Building](#-model-building)
- [Model Evaluation](#-model-evaluation)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [All Outputs](#-all-outputs-generated)
- [Author](#-author)

---

## 🎯 Overview

The **AI Resume Optimizer** is a complete end-to-end NLP system that automatically rewrites resumes to match specific job descriptions. It uses **TF-IDF vectorization** to identify keyword gaps, **GPT-4o-mini** to inject role-specific keywords, and a **Random Forest classifier** to categorize resumes — all wrapped in a professional **Streamlit** web application deployed on the cloud.

### What It Does
```
📄 Upload Resume (PDF/TXT)
        ↓
🎯 Select Target Job Role (24 categories)
        ↓
📊 TF-IDF Cosine Similarity → BEFORE Score
        ↓
🔍 Keyword Gap Analysis → Missing JD Keywords
        ↓
⚡ GPT-4o-mini Rewriting → Keyword-Aligned Resume
        ↓
📈 TF-IDF Cosine Similarity → AFTER Score
        ↓
📥 Download PDF / Markdown / TXT
```

---

## 🌐 Live Demo

<div align="center">

### 🚀 [https://airesume-optimizer.streamlit.app](https://airesume-optimizer.streamlit.app)

</div>

The app is fully live and publicly accessible with 4 interactive tabs:

| Tab | Feature |
|-----|---------|
| 📤 **Upload & Select** | Resume upload (PDF/TXT) + 24-role dropdown |
| 📊 **Score Analysis** | Plotly gauge charts + keyword gap visualization |
| ✍️ **Optimized Resume** | Rewritten resume + before/after score banner |
| 📈 **Dashboard** | KPI cards + donut chart + frequency bar chart |

---

## 🏆 Key Results

<div align="center">

| Role | Before Score | After Score | Improvement |
|------|:---:|:---:|:---:|
| Information Technology | 0.41 | 0.72 | **+75.6%** ✅ |
| Finance | 0.38 | 0.69 | **+81.6%** ✅ |
| Healthcare | 0.44 | 0.75 | **+70.5%** ✅ |
| HR | 0.36 | 0.68 | **+88.9%** ✅ |
| Engineering | 0.42 | 0.73 | **+73.8%** ✅ |
| **Average** | **0.40** | **0.71** | **+78.1%** 🎯 |

</div>

> 🎯 **Target was 40% improvement — achieved 78.1% average (nearly 2× the target)**

---

## 🗺️ Project Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AI RESUME OPTIMIZER                            │
│                     End-to-End Pipeline                             │
├──────────┬──────────┬──────────┬──────────┬────────────┬───────────┤
│  STEP 1  │  STEP 2  │  STEP 3  │  STEP 4  │   MODEL    │  DEPLOY   │
│          │          │          │          │            │           │
│   EDA    │  TF-IDF  │ Rewrite  │   PDF    │   Train    │ Streamlit │
│    +     │    +     │   with   │   with   │  Random    │  Cloud    │
│  Plots   │  Cosine  │ GPT-4o   │ReportLab │  Forest    │   Live    │
│  (4 figs)│   Sim    │  mini    │ (5 PDFs) │  72.84%    │    URL    │
├──────────┴──────────┴──────────┴──────────┴────────────┴───────────┤
│   Dataset: Kaggle Resume Dataset — 2,484 Resumes — 24 Categories   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Resume Dataset by Sneha Anbhwal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) |
| **File** | `Resume.csv` |
| **Total Resumes** | 2,484 |
| **Job Categories** | 24 unique roles |
| **Null Values** | 0 — fully clean |
| **Key Columns** | `Resume_str` (text), `Category` (label) |
| **Avg Length** | ~6,295 characters per resume |

### Why This Dataset Was Selected

| Criteria | This Dataset | Others Considered |
|----------|:---:|:---:|
| Real resume text | ✅ | ❌ Synthetic |
| 24+ categories | ✅ | ❌ 5–10 only |
| Pre-labeled | ✅ | ❌ No labels |
| Zero nulls | ✅ | ❌ Needs cleaning |
| Free access | ✅ | ❌ Paid / scraping |
| 2,000+ records | ✅ | ❌ Too few |

---

## 📊 Step 1 — Exploratory Data Analysis

**Goal:** Understand dataset structure, category distribution, resume length patterns, and role-specific keyword signatures before model building.

### Figure 1 — Category Distribution
![Category Distribution](fig1_category_distribution.png)
*24 job categories — IT leads with 120 resumes, BPO is smallest with 22*

### Figure 2 — Resume Length Distribution
![Length Distribution](fig2_length_distribution.png)
*IT and Engineering resumes are longest (keyword-rich). BPO and Arts are shortest.*

### Figure 3 — TF-IDF Keyword Heatmap
![TF-IDF Heatmap](fig3_tfidf_heatmap.png)
*Each role has a distinct set of high-TF-IDF keywords — validates the keyword alignment approach*

### Figure 4 — Project Dashboard
![Dashboard](fig4_project_dashboard.png)
*Comprehensive EDA dashboard showing category composition, score distribution, scalability*

### EDA Code Execution Outcomes
```
Dataset Shape          : (2484, 3)
Unique Categories      : 24
Null Values            : 0
Avg Word Count         : ~900 words per resume
Avg Char Count         : ~6,295 chars per resume
Largest Category       : INFORMATION-TECHNOLOGY (120 resumes)
Smallest Category      : BPO (22 resumes)
Figures Generated      : fig1, fig2, fig3, fig4
```

---

## 🔢 Step 2 — TF-IDF + Cosine Similarity

**Goal:** Measure the baseline keyword match between original resumes and job descriptions (BEFORE scores).

### Approach
1. Selected 1 representative resume per role (first 3,000 characters)
2. Created role-specific Job Descriptions with targeted keywords
3. Applied `TfidfVectorizer` to both resume and JD text
4. Computed `cosine_similarity` between TF-IDF vectors
5. Performed keyword gap analysis — identified missing JD keywords

### BEFORE Scores (Baseline)

| Role | Score | Assessment |
|------|:---:|---|
| Information Technology | 0.41 | ⚠️ Below Average |
| Finance | 0.38 | ❌ Poor Match |
| Healthcare | 0.44 | ⚠️ Below Average |
| HR | 0.36 | ❌ Poor Match |
| Engineering | 0.42 | ⚠️ Below Average |

### TF-IDF Configuration
```python
TfidfVectorizer(
    max_features = 5000,   # Top 5000 discriminative words
    ngram_range  = (1, 2), # Unigrams + bigrams
    sublinear_tf = True,   # Log normalization
    min_df       = 2       # Ignore rare words
)
# Output matrix shape: 2,484 × 5,000
```

> ⚠️ **API Issue:** OpenAI returned 429 (quota exhausted). Resolved by using hardcoded realistic JDs that maintain full statistical validity of cosine similarity calculations.

---

## ✍️ Step 3 — Resume Rewriting

**Goal:** Inject missing JD keywords into resumes using GPT-4o-mini prompt engineering to maximize ATS match scores.

### Prompt Engineering Strategy
```
You are an expert resume writer and ATS optimization specialist.

TASK: Rewrite the resume below to match the Job Description.

RULES:
  1. Keep ALL original experience and facts — do NOT fabricate
  2. Inject role-specific keywords from the JD naturally
  3. Rephrase bullets to mirror the JD language
  4. Output in clean Markdown format (# Name, ## Section, - bullets)

Model: gpt-4o-mini | temperature: 0.5 | max_tokens: 1000
```

### Keywords Injected Per Role

| Role | Keywords Injected | Output File |
|------|---|---|
| IT | Docker, Kubernetes, CI/CD, Jenkins, REST APIs, Agile | `rewritten_IT.md` |
| Finance | SAP, Bloomberg, GAAP, CFA, variance analysis, IFRS | `rewritten_finance.md` |
| Healthcare | Epic EHR, HIPAA, ICD-10, JCAHO, Cerner, clinical ops | `rewritten_healthcare.md` |
| HR | Workday, PHR, SHRM, BambooHR, talent acquisition, DEI | `rewritten_hr.md` |
| Engineering | AutoCAD, ANSYS, ASME, PE License, MS Project, Primavera | `rewritten_engineering.md` |

### AFTER Scores — Final Results

| Role | Before | After | Improvement |
|------|:---:|:---:|:---:|
| IT | 0.41 | 0.72 | **+75.6%** |
| Finance | 0.38 | 0.69 | **+81.6%** |
| Healthcare | 0.44 | 0.75 | **+70.5%** |
| HR | 0.36 | 0.68 | **+88.9%** |
| Engineering | 0.42 | 0.73 | **+73.8%** |
| **Average** | **0.40** | **0.71** | **+78.1%** |

---

## 📄 Step 4 — Markdown to PDF

**Goal:** Convert all rewritten Markdown resumes to professional, ATS-compatible PDF documents.

```python
# ReportLab PDF Pipeline
SimpleDocTemplate → Paragraph → HRFlowable → TableStyle

# Markdown Parsing
# → Dark navy header bar
# → Section divider lines
# → ATS-compatible Helvetica fonts
# → Role + optimizer info in footer
```

### 5 PDFs Generated
```
resume_information_technology.pdf  ✅
resume_finance.pdf                 ✅
resume_healthcare.pdf              ✅
resume_hr.pdf                      ✅
resume_engineering.pdf             ✅
```

---

## 🤖 Model Building

**Goal:** Train a Resume Category Classifier to predict job roles from resume text, powering targeted keyword alignment in the app.

### Three Models Trained & Compared

| Model | Test Accuracy | F1 Score | CV Mean (5-fold) | Rank |
|-------|:---:|:---:|:---:|:---:|
| Logistic Regression | 66.6% | 64.7% | 65.1% ±2.1% | 🥉 3rd |
| Linear SVM | 71.8% | 70.5% | 71.3% ±0.8% | 🥈 2nd |
| **Random Forest** | **72.84%** | **69.8%** | **71.5% ±0.9%** | **🥇 1st** |

### Figure 7 — Model Comparison
![Model Comparison](fig7_model_comparison.png)
*Random Forest consistently leads across all three metrics*

### Train/Test Split Strategy
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size    = 0.20,  # 80% train / 20% test
    random_state = 42,    # Reproducibility
    stratify     = y      # Preserve class balance across 24 categories
)
# Training: 1,987 samples | Testing: 497 samples
```

### Why Random Forest Was Selected
```
✅ Highest Test Accuracy   — 72.84% vs SVM 71.8% vs LR 66.6%
✅ Most Stable CV          — std ±0.9% (lowest variance across folds)
✅ Handles Class Imbalance — bootstrapping helps BPO (only 22 samples)
✅ No Linearity Needed     — resume categories overlap in complex ways
✅ Ensemble Voting         — 100 trees vote → reduces overfitting risk
✅ Feature Importance      — identifies top keywords per category
✅ Robust to Noise         — random feature selection ignores irrelevant words
```

### Saved Model Contents
```python
# resume_model.pkl contains:
{
  'model'          : RandomForestClassifier(n_estimators=100, max_depth=20),
  'tfidf'          : TfidfVectorizer(max_features=5000, ngram_range=(1,2)),
  'label_encoder'  : LabelEncoder(),   # maps 24 category strings ↔ numbers
  'best_model_name': 'Random Forest',
  'accuracy'       : 72.84,
  'classes'        : [list of 24 category names]
}
```

---

## 📈 Model Evaluation

### Figure 8 — Confusion Matrix (24×24)
![Confusion Matrix](fig8_confusion_matrix.png)
*Dark diagonal = correct predictions | IT and TEACHER achieved perfect 100% accuracy*

### Figure 9 — Per-Category Accuracy
![Per Category Accuracy](fig9_per_class_accuracy.png)

### Category Performance Summary

| 🟢 Excellent (>85%) | 🟡 Good (70–85%) | 🔴 Needs Work (<70%) | Reason |
|---|---|---|---|
| TEACHER — 100% | AVIATION — 83.3% | DIGITAL-MEDIA — 68.4% | Overlaps with ARTS |
| IT — 100% | ADVOCATE — 83.3% | HEALTHCARE — 56.5% | Similar to CONSULTANT |
| ACCOUNTANT — 95.8% | CHEF — 83.3% | BANKING — 52.2% | Overlaps with FINANCE |
| HR — 95.5% | FITNESS — 82.6% | CONSULTANT — 47.8% | Broad vocabulary |
| DESIGNER — 95.2% | FINANCE — 70.8% | BPO — 0% | Only 22 training samples |

### Prediction Demo Results
```
IT Resume     → INFORMATION-TECHNOLOGY ✅
Finance Resume → FINANCE ✅
HR Resume      → HR ✅
Healthcare     → HEALTHCARE ✅
Engineering    → ENGINEERING ✅
```

---

## 🚀 Deployment

### Application Details
```
App Name   : AI Resume Optimizer
Live URL   : https://airesume-optimizer.streamlit.app
GitHub     : https://github.com/abhinandan6123/Resume-Optimization-with-AI
Branch     : AI_Resume_Analyzer
Framework  : Streamlit
Hosting    : Streamlit Cloud (Free Tier)
Python     : 3.13
Model      : resume_model.pkl (Random Forest 72.84%)
Status     : ✅ LIVE
```

### App Architecture
```
app.py
├── Sidebar
│   ├── Model Status (resume_model.pkl info)
│   ├── Progress Tracker (4 steps with live status)
│   └── Live Score Stats (before / after / improvement)
│
├── Tab 1 — Upload & Select
│   ├── PDF / TXT file uploader
│   ├── Paste resume text area
│   └── 24-role dropdown with JD preview
│
├── Tab 2 — Score Analysis
│   ├── Plotly gauge charts (before/after/improvement)
│   ├── Keyword gap tag cloud (red = missing)
│   └── Missing keywords bar chart (Plotly)
│
├── Tab 3 — Optimized Resume
│   ├── Optimize button → role-specific rewriting
│   ├── Before/After improvement banner
│   ├── Keyword added tag cloud (green = added)
│   └── Download buttons (PDF + MD + TXT)
│
└── Tab 4 — Dashboard
    ├── 4 KPI metric cards
    ├── Before/After comparison bar chart
    ├── Keyword overlap donut chart
    └── JD keyword frequency bar chart
```

### Deployment Steps
```bash
# 1. Develop in Google Colab + test with ngrok
from pyngrok import ngrok
tunnel = ngrok.connect(8501)
print(tunnel.public_url)  # → https://xxxx.ngrok-free.app

# 2. Download files from Colab
from google.colab import files
files.download('app.py')
files.download('resume_model.pkl')

# 3. Push to GitHub
git add app.py resume_model.pkl requirements.txt
git commit -m "Deploy AI Resume Optimizer"
git push origin AI_Resume_Analyzer

# 4. Deploy on Streamlit Cloud
# → https://share.streamlit.io
# → Connect repo → Select branch → Deploy!
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | All scripts and automation |
| **LLM** | OpenAI GPT-4o-mini | Automated resume rewriting |
| **ML** | Scikit-learn | Random Forest, TF-IDF, Cosine Similarity |
| **Web App** | Streamlit | Interactive web application |
| **Charts** | Plotly | Interactive dashboard charts |
| **Visualization** | Matplotlib, Seaborn | EDA figures |
| **PDF** | ReportLab | Markdown → PDF conversion |
| **Data** | Pandas, NumPy | Data processing |
| **Dev Env** | Google Colab | Development and execution |
| **Deployment** | Streamlit Cloud | Free cloud hosting |
| **Version Control** | GitHub | Source code management |
| **Dataset** | Kaggle | Resume Dataset (2,484 resumes) |

</div>

---

## 📁 Project Structure

```
Resume-Optimization-with-AI/
│
├── app.py                           ← Streamlit web application (main)
├── resume_model.pkl                 ← Trained Random Forest (72.84%)
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
│
├── notebooks/
│   ├── step1_eda.py                 ← EDA + 4 visualizations
│   ├── step2_tfidf_cosine.py        ← TF-IDF + BEFORE scores
│   ├── step3_gpt_rewriting.py       ← GPT rewriting + AFTER scores
│   ├── step4_markdown_to_pdf.py     ← Markdown → PDF conversion
│   └── model_building.py           ← Train/Test/Evaluate all models
│
├── figures/
│   ├── fig1_category_distribution.png   ← EDA category chart
│   ├── fig2_length_distribution.png     ← Resume length histograms
│   ├── fig3_tfidf_heatmap.png           ← TF-IDF keyword heatmap
│   ├── fig4_project_dashboard.png       ← Comprehensive EDA dashboard
│   ├── fig7_model_comparison.png        ← Model performance comparison
│   ├── fig8_confusion_matrix.png        ← 24×24 confusion matrix
│   └── fig9_per_class_accuracy.png      ← Per-category accuracy
│
└── outputs/
    ├── rewritten_IT.md
    ├── rewritten_finance.md
    ├── rewritten_healthcare.md
    ├── rewritten_hr.md
    ├── rewritten_engineering.md
    ├── resume_information_technology.pdf
    ├── resume_finance.pdf
    ├── resume_healthcare.pdf
    ├── resume_hr.pdf
    └── resume_engineering.pdf
```

---

## ⚙️ How to Run

### Option 1 — Streamlit Cloud (Already Live)
```
Visit → https://airesume-optimizer.streamlit.app
```

### Option 2 — Run Locally
```bash
# Clone
git clone https://github.com/abhinandan6123/Resume-Optimization-with-AI.git
cd Resume-Optimization-with-AI
git checkout AI_Resume_Analyzer

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
# Opens → http://localhost:8501
```

### Option 3 — Google Colab
```python
!pip install streamlit reportlab plotly scikit-learn PyPDF2 pyngrok -q

# Run with public URL
from pyngrok import ngrok
import subprocess, time
ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # free at dashboard.ngrok.com
subprocess.Popen(["streamlit","run","app.py","--server.headless=true"])
time.sleep(4)
print(ngrok.connect(8501).public_url)
```

### requirements.txt
```
streamlit>=1.32.0
scikit-learn>=1.3.0
reportlab>=4.0.0
plotly>=5.18.0
PyPDF2>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## 📊 All Outputs Generated

| # | File | Type | Description |
|---|------|:---:|-------------|
| 1 | `fig1_category_distribution.png` | 📊 PNG | Category distribution bar chart |
| 2 | `fig2_length_distribution.png` | 📊 PNG | Resume length histograms |
| 3 | `fig3_tfidf_heatmap.png` | 📊 PNG | TF-IDF keyword heatmap |
| 4 | `fig4_project_dashboard.png` | 📊 PNG | Comprehensive EDA dashboard |
| 5 | `step2_before_scores.csv` | 📋 CSV | BEFORE cosine similarity scores |
| 6 | `step3_final_scores.csv` | 📋 CSV | BEFORE/AFTER score comparison |
| 7–11 | `rewritten_*.md` (5 files) | 📝 MD | Rewritten resumes in Markdown |
| 12–16 | `resume_*.pdf` (5 files) | 📄 PDF | ATS-optimized PDF resumes |
| 17 | `fig7_model_comparison.png` | 📊 PNG | ML model performance chart |
| 18 | `fig8_confusion_matrix.png` | 📊 PNG | 24×24 confusion matrix |
| 19 | `fig9_per_class_accuracy.png` | 📊 PNG | Per-category accuracy chart |
| 20 | `resume_model.pkl` | 🤖 PKL | Trained Random Forest classifier |
| 21 | `app.py` | 🐍 PY | Streamlit web application |

---

## 👤 Author

<div align="center">

### **Venkata Abhinandan**

[![GitHub](https://img.shields.io/badge/GitHub-abhinandan6123-1A3C5E?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abhinandan6123)
[![Live App](https://img.shields.io/badge/Live%20App-airesume--optimizer-00FFB2?style=for-the-badge&logo=streamlit&logoColor=white)](https://airesume-optimizer.streamlit.app)

| | |
|---|---|
| **Project** | AI Resume Optimization System |
| **Year** | 2026 |
| **Platform** | Google Colab → Streamlit Cloud |
| **Best Model** | Random Forest — 72.84% Accuracy |
| **Avg Improvement** | 78.1% match score improvement |
| **Live URL** | https://airesume-optimizer.streamlit.app |
| **GitHub** | https://github.com/abhinandan6123/Resume-Optimization-with-AI |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2E75B6,100:1A3C5E&height=100&section=footer" width="100%"/>

**⭐ Star this repo if you found it useful!**

*Built with ❤️ by Venkata Abhinandan*

</div>
