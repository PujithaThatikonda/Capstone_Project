# Capstone Project – Data Engineering, Analytics & AI Support Assistant

## Project Overview

This capstone project was completed as part of the AI & Machine Learning learning journey and demonstrates the implementation of three industry-relevant modules:

1. Data Engineering Pipeline
2. Data Analytics & Machine Learning Pipeline
3. AI-Powered Support Assistant (RAG Application)

The project covers the complete lifecycle of data:

Data Collection → Data Cleaning → Data Storage → Data Analysis →
Machine Learning → Information Retrieval → API Deployment

---

# Project Structure

capstone_project/
│
├── data_pipeline/
│   ├── scrape_and_load.py
│   ├── run_queries.py
│   ├── books.db
│   ├── cleaned_books.csv
│   └── README.md
│
├── analytics/
│   ├── cleaned_titanic.csv
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── model_pipeline.pkl
│   ├── titanic.csv
│   └── README.md
│
├── support_assistant/
│   ├── docs/
│   ├── chroma_db/
│   ├── ingest.py
│   ├── graph.py
│   ├── models.py
│   ├── main.py
│   └── README.md
│
├── requirements.txt
└── README.md

---

# Module 1 – Data Engineering Pipeline

## Objective

Build a complete ETL pipeline that:

- Scrapes book data from Books to Scrape
- Cleans and validates data
- Converts currency values
- Stores data in SQLite
- Executes SQL queries
- Performs analysis using Pandas

---

## Technologies Used

- Python
- Requests
- BeautifulSoup
- SQLite3
- Pandas
- NumPy

---

## Workflow

BooksToScrape Website
        │
        ▼
Web Scraping
        │
        ▼
Data Cleaning
        │
        ▼
Currency Conversion
        │
        ▼
SQLite Database
        │
        ▼
SQL Queries
        │
        ▼
Pandas Analysis

---

## Data Collected

The scraper collects:

- Title
- Price
- Rating
- Availability
- Category

Categories scraped:

- Travel
- Mystery
- Historical Fiction
- Science Fiction

Total Books Collected:

67 Books

---

## Data Cleaning

Performed:

- Currency symbol removal
- Price conversion to float
- Rating conversion to numeric values
- Availability conversion to Boolean
- Missing value handling
- Median imputation

---

## Currency Conversion

Fixed conversion rate:

1 GBP = 105.50 INR

Formula:

price_inr = price_gbp × 105.50

---

## Database Design

### Categories Table

- category_id
- category_name

### Books Table

- book_id
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id

Relationship:

Categories (1) → (Many) Books

---

## SQL Operations

Implemented:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- IN
- JOIN

---

## Pandas Analysis

Performed:

- DataFrame creation
- SQL result loading
- Data aggregation
- Data filtering
- pd.merge() joins

---

# Module 2 – Analytics & Machine Learning Pipeline

## Objective

Develop a complete machine learning workflow using the Titanic dataset.

Workflow:

Load Data → Clean Data → EDA → Feature Engineering →
Train Models → Evaluate Models →
Tune Models → Save Final Model

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn
- Joblib

---

## Dataset

Titanic Dataset

Target Variable:

survived

0 = Did Not Survive

1 = Survived

---

## Data Cleaning

Performed:

- Missing value analysis
- Median imputation
- Mode imputation
- Data validation
- Feature preprocessing

---

## Exploratory Data Analysis

### Univariate Analysis

- Age Distribution
- Fare Distribution
- Histograms
- Boxplots
- Outlier Detection

### Bivariate Analysis

- Survival vs Gender
- Survival vs Passenger Class
- Survival vs Gender + Class

### Correlation Analysis

Features:

- survived
- pclass
- age
- sibsp
- parch
- fare

Heatmap generated using Seaborn.

---

## Feature Engineering

Implemented:

- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Pipeline

---

## Machine Learning Models

### Logistic Regression

Baseline classification model.

### Decision Tree

Interpretable classification model.

### Random Forest

Ensemble model with best performance.

---

## Model Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Additional Analysis:

- Confusion Matrix
- ROC Curves
- Classification Reports

---

## Handling Imbalanced Data

Compared:

1. Baseline Model
2. Class Weight Balancing
3. SMOTE Oversampling

---

## Hyperparameter Tuning

Used:

GridSearchCV

Tuned Parameters:

- n_estimators
- max_depth
- max_features

---

## Regression Task

Target:

fare

Algorithm:

Linear Regression

Metrics:

- MAE
- RMSE
- R²
- Adjusted R²

---

## Model Persistence

Saved using:

joblib.dump()

Output:

model_pipeline.pkl

---

# Module 3 – AI Support Assistant

## Objective

Build a Retrieval-Augmented Generation (RAG) application capable of answering questions from a custom document knowledge base.

Workflow:

Documents → Embeddings → Vector Database →
Retrieval → Answer Generation → API Response

---

## Technologies Used

- Python
- FastAPI
- Pydantic
- ChromaDB
- Sentence Transformers
- LangGraph
- Uvicorn

---

## Architecture

User Query
      │
      ▼
FastAPI Endpoint
      │
      ▼
LangGraph Workflow
      │
      ▼
Embedding Generation
      │
      ▼
ChromaDB Retrieval
      │
      ▼
Relevant Documents
      │
      ▼
Answer Generation
      │
      ▼
API Response

---

## Knowledge Base

Documents stored inside:

support_assistant/docs/

Examples:

- delivery_policy.txt
- refund_policy.txt
- support_faq.txt

---

## Document Ingestion

File:

ingest.py

Responsibilities:

- Read documents
- Generate embeddings
- Store embeddings
- Save metadata

Command:

python support_assistant/ingest.py

Output:

Documents indexed successfully

---

## Embedding Model

Model Used:

sentence-transformers/all-MiniLM-L6-v2

Purpose:

Convert text into semantic vector representations.

---

## Vector Database

Technology:

ChromaDB

Stores:

- Embeddings
- Metadata
- Document content

Provides:

- Similarity search
- Persistent storage
- Fast retrieval

---

## LangGraph Workflow

File:

graph.py

Responsibilities:

- Process queries
- Retrieve documents
- Generate answers
- Return confidence scores

Response Structure:

{
    "answer": "...",
    "sources": [...],
    "confidence": 0.92
}

---

## FastAPI Service

### Home Endpoint

GET /

Response:

{
    "message":
    "Zepto Support Assistant Running"
}

### Ask Endpoint

POST /ask

Request:

{
    "query":
    "What is the refund policy?"
}

Response:

{
    "answer":
    "...",
    "sources":
    ["refund_policy.txt"],
    "confidence":
    0.92
}

---

## Running the Support Assistant

Index Documents:

python support_assistant/ingest.py

Run API:

uvicorn support_assistant.main:app --reload

Swagger Documentation:

http://127.0.0.1:8000/docs

---

# Learning Outcomes

Through this capstone project I gained hands-on experience in:

### Data Engineering

- Web Scraping
- ETL Pipelines
- SQLite Database Design
- SQL Querying
- Data Cleaning

### Data Analytics & Machine Learning

- Exploratory Data Analysis
- Feature Engineering
- Classification Models
- Regression Models
- Hyperparameter Tuning
- Model Evaluation

### Generative AI & RAG

- Sentence Embeddings
- Vector Databases
- Semantic Search
- ChromaDB
- LangGraph
- FastAPI Deployment
- Retrieval-Augmented Generation

---

# Results

Successfully Implemented:

✔ Web Scraping Pipeline

✔ Data Cleaning Pipeline

✔ Currency Conversion

✔ SQLite Database Design

✔ SQL Querying

✔ Pandas Analysis

✔ Exploratory Data Analysis

✔ Logistic Regression

✔ Decision Tree

✔ Random Forest

✔ SMOTE Oversampling

✔ Hyperparameter Tuning

✔ Linear Regression

✔ Model Serialization

✔ FastAPI Service

✔ ChromaDB Vector Store

✔ Sentence Transformer Embeddings

✔ LangGraph Workflow

✔ Retrieval-Augmented Generation (RAG)

