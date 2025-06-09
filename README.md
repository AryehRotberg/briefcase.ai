# briefcase.ai

briefcase.ai is an AI-powered system for extracting, categorizing, and evaluating privacy-related statements from online service documents (such as privacy policies, terms of service, and related legal documents). The project leverages state-of-the-art NLP models to automate the analysis of privacy documents, providing structured bullet points and criticality-based categorization for downstream applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [API Usage](#api-usage)
- [Model Training & Evaluation](#model-training--evaluation)
- [Data Ingestion](#data-ingestion)
- [Setup & Installation](#setup--installation)
- [License](#license)

---

## Project Overview
briefcase.ai automates the extraction and classification of privacy-related statements from service documents. It uses a fine-tuned SentenceTransformer model to identify and categorize key statements, helping users and organizations better understand privacy risks and obligations in online services.

## Features
- **Automated Document Extraction:** Fetches and processes privacy-related documents from the web.
- **Bullet Point Extraction:** Identifies and extracts key statements from documents.
- **Criticality Categorization:** Assigns extracted statements to high, medium, or low criticality categories based on privacy impact.
- **Model Training & Evaluation:** Provides scripts and utilities for training and evaluating custom NLP models.

## Project Structure
```
utils/
  categories.py               # Definitions of privacy categories and criticality levels
src/
  data_ingestion/
    ingestion.py              # Data fetching and preprocessing from ToS;DR and other sources
    constants.py, utils.py    # Supporting constants and helpers
  model_training/
    training.py               # Model training pipeline for SentenceTransformer
    constants.py              # Training hyperparameters
  model_evaluation/
    evaluation.py             # Model evaluation and reporting
models/                       # Fine-tuned and pre-trained SentenceTransformer models
notebooks/                    # 
```

## Key Components

### 1. Data Ingestion (`src/data_ingestion/ingestion.py`)
- Fetches reviewed service IDs and downloads privacy-related data from ToS;DR APIs.
- Filters and saves data for model training and evaluation.

### 2. Model Training & Evaluation
- **Training Pipeline (`src/model_training/training.py`)**: Utilities for creating triplet datasets, training SentenceTransformer models, and saving checkpoints.
- **Evaluation (`src/model_evaluation/evaluation.py`)**: Computes F1 scores, confusion matrices, and classification reports for model performance.


## API Usage

For detailed API usage instructions, refer to the documentation available on Hugging Face: https://huggingface.co/spaces/AryehRotberg/briefcaseai-api.

## Website 

## Model Training & Evaluation
- **Training:** Use `src/model_training/training.py` to fine-tune SentenceTransformer models on your data.
- **Evaluation:** Use `src/model_evaluation/evaluation.py` to assess model performance (F1, confusion matrix, etc.).

## Data Ingestion
- **Ingestion:** Use `src/data_ingestion/ingestion.py` to fetch and preprocess data from ToS;DR and related sources.
- **Filtering:** Filter data by language and save as CSV for training.

## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AryehRotberg/briefcase.ai
   cd briefcaseai
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
