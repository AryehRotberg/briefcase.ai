# briefcase.ai

![System Architecture](assets/System%20Architecture.png)

briefcase.ai is an AI-powered system for extracting, categorizing, and evaluating privacy-related statements from online service documents (such as privacy policies, terms of service, and related legal documents). The project leverages state-of-the-art NLP models to automate the analysis of privacy documents, providing structured bullet points and criticality-based categorization for downstream applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Database Models](#database-models)
- [Notebooks](#notebooks)
- [Website Interface](#website-interface)
- [Setup & Installation](#setup--installation)
- [License](#license)

---

## Project Overview
briefcase.ai automates the extraction and classification of privacy-related statements from service documents. It uses a fine-tuned SentenceTransformer model to identify and categorize key statements, helping users and organizations better understand privacy risks and obligations in online services. The system is built using briefcase.ai branding for the web interface.

## Features
- **Automated Document Processing:** Extracts and processes privacy-related documents using advanced NLP techniques.
- **Bullet Point Extraction:** Identifies and extracts key statements from documents using ChromaDB-based similarity matching.
- **Intelligent Categorization:** Assigns extracted statements to privacy categories with standardized mappings.
- **Multi-Database Support:** Supports PostgreSQL, MongoDB, and ChromaDB for flexible data storage.
- **Model Training & Evaluation:** Comprehensive pipelines for training and evaluating custom SentenceTransformer models.
- **Data Pipeline:** Complete ETL pipeline for ingesting, processing, and storing privacy document data.

## Project Structure
```
briefcase.ai/
├── data/                           # Raw and processed datasets
│   ├── data_all.csv               # Complete dataset
│   ├── data_all_cleaned.csv       # Cleaned dataset for training
│   └── reviewed_service_ids.txt   # Service IDs for processing
├── database/                      # Database abstraction layer
│   ├── base.py                    # SQLAlchemy base configuration
│   ├── chromadb.py               # ChromaDB connection and operations
│   ├── mongodb.py                # MongoDB connection and operations
│   ├── postgresql.py             # PostgreSQL connection and operations
│   ├── config.py                 # Database configuration
│   ├── models/                   # Database models
│   │   ├── policy.py             # Policy and PolicyProcessed models
│   │   └── service_id.py         # Service ID model
│   └── utils/
│       └── categories.py         # Category utility functions
├── src/                          # Main source code
│   ├── components/
│   │   ├── data_ingestion/       # Data loading and preprocessing
│   │   │   ├── data_loader.py    # Data fetching from ToS;DR APIs
│   │   │   ├── preprocessor.py   # Data cleaning and standardization
│   │   │   ├── utils.py          # Language detection and text cleaning
│   │   │   └── config/           # Configuration files
│   │   │       ├── api_settings.py      # API endpoint configurations
│   │   │       ├── category_maps.py     # Category standardization mappings
│   │   │       └── classification_levels.py  # Classification level definitions
│   │   ├── model_training/       # Model training pipeline
│   │   │   ├── training.py       # SentenceTransformer training logic
│   │   │   └── constants.py      # Training hyperparameters
│   │   └── model_evaluation/     # Model evaluation and metrics
│   │       └── evaluation.py     # Performance evaluation and reporting
│   └── pipelines/
│       └── data_ingestion.py     # Main data pipeline orchestration
└──  utils/                        # Utility modules
   └── chroma_bullet_extractor.py  # ChromaDB-based text extraction
```

## Key Components

### 1. Data Ingestion Pipeline (`src/components/data_ingestion/`)
- **`data_loader.py`**: Fetches privacy policy data from ToS;DR APIs and stores in databases
- **`preprocessor.py`**: Cleans and standardizes category mappings using predefined transformations
- **`utils.py`**: Language detection, text cleaning, and data validation utilities
- **Configuration**: API settings, category mappings, and classification level definitions

### 2. Model Training (`src/components/model_training/`)
- **`training.py`**: Complete SentenceTransformer training pipeline with triplet loss
- **`constants.py`**: Training hyperparameters and model configuration
- Supports checkpoint saving and resuming training

### 3. Model Evaluation (`src/components/model_evaluation/`)
- **`evaluation.py`**: Comprehensive model performance analysis
- Generates F1 scores, confusion matrices, and classification reports
- Supports query-based predictions for testing

### 4. Database Layer (`database/`)
- **Multi-database support**: PostgreSQL, MongoDB, and ChromaDB
- **SQLAlchemy models**: Policy and ServiceId models for structured data
- **Abstraction layer**: Unified interface across different database backends

### 5. Bullet Point Extraction (`utils/chroma_bullet_extractor.py`)
- **ChromaDB Integration**: Vector-based similarity search for document analysis
- **Smart text chunking**: Intelligent document segmentation
- **Category-based extraction**: Organized extraction by privacy categories

### 6. Web Interface (`website/`)
- **Modern UI**: Interactive demo interface branded as briefcase.ai
- **Dual functionality**: Document finding and text analysis capabilities
- **Real-time processing**: Live document analysis with visual feedback

## Database Models

### Policy Models (`database/models/policy.py`)
- **`Policy`**: Raw policy data with point_id, category, source, service_id, and service_name
- **`PolicyProcessed`**: Cleaned and processed policy data with standardized categories

### Service Models (`database/models/service_id.py`)
- Service identification and metadata management

## Notebooks

### Data Analysis and Experimentation
- **`Data Cleaning.ipynb`**: Interactive data preprocessing and cleaning workflows
- **`Training.ipynb`**: Model training experiments and hyperparameter tuning
- **`Evaluation.ipynb`**: Model performance analysis and visualization

## Website Interface

The project includes a modern web interface branded as **briefcase.ai** that provides:
- **Interactive Demo**: Two-tab interface for document finding and text analysis
- **Real-time Analysis**: Live processing of privacy policies and terms of service
- **Category Visualization**: Organized display of extracted privacy points
- **Responsive Design**: Mobile-friendly interface with modern styling

## Setup & Installation

### Prerequisites
- Python 3.12 or higher
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AryehRotberg/briefcase.ai.git
   cd briefcase.ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt  # If requirements.txt exists
   # Or install individually based on the modules you need:
   pip install sentence-transformers chromadb sqlalchemy pandas numpy scikit-learn
   pip install postgresql-adapter pymongo  # For database support
   pip install aiohttp asyncio  # For async data loading
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env  # If .env.example exists
   # Edit .env with your database credentials and API keys
   ```

5. **Initialize databases (if using):**
   ```python
   # Run database initialization scripts
   python -c "from database.base import Base; from database.postgresql import engine; Base.metadata.create_all(engine)"
   ```

### Usage

#### Data Pipeline
```bash
# Run the complete data ingestion pipeline
python src/pipelines/data_ingestion.py
```

#### Model Training
```python
# Train a new model
from src.components.model_training.training import ModelTraining
trainer = ModelTraining()
# Configure and run training
```

#### Web Interface
Open `website/index.html` in a web browser to access the briefcase.ai interface.

#### Bullet Point Extraction
```python
from utils.chroma_bullet_extractor import ChromaBulletExtractor
# Initialize and use the extractor
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
