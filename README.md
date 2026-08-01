# Crime Type Prediction System

> **AI-Powered Crime Classification from FIR Narrative Text**

A machine learning-based system designed to automatically classify crime types from First Information Report (FIR) narratives and complaint texts using advanced NLP techniques.

## Try the App Now!

**[Visit the Live Application](https://ravichandranayakar-crime-prediction-modelv1-app-aihlwf.streamlit.app/)**

<img width="1919" height="1029" alt="image" src="https://github.com/user-attachments/assets/4456448c-af74-40e5-8187-432964b7700d" />


Click the link above to test the crime prediction model in action! No installation required - just paste a FIR narrative and get instant predictions.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Usage](#usage)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## 🎯 Overview

The Crime Type Prediction System is an intelligent classification engine that analyzes FIR narratives and complaint texts to predict associated crime types. Built with machine learning and deployed via a modern web interface, it provides law enforcement and legal professionals with AI-assisted crime categorization.

### Key Capabilities
- **Multi-label Classification**: Predict multiple crime types from a single narrative
- **High Precision**: Uses optimized confidence thresholds for each crime category
- **Real-time Processing**: Instant predictions on submitted text
- **Professional Interface**: User-friendly web application for easy access
- **Scalable Architecture**: Ready for enterprise deployment

---

##  Features

### Core Features
- **TF-IDF + Logistic Regression** - Robust ML classification
- **Multi-label Predictions** - Detect multiple crime types simultaneously
- **Confidence Scoring** - Probability estimates for each prediction
- **Dual View Modes** - Citizen view and Lawyer/detailed view
- **Input Validation** - Intelligent text validation (minimum 8 words)
- **Real-time Processing** - Instant results

### Application Features
- **Professional UI** - Streamlit-based web interface
- **Detailed Analytics** - View all crime probabilities
- **Legal Disclaimers** - Clear warning about model limitations
- **Data Privacy** - No data storage or logging
- **Responsive Design** - Works on desktop and mobile
- **Cloud Ready** - Easy deployment to Streamlit Cloud, Heroku, AWS

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Web Interface                    │
│  (app.py - Text Input, Results Display, Configuration)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Input Validation & Processing                   │
│    (Text cleaning, word count validation, formatting)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│         TF-IDF Text Vectorization (tfidf.pkl)               │
│    (Converts narrative text to numerical features)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│    Logistic Regression Model (crime_model_v1.pkl)           │
│    (Generates probability scores for 15 crime categories)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│   Threshold Application (label_thresholds.json)             │
│  (Converts probabilities to binary predictions per category)│
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│   Multi-Label Binarizer (mlb.pkl)                           │
│    (Decodes predictions to readable crime labels)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│          Results & Confidence Scores                        │
│    (Displays predictions with probabilities and warnings)   │
└─────────────────────────────────────────────────────────────┘
```

---

##  Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Web Framework** | Streamlit | 1.36.0 |
| **ML Library** | scikit-learn | 1.3.0 |
| **Data Processing** | NumPy | 1.24.3 |
| **Serialization** | joblib | 1.3.2 |
| **Language** | Python | 3.8+ |
| **Deployment** | Docker / Cloud Platforms | Latest |

---

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum
- 500MB disk space

### Setup Instructions

**1. Clone the Repository**
```bash
git clone https://github.com/Ravichandranayakar/crime_prediction_modelV1.git
cd crime_prediction_modelV1
```

**2. Create Virtual Environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify Installation**
```bash
streamlit --version
```

---

##  Quick Start

### Run Locally

```bash
streamlit run app.py
```

The application will launch at: `http://localhost:8501`

### Using the Application

1. **Enter FIR Narrative**: Paste or type the FIR narrative/complaint text
2. **Minimum Requirements**: At least 20 characters and 8 words
3. **Select View Mode**: Choose between Citizen view or Lawyer/Detailed view
4. **Predict**: Click "Predict Crime Types" button
5. **Review Results**: Examine predicted crime types and confidence scores

### Example Input
```
A person was attacked with a knife in a dark alley by an unknown individual 
who then fled the scene. The victim was severely injured and required 
immediate medical attention at the hospital.
```

---

## 📁 Project Structure

```
crime_prediction_modelV1/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── PROJECT_SUMMARY.md                  # Detailed project documentation
├── DEPLOYMENT_GUIDE.md                 # Deployment instructions
│
├── crime_prediction_modelV1/           # Core project directory
│   │
│   ├── data_clean_filesV1/             # Data preprocessing
│   │   ├── dataset_clean1.ipynb        # Data cleaning notebook 1
│   │   ├── dataset_clean2.ipynb        # Data cleaning notebook 2
│   │   └── dataset_clean3.ipynb        # Data cleaning notebook 3
│   │
│   ├── Data_pipelineV1/                # Data ingestion
│   │   └── download_bail_dataset.py    # Dataset download script
│   │
│   ├── model_trainingV1/               # Model development
│   │   ├── models/
│   │   │   └── 01_train_basline_tfidf_logreg.ipynb  # Training notebook
│   │   │
│   │   └── artifacts/                  # Trained model files
│   │       ├── crime_model_v1.pkl      # Logistic Regression model
│   │       ├── tfidf.pkl               # TF-IDF vectorizer
│   │       ├── mlb.pkl                 # Multi-label binarizer
│   │       └── label_thresholds.json   # Decision thresholds
│   │
│   └── inferenceV1/                    # Prediction engine
│       ├── crime_predictor.py          # Core prediction logic
│       └── __pycache__/
│
└── .git/                               # Version control
```

---

## 🤖 Model Details

### Algorithm
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifier**: Logistic Regression (Multi-label)
- **Approach**: One-vs-Rest strategy for multi-label classification

### Crime Categories (15 Classes)

| Category | Threshold | Description |
|----------|-----------|-------------|
| Homicide | 0.25 | Unlawful killing |
| Sexual Offence | 0.25 | Sexual assault/abuse |
| Kidnapping | 0.30 | Unlawful confinement |
| Domestic Violence | 0.30 | Violence in domestic settings |
| Cyber Crime | 0.35 | Digital/online crimes |
| Fraud | 0.40 | Financial deception |
| Cheating | 0.40 | Deceptive practices |
| Robbery | 0.40 | Theft with force |
| Extortion | 0.40 | Coerced payments |
| Assault | 0.45 | Physical attack |
| Theft | 0.55 | Taking property |
| Criminal Intimidation | 0.55 | Threats/coercion |
| House Trespass | 0.60 | Unlawful entry |
| Weapon Used | 0.60 | Weapons involved |
| Grievous Hurt | 0.60 | Severe injury |

### Model Performance
- **Training Approach**: Multi-label classification with optimized thresholds
- **Confidence Thresholds**: Calibrated per crime category for optimal precision
- **Minimum Input**: 8 words (ensures reliable predictions)
- **Processing Time**: <1 second per prediction

---

## 📖 Usage

### Basic Usage

```python
from crime_prediction_modelV1.inferenceV1.crime_predictor import CrimePredictor

# Initialize predictor
predictor = CrimePredictor(model_dir="crime_prediction_modelV1/model_trainingV1/artifacts")

# Make prediction
result = predictor.predict(
    text="Your FIR narrative here...",
    mode="citizen"  # or "lawyer" for detailed view
)

# Results
print(result)
# {
#     "labels": ["robbery", "assault"],
#     "confidence": 0.78,
#     "warning": "This is only an AI indication, not legal advice."
# }
```

### View Modes

**Citizen View** (Default)
- Shows only commonly understood crime types
- Simplified output for general public
- Less technical information

**Lawyer View** (Detailed)
- Shows all 15 crime categories
- Detailed probability scores
- Full classification information

---

##  Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud (Free)
1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub account and deploy

### Docker Deployment
```bash
docker build -t crime-predictor .
docker run -p 8501:8501 crime-predictor
```

### Production Deployment
See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions on:
- Heroku deployment
- AWS deployment
- Azure deployment
- Google Cloud deployment

---

##  API Reference

### CrimePredictor Class

**Initialization**
```python
CrimePredictor(model_dir="path/to/artifacts")
```

**Methods**

#### `predict(text, mode="citizen")`
Predicts crime types from narrative text.

**Parameters:**
- `text` (str): FIR narrative or complaint text
- `mode` (str): "citizen" or "lawyer" for different detail levels

**Returns:**
- `dict`: Prediction results with labels, confidence, and warning

**Example:**
```python
result = predictor.predict("Your narrative text", mode="citizen")
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Supported Crime Categories** | 15 |
| **Average Prediction Time** | <1 second |
| **Minimum Input Length** | 8 words |
| **Maximum Input Length** | 5000 characters |
| **Model Size** | ~5MB |
| **Memory Footprint** | ~200MB |

---

## ⚠️ Disclaimer

**IMPORTANT LEGAL NOTICE:**

1. **Not Legal Advice**: This system provides AI-based predictions for reference only and should NOT be considered legal advice.

2. **For Professional Use Only**: Results should be reviewed by qualified legal professionals before any official use.

3. **Accuracy Limitations**: The model may not achieve 100% accuracy. Always verify predictions through proper legal channels.

4. **Data Privacy**: This application does not store, log, or retain any submitted text data.

5. **Liability**: Users accept full responsibility for the use and interpretation of predictions.

6. **Compliance**: Ensure compliance with local data protection regulations when deploying this system.

---

## 📄 License

This project is provided for educational and research purposes. All rights reserved.

---

##  Support & Contact

For issues, questions, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/Ravichandranayakar/crime_prediction_modelV1/issues)
- **Documentation**: See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) and [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---


---

## 🙏 Acknowledgments

Built with cutting-edge machine learning techniques and best practices in software engineering.

---
