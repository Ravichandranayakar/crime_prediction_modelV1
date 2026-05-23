# 🚀 Streamlit Deployment Guide

## Crime Type Prediction - Streamlit App

This guide will help you run the Crime Type Prediction application locally or deploy it to the cloud.

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)

---

## 🛠️ Local Setup & Execution

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the App

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### Step 3: Use the Application

1. **Paste or type** the FIR narrative or complaint text in the text area
2. **Minimum requirement**: 20 characters and at least 8 words
3. **Select view mode**: Citizen view or Lawyer/Detailed view
4. **Click** "Predict Crime Types" button
5. **Review** the predicted crime categories and confidence scores

---

## 📁 Project Structure

```
crime_prediction_modelV1/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── DEPLOYMENT_GUIDE.md             # This file
├── crime_prediction_modelV1/
│   ├── data_clean_filesV1/         # Data cleaning notebooks
│   ├── Data_pipelineV1/            # Data pipeline scripts
│   ├── inferenceV1/                # Inference logic
│   │   └── crime_predictor.py      # Crime prediction class
│   └── model_trainingV1/
│       ├── artifacts/              # Trained model files
│       │   ├── crime_model_v1.pkl
│       │   ├── tfidf.pkl
│       │   ├── mlb.pkl
│       │   └── label_thresholds.json
│       └── models/                 # Training notebooks
```

---

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

1. **Push your repository to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit deployment"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path to: `app.py`
   - Click "Deploy"

Your app will be live at: `https://[your-username]-crime-prediction.streamlit.app`

---

### Option 2: Heroku

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Create Procfile** in project root:
   ```
   web: streamlit run app.py --logger.level=error
   ```

3. **Create .streamlit/config.toml**:
   ```toml
   [server]
   headless = true
   port = $PORT
   enableCORS = false
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

---

### Option 3: AWS / Google Cloud / Azure

Similar deployment process - refer to their Streamlit integration guides.

---

## 🔧 Configuration

### Change Model Path (if needed)

Edit the `MODEL_DIR` path in `app.py`:

```python
MODEL_DIR = Path(__file__).parent / "your/model/path"
```

### Adjust Thresholds

Modify `label_thresholds.json` to change prediction confidence thresholds for each crime category.

---

## 📊 Features

- **AI-Powered Classification** - TF-IDF + Logistic Regression
- **Multi-label Prediction** - Predict multiple crime types from single narrative
- **Confidence Scores** - Get probability estimates for predictions
- **Dual View Modes** - Citizen view (simple) and Lawyer view (detailed)
- **Professional UI** - Clean, responsive interface
- **Input Validation** - Minimum 8 words required for reliability
- **Legal Disclaimer** - Clear warning about AI limitations

---

## ⚠️ Important Notes

- **Results for Reference Only**: This model is for research and reference purposes
- **Not Legal Advice**: Always consult qualified legal professionals
- **Data Privacy**: Ensure compliance with data protection regulations
- **Model Limitations**: AI predictions may not be 100% accurate

---

## 🐛 Troubleshooting

### App won't start
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/
streamlit run app.py
```

### Model files not found
- Ensure model files are in the correct directory: `crime_prediction_modelV1/model_trainingV1/artifacts/`
- Check paths in `app.py` match your project structure

### Out of memory errors
- Consider running on a machine with more RAM
- Or deploy to cloud with more resources

---

## 📧 Support

For issues or questions, please create an issue in the repository.

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Happy Predicting!**
