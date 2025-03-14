# Heart Disease Prediction System

A web-based application built with FastAPI that uses machine learning to predict heart disease risk through clinical data analysis. The system provides comprehensive diagnostic reports in multiple languages using Google's Gemini LLM.

## Features

- Clinical data analysis using Deep Learning model
- Multi-language diagnostic reports using Gemini Model
- Secure user authentication
- Interactive dashboard
- Responsive web interface
- Detailed medical reports with:
  - Prediction Summary
  - Clinical Analysis
  - Risk Factor Assessment
  - Differential Considerations
  - Management Recommendations
  - Follow-up Protocol

## Prerequisites

- Python 3.10
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/heart-disease-pred-fastapi.git
   cd heart-disease-pred-fastapi
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup:**
   Create a `.env` file in the root directory:
   ```
   SECRET_KEY=your_secret_key_here
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   You can generate the SECRET_KEY using the following python code and copy/paste in the .env file:
   ```python
   import os
   import base64
   secret_key = base64.b64encode(os.urandom(32)).decode('utf-8')
   print(f"Generated SECRET_KEY: {secret_key}")
   ```
   You can get the Gemini API Key from the following link:
   https://aistudio.google.com/app/apikey

## Running the Application

1. **Start the server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   or
   ```bash
   python run.py
   ```

2. **Access the application:**
   - Web Interface: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

## Project Structure
```
heart-disease-pred-fastapi/
├── app/
│ ├── init.py
│ ├── auth.py # Authentication and authorization
│ ├── config.py # Configuration settings
│ ├── database.py # Database connection and definitions
│ ├── llm.py # Gemini Pro integration for report generation
│ ├── logger.py # Logging configuration
│ ├── main.py # FastAPI application entry point
│ ├── models.py # SQLAlchemy database models
│ ├── prediction.py # ML model prediction logic
│ └── utils.py # Utility functions
│
├── final_models/ # Machine Learning Models
│ ├── dl_best_model.h5
│ └── scaler_object.joblib
│
├── static/ # Static Files
│ ├── css/
│ │ └── style.css # Custom CSS styles
│ └── js/
│ └── markdown-converter.js # Markdown parsing utilities
│
├── templates/ # HTML Templates
│ ├── base.html # Base template with common elements
│ ├── dashboard.html # User dashboard template
│ ├── home.html # Prediction page template
│ ├── index.html # Landing page template
│ ├── login.html # Login page template
│ └── signup.html # Registration page template
│
├── .env # Environment variables
├── requirements.txt # Python dependencies
└── run.py # Application runner script
```

## Using the System

1. **Login/Registration:**
   - Create a new account or login with existing credentials
   - System uses JWT tokens for secure authentication

2. **Making Predictions:**
   - Enter clinical data including:
     - Age
     - Gender
     - Chest Pain Type
     - Blood Pressure
     - Cholesterol Level
     - Fasting Blood Sugar
     - ECG Results
     - Maximum Heart Rate
     - Exercise Angina
     - ST Depression
     - ST Slope
   - Select preferred language for the report
   - Submit for analysis

3. **Viewing Results:**
   - Get immediate prediction results
   - View detailed diagnostic reports
   - Access historical predictions in the dashboard

4. **Dashboard Features:**
   - View prediction history
   - Access detailed reports for each prediction

## Important Notes

- Ensure all ML model files are present in the `final_models/` directory
- Required model files:
  - `dl_best_model.h5`
  - `scaler_object.joblib`
- Internet connection required for Gemini Pro API
- All clinical measurements should be in standard units

## Troubleshooting

If you encounter any issues:

1. **Application won't start:**
   - Verify Python version (3.10)
   - Check if all dependencies are installed
   - Ensure environment variables are set correctly

2. **Prediction errors:**
   - Verify model files are in correct location
   - Ensure all clinical data fields are properly filled
   - Check data input ranges are within normal limits

3. **Report generation issues:**
   - Verify internet connection
   - Check Gemini API key in .env file
   - Ensure selected language is supported

4. **Page keeps loading on startup:**
   - Clear the Cookies and Site Data
   - Clear cache memory
   - Restart the server
