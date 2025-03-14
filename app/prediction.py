import os
import joblib
import numpy as np
import tensorflow as tf
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import json
from app.logger import logger
from .models import Prediction
from .llm import LLM
from .config import MODELS_DIR

# Custom exceptions
class ModelLoadingError(Exception):
    pass

class PreprocessingError(Exception):
    pass

class PredictionError(Exception):
    pass


# Data preprocessing class
class DataPreprocessor:
    """Handles data preprocessing for the ML model input."""

    def __init__(self):
        try:
            logger.info("Initializing DataPreprocessor")
            self.scaler = joblib.load(f"{MODELS_DIR}/scaler_object.joblib")
            logger.debug("Scaler loaded successfully")
        except Exception as e:
            logger.error("Failed to load scaler", exc_info=True)
            raise ModelLoadingError("Could not load the scaler for data preprocessing.")

    def preprocess(self, input_data):
        """Preprocess input data using the loaded scaler."""
        try:
            logger.debug(f"Preprocessing input data: {input_data}")
            scaled_data = self.scaler.transform(np.array(input_data).reshape(1, -1))
            return scaled_data
        except Exception as e:
            logger.error("Error in data preprocessing", exc_info=True)
            raise PreprocessingError("Preprocessing failed. Ensure input data format is correct.")

# ML model predictor
class ML_Model_Predictor:
    """Handles prediction using the ML Model."""

    def __init__(self):
        try:
            self.model = tf.keras.models.load_model(f"{MODELS_DIR}/dl_best_model.h5")
            logger.info("ML model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load ML model", exc_info=True)
            raise ModelLoadingError("Could not load the ML model.")
        
    def predict(self, preprocessed_data):
        """Make predictions using the loaded ML model."""
        try:
            prediction = self.model.predict(preprocessed_data)
            logger.info("ML model prediction completed successfully.")
            return prediction[0]
        except Exception as e:
            logger.error("Error during ML prediction", exc_info=True)
            raise PredictionError("ML prediction failed.")


# Global Variables to store model instances
_preprocessor = None
_ml_predictor = None
_llm = None

def initialize_models():
    """Initialize and load all models once during application startup"""
    global _preprocessor, _ml_predictor, _llm

    try:
        logger.info("Initializing models on application startup")
        _preprocessor = DataPreprocessor()
        _ml_predictor = ML_Model_Predictor()
        _llm = LLM()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error("Failed to load models on startup", exc_info=True)
        raise ModelLoadingError(f"Could not load models during initialization: {str(e)}")

def clear_models():
    """Clear all model from memory during application shutdown"""
    global _preprocessor, _ml_predictor, _llm
    _preprocessor = None
    _ml_predictor = None
    _llm = None

def get_models():
    """Get the initialized models"""
    global _preprocessor, _ml_predictor, _llm

    if _preprocessor is None or _ml_predictor is None or _llm is None:
        # If models aren't initialized yet, initialize them
        initialize_models()
    return _preprocessor, _ml_predictor, _llm

def make_prediction(
        db: Session,
        user_id: int,
        clinical_data: Dict[str, Any],
        language: str = "English"
) -> Prediction:
    try:
        # Get the already initialized models
        preprocessor, ml_predictor, llm = get_models()

        # Prepare structured data input for ML model (extract from clinical_data dict)
        structured_data = [
            clinical_data.get('age'),
            clinical_data.get('gender'),
            clinical_data.get('chest_pain'),
            clinical_data.get('bp'),
            clinical_data.get('cholesterol'),
            clinical_data.get('blood_sugar'),
            clinical_data.get('electrocardiographic'),
            clinical_data.get('heart_rate'),
            clinical_data.get('exercise_angina'),
            clinical_data.get('oldpeak'),
            clinical_data.get('slope')
        ]

        # Preprocess the data
        preprocessed_data = preprocessor.preprocess(structured_data)

        # Get prediction from ML model
        ml_prediction_result = np.round(ml_predictor.predict(preprocessed_data)).astype(int)

        # Log the final predictions
        logger.info(f"Final ML prediction (clinical model): {ml_prediction_result} - {'Affected' if ml_prediction_result == 1 else 'Not Affected'}")
        
        # Format result string
        result = f"""
        Heart Disease Diagnosis Report:

        **ML Model Prediction:** {'Yes' if ml_prediction_result == 1 else 'No'}
        
        **Input Data:**
        - Age: {clinical_data.get('age')}
        - Gender: {clinical_data.get('gender')}
        - Chest Pain Type: {clinical_data.get('chest_pain')}
        - Blood Pressure: {clinical_data.get('bp')}
        - Cholesterol: {clinical_data.get('cholesterol')}
        - Blood Sugar: {clinical_data.get('blood_sugar')}
        - Electrocardiographic: {clinical_data.get('electrocardiographic')}
        - Heart Rate: {clinical_data.get('heart_rate')}
        - Exercise Angina: {clinical_data.get('exercise_angina')}
        - Oldpeak: {clinical_data.get('oldpeak')}
        - Slope: {clinical_data.get('slope')}
        """

        # Generate LLM report
        report = llm.inference(result=result, language=language)

        # Create prediction record
        prediction = Prediction(
            user_id=user_id,
            clinical_features=clinical_data,
            clinical_model_result=bool(ml_prediction_result==1),
            language=language,
            report=report
        )

        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        return prediction
    
    except Exception as e:
        logger.error("Error in prediction function", exc_info=True)
        db.rollback()
        raise PredictionError(f"Prediction function encountered an error: {str(e)}")


def get_user_predictions(db: Session, user_id: int) -> List[Prediction]:
    """Get all predictions for a user"""
    return db.query(Prediction).filter(Prediction.user_id == user_id).order_by(Prediction.created_at.desc()).all()
