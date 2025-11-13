import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

# IMPORTS ADDED TO SUPPORT UNPICKLING (If used in your pipeline)
# If your model is a Keras model wrapped in Scikeras:
try:
    from scikeras.wrappers import KerasClassifier
except ImportError:
    pass # This will now be handled by requirements.txt

# List of columns dropped in your notebook
COLUMNS_TO_DROP = [
    'caseid', 'height', 'weight', 'op_duration', 'preop_sbp', 'preop_dbp', 
    'preop_pr', 'preop_rr', 'preop_temp', 'preop_plt', 'preop_bun', 'preop_cr', 
    'preop_na', 'preop_k', 'preop_cl', 'preop_glucose', 'preop_albumin', 
    'preop_pt', 'preop_ptt', 'preop_ph', 'preop_pao2', 'preop_paco2', 
    'preop_o2sat', 'preop_hco3', 'preop_be', 'cline2'
]

class ModelService:
    
    def __init__(self, preprocessor_path='./artifacts/preprocessor.pkl', model_path='./artifacts/icu_model.pkl'):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
        self.class_mapping = {0: '0 Days', 1: '1 Day', 2: '>1 Day'} 

    def load_artifacts(self) -> bool:
        try:
            print(f"Loading preprocessor from: {self.preprocessor_path}")
            self.preprocessor = joblib.load(self.preprocessor_path)
            
            print(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            print("Artifacts loaded successfully.")
            return True
        except FileNotFoundError as e:
            print(f"Error: Artifact file not found. Details: {e}")
            self.preprocessor = None
            self.model = None
            return False
        except Exception as e:
            print(f"An error occurred during artifact loading: {e}")
            return False

    def preprocess_input(self, raw_data: Dict[str, Any]) -> np.ndarray:
        df = pd.DataFrame([raw_data])
        
        # 1. Drop irrelevant columns
        df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')
        
        # 2. Handle 'age' feature
        if 'age' in df.columns:
            df['age'] = df['age'].astype(str).replace('>89', '90').replace('None', np.nan).astype(float)
        
        # 3. Handle text feature imputation
        text_features = ['preop_ecg', 'preop_pft', 'dx', 'opname', 'optype']
        for col in text_features:
             if col in df.columns:
                 df[col] = df[col].fillna('')

        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded. Cannot process data.")

        # 4. Apply all fitted transformations
        processed_features = self.preprocessor.transform(df)
        
        return processed_features

    def predict(self, processed_data: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make prediction.")

        prediction = self.model.predict(processed_data)
        
        # Handle potential nested array output from Keras/Scikeras
        if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
            predicted_class_label = int(prediction.flatten()[0])
        else:
            predicted_class_label = int(prediction[0])
            
        return self.class_mapping.get(predicted_class_label, f"Unknown Class ({predicted_class_label})")

model_service = ModelService()

if not model_service.load_artifacts():
    pass
