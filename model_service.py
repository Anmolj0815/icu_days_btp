import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

# Required for unpickling models trained with Scikeras
try:
    from scikeras.wrappers import KerasClassifier
except ImportError:
    pass 

# All numerical features that require imputation/scaling/PCA (must match artifact_saver)
ALL_NUMERICAL_FEATURES = [
    'age', 'preop_hb', 'preop_wbc', 'intraop_ebl', 'bmi', 
    'preop_sbp', 'preop_dbp', 'preop_pr', 'preop_rr', 'preop_temp', 
    'preop_plt', 'preop_bun', 'preop_cr', 'preop_na', 'preop_k', 'preop_cl', 
    'preop_glucose', 'preop_alb', 'preop_pt', 'preop_ptt', 'preop_ph', 
    'preop_pao2', 'preop_paco2', 'preop_o2sat', 'preop_hco3', 'preop_be', 
    'intraop_sbp_min', 'intraop_dbp_min', 'intraop_pr_min', 'urine_output', 
    'postop_hb', 'postop_wbc', 'postop_cr', 'cardiac_risk', 'room_temp', 
    'preop_inr', 'preop_fibrinogen', 'intraop_temp_min', 'intraop_temp_max',
    'height', 'weight', 'op_duration', 'asa', 'intraop_crystalloid', 'intraop_colloid', 
    'intraop_rbc', 'intraop_ffp', 'intraop_ftn', 'intraop_mdz', 'intraop_vecu', 
    'intraop_rocu', 'intraop_eph', 'intraop_epi', 'intraop_phe', 'intraop_ppf', 
    'intraop_ca', 'tubesize', 'preop_gluc', 'preop_ast', 'preop_alt', 'preop_aptt'
]

# Columns that were explicitly dropped in the notebook
# IMPORTANT FIX: Removed core lab values that were causing repeated errors.
COLUMNS_TO_DROP = [
    'caseid', 'cline2', 'op_duration', 'height', 'weight', 'preop_glucose'
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
        
        # 1. Standardize DataFrame structure and ensure all NUMERICAL columns exist as floats (with NaN for missing)
        for col in ALL_NUMERICAL_FEATURES:
            # This ensures the column exists in the DataFrame for the Imputer to work
            if col not in df.columns:
                df[col] = np.nan
            else:
                # Convert to numeric, errors='coerce' turns strings/unparsable data into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Handle 'age' feature conversion to numeric 
        if 'age' in df.columns:
            df['age'] = df['age'].astype(str).str.replace('>89', '90', regex=False)
            df['age'] = pd.to_numeric(df['age'], errors='coerce') 

        # 3. Handle text feature imputation 
        text_features = ['preop_ecg', 'preop_pft', 'dx', 'opname', 'optype']
        for col in text_features:
             if col in df.columns:
                 df[col] = df[col].fillna('')

        # 4. Drop columns that are completely irrelevant and not part of the preprocessing pipeline
        df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')

        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded. Cannot process data.")

        # 5. Apply all fitted transformations
        processed_features = self.preprocessor.transform(df)
        
        return processed_features

    def predict(self, processed_data: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make prediction.")

        prediction = self.model.predict(processed_data)
        
        if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
            predicted_class_label = int(prediction.flatten()[0])
        else:
            predicted_class_label = int(prediction[0])
            
        return self.class_mapping.get(predicted_class_label, f"Unknown Class ({predicted_class_label})")

model_service = ModelService()

if not model_service.load_artifacts():
    pass
