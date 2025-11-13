from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np

from model_service import model_service

class ICUData(BaseModel):
    
    # 1. IDENTIFIERS & BASIC DEMOGRAPHICS
    caseid: Optional[str] = None
    sex: Literal['M', 'F', 'nan', 'MALE', 'FEMALE']
    age: str
    height: Optional[float] = None
    weight: Optional[float] = None
    bmi: Optional[float] = Field(None, description="Body Mass Index (BMI).") 
    
    # 2. SURGICAL & ANESTHESIA DETAILS
    department: str
    approach: str
    position: str
    ane_type: str
    op_duration: Optional[float] = None
    iv1: str
    iv2: Optional[str] = None
    blood_prod: Optional[str] = None
    asa: Optional[float] = None
    emop: Optional[Literal['Y', 'N']] = Field(None, description="Emergency operation status.") 
    tubesize: Optional[float] = Field(None, description="Endotracheal tube size.") 

    # 3. TEXTUAL FEATURES 
    preop_ecg: Optional[str] = None
    preop_pft: Optional[str] = None
    dx: Optional[str] = None
    opname: Optional[str] = None
    optype: Optional[str] = None
    
    # 4. PRE-OPERATIVE LABS & VITALS
    preop_sbp: Optional[float] = None
    preop_dbp: Optional[float] = None
    preop_pr: Optional[float] = None
    preop_rr: Optional[float] = None
    preop_temp: Optional[float] = None
    
    preop_hb: Optional[float] = None
    preop_wbc: Optional[float] = None
    preop_plt: Optional[float] = None 
    preop_bun: Optional[float] = None 
    preop_cr: Optional[float] = None 
    preop_na: Optional[float] = None 
    preop_k: Optional[float] = None 
    preop_cl: Optional[float] = None
    preop_glucose: Optional[float] = None
    preop_albumin: Optional[float] = None 
    preop_pt: Optional[float] = None 
    preop_ptt: Optional[float] = None

    # ADDED MISSING PRE-OP LABS
    preop_gluc: Optional[float] = Field(None, description="Pre-op Glucose/Gluc.") 
    preop_ast: Optional[float] = Field(None, description="Pre-op AST liver enzyme.") 
    preop_alt: Optional[float] = Field(None, description="Pre-op ALT liver enzyme.") 
    preop_aptt: Optional[float] = Field(None, description="Pre-op aPTT.") 
    preop_dm: Optional[Literal['Y', 'N', 'Unknown']] = Field(None, description="Pre-op Diabetes Mellitus status.") 
    
    preop_ph: Optional[float] = None
    preop_pao2: Optional[float] = None
    preop_paco2: Optional[float] = None
    preop_o2sat: Optional[float] = None
    preop_hco3: Optional[float] = None
    preop_be: Optional[float] = None
    cline2: Optional[str] = None

    # 5. INTRA-OPERATIVE PARAMETERS
    intraop_sbp_min: Optional[float] = None
    intraop_dbp_min: Optional[float] = None
    intraop_pr_min: Optional[float] = None
    intraop_ebl: Optional[float] = None
    urine_output: Optional[float] = None
    
    # ADDED NEW INTRA-OP DRUGS/FLUIDS
    intraop_crystalloid: Optional[float] = Field(None, description="Intra-op Crystalloid volume.") 
    intraop_colloid: Optional[float] = Field(None, description="Intra-op Colloid volume.") 
    intraop_rbc: Optional[float] = Field(None, description="Intra-op Red Blood Cell transfusion units.") 
    intraop_ffp: Optional[float] = Field(None, description="Intra-op FFP units.") 
    
    # ADDED NEW INTRA-OP DRUGS/AGENTS
    intraop_ftn: Optional[float] = Field(None, description="Intra-op Fentanyl dose.") 
    intraop_mdz: Optional[float] = Field(None, description="Intra-op Midazolam dose.") 
    intraop_vecu: Optional[float] = Field(None, description="Intra-op Vecuronium dose.") 
    intraop_rocu: Optional[float] = Field(None, description="Intra-op Rocuronium dose.") 
    intraop_eph: Optional[float] = Field(None, description="Intra-op Ephedrine dose.") 
    intraop_epi: Optional[float] = Field(None, description="Intra-op Epinephrine dose.") 
    intraop_phe: Optional[float] = Field(None, description="Intra-op Phenylephrine dose.") 
    intraop_ppf: Optional[float] = Field(None, description="Intra-op Propofol dose.") 
    intraop_ca: Optional[float] = Field(None, description="Intra-op Calcium dose.") 
    
    # 6. POST-OPERATIVE/RISK PARAMETERS
    postop_hb: Optional[float] = None
    postop_wbc: Optional[float] = None
    postop_cr: Optional[float] = None
    cardiac_risk: Optional[float] = None
    renal_function_status: Optional[str] = None
    liver_function_status: Optional[str] = None
    preop_diabetes: Optional[Literal['Y', 'N', 'Unknown']] = None
    preop_htn: Optional[Literal['Y', 'N', 'Unknown']] = None
    preop_copd: Optional[Literal['Y', 'N', 'Unknown']] = None
    preop_chf: Optional[Literal['Y', 'N', 'Unknown']] = None
    ventilation_mode: Optional[str] = None
    muscle_relaxant: Optional[str] = None
    opioid_used: Optional[str] = None
    antibiotic_prophylaxis: Optional[str] = None
    surgeon_id: Optional[str] = None
    anesthesiologist_id: Optional[str] = None
    room_temp: Optional[float] = None
    preop_inr: Optional[float] = None
    preop_fibrinogen: Optional[float] = None
    intraop_temp_min: Optional[float] = None
    intraop_temp_max: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "caseid": "C-2025", "sex": "F", "age": "70", "height": 165.0, "weight": 65.0, "bmi": 23.9,
                "department": "Cardio", "approach": "Minimally Invasive", "position": "Lateral",
                "ane_type": "General", "op_duration": 180.0, "iv1": "Ringer Lactate", "iv2": "None",
                "blood_prod": "None", "asa": 3.0, "emop": "N", "tubesize": 7.5,
                "preop_ecg": "Atrial fibrillation, controlled rate", "preop_pft": "Normal", 
                "dx": "Mitral valve regurgitation", "opname": "Valve Repair",
                "optype": "Major", "preop_sbp": 140.0, "preop_dbp": 90.0, "preop_pr": 85.0,
                "preop_rr": 16.0, "preop_temp": 36.8, "preop_hb": 10.5, "preop_wbc": 12.1,
                "preop_plt": 200.0, "preop_bun": 25.0, "preop_cr": 1.5, "preop_na": 135.0,
                "preop_k": 4.8, "preop_cl": 105.0, "preop_glucose": 120.0, "preop_albumin": 3.0,
                "preop_pt": 15.0, "preop_ptt": 40.0,
                "preop_gluc": 120.0, "preop_ast": 25.0, "preop_alt": 20.0, "preop_aptt": 30.0, "preop_dm": "N",
                "preop_ph": 7.35, "preop_pao2": 80.0, "preop_paco2": 45.0, "preop_o2sat": 95.0,
                "preop_hco3": 26.0, "preop_be": -1.0, "cline2": "Patient is frail", 
                "intraop_sbp_min": 85.0, "intraop_dbp_min": 45.0, "intraop_pr_min": 70.0, 
                "intraop_ebl": 500.0, "urine_output": 100.0, 
                "intraop_crystalloid": 2000.0, "intraop_colloid": 500.0, "intraop_rbc": 2.0, "intraop_ffp": 2.0,
                "intraop_ftn": 100.0, "intraop_mdz": 5.0, "intraop_vecu": 10.0, "intraop_rocu": 50.0,
                "intraop_eph": 10.0, "intraop_epi": 0.1, "intraop_phe": 0.5, "intraop_ppf": 150.0, "intraop_ca": 0.5,
                "postop_hb": 9.5, "postop_wbc": 14.0, "postop_cr": 1.8, "cardiac_risk": 4.0, 
                "renal_function_status": "CKD Stage 3", "liver_function_status": "Normal", 
                "preop_diabetes": "Y", "preop_htn": "Y", "preop_copd": "N", "preop_chf": "Y", 
                "ventilation_mode": "SIMV", "muscle_relaxant": "Vecuronium", "opioid_used": "Morphine", 
                "antibiotic_prophylaxis": "Vancomycin", "surgeon_id": "S105",
                "anesthesiologist_id": "A208", "room_temp": 23.0, "preop_inr": 1.2, "preop_fibrinogen": 250.0,
                "intraop_temp_min": 35.0, "intraop_temp_max": 37.0
            }
        }

app = FastAPI(
    title="ICU Stay Duration Prediction API",
    description="A microservice for predicting ICU stay duration category.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    if model_service.model and model_service.preprocessor:
        return {"message": "ICU Stay Duration Predictor is fully operational.", "status": "Ready"}
    else:
        raise HTTPException(status_code=503, detail="Model service is down. Artifacts failed to load on startup.")


@app.post("/predict")
def predict_icu_stay(data: ICUData):
    if not model_service.model or not model_service.preprocessor:
        raise HTTPException(
            status_code=503, 
            detail="Model service is not ready. Artifacts failed to load."
        )
    
    try:
        raw_input_dict = data.model_dump(exclude_none=True)

        processed_features = model_service.preprocess_input(raw_input_dict)

        prediction_label = model_service.predict(processed_features)

        return {
            "status": "success",
            "prediction": prediction_label,
            "description": f"The predicted ICU stay duration is: {prediction_label}.",
            "input_summary": {
                "age": raw_input_dict.get('age'),
                "sex": raw_input_dict.get('sex'),
                "preop_hb": raw_input_dict.get('preop_hb'),
                "intraop_ebl": raw_input_dict.get('intraop_ebl'),
            }
        }

    except Exception as e:
        print(f"Prediction failed due to internal error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: An error occurred during prediction. Error: {e}")
