from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any

# --- This input model is kept for documentation (Swagger UI) ---
# It ensures the API documentation shows the user how to structure the input JSON (all 70+ fields).
class ICUData(BaseModel):
    # Mandatory Fields (based on your schema)
    sex: Literal['M', 'F', 'nan', 'MALE', 'FEMALE']
    age: str
    department: str
    approach: str
    position: str
    ane_type: str
    iv1: str
    
    # Example Optional/Complex Fields (A representative subset of your 70+ fields)
    caseid: Optional[str] = None
    height: Optional[float] = None
    bmi: Optional[float] = Field(None, description="Body Mass Index (BMI).") 
    preop_hb: Optional[float] = None
    preop_wbc: Optional[float] = None
    preop_alb: Optional[float] = None 
    preop_ecg: Optional[str] = None
    preop_pft: Optional[str] = None
    emop: Optional[Literal['Y', 'N']] = Field(None, description="Emergency operation status.") 
    intraop_ebl: Optional[float] = None
    
    # Allow extra fields for schema completeness in the demonstration
    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "age": "70", "sex": "F", "department": "Cardio", "approach": "Minimally Invasive", 
                "position": "Lateral", "ane_type": "General", "iv1": "Ringer Lactate", 
                "preop_hb": 10.5, "preop_wbc": 12.1, "preop_alb": 3.0, "emop": "N",
                "preop_ecg": "Atrial fibrillation, controlled rate", "intraop_ebl": 500.0,
                # Add 50-60 more fields here for the presentation demo's input realism
            }
        }


app = FastAPI(
    title="ICU Stay Duration Prediction API (DEMO MODE)",
    description="This endpoint is set to return a fixed output for presentation purposes, bypassing model failures.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "API is running in DEMO MODE (Model Disabled)."}


@app.post("/predict")
def predict_icu_stay(data: ICUData):
    """
    Returns a fixed, successful prediction response for presentation demonstration.
    The complex input data is validated but not used for prediction calculation.
    """
    # Fixed Output for Presentation
    return {
        "status": "success",
        "prediction_class": "1 Day",
        "confidence": 0.91,
        "model_status": "DISABLED - DEMO MODE",
        "justification": "Fixed output provided to showcase successful architecture deployment and data I/O."
    }
