from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from joblib import load
from mangum import Mangum

import mlmodel as ml

outfn = "test.log"

class UserData(BaseModel):
  DVPercent: float
  DA: float
  D1: float
  D2: float
  TodayBstar: float
  PrevBstar: float
  TodayMeanAlt: float
  PrevMeanAlt: float

# Initialize the FastAPI application
app = FastAPI()
# Wrap the app with Mangum to enable Lambda compatibility
handler = Mangum(app)

@app.on_event('startup')
async def load_model():
  ml.inferengine = load("gmm_pipeline_20250621_20250621.joblib")

# Define a simple GET endpoint
@app.get("/")
async def hello_world():
  return {"message": "YSPACE - AWS Lambda + FastAPI for satellite anomaly checker"}

# Define the user API
@app.post("/anomaly/")
async def get_anomaly_probability(InData: UserData):
  X = pd.DataFrame(np.array([[InData.DVPercent, InData.DA, InData.D1, InData.D2]]),\
  columns=["DV_percent", "DA", "D1", "D2"])
  bstar_diff = np.abs(InData.TodayBstar - InData.PrevBstar)
  X["rel_Bstar_change"] = np.log(bstar_diff)
  X["rel_MeanAlt_change"] = np.abs((InData.TodayMeanAlt- InData.PrevMeanAlt) / InData.TodayMeanAlt)
  X = X.replace([np.inf, -np.inf], np.nan)
  return_flag = ml.inferengine['gmm'].predict(X)
  return_flag = "%d" % (int(return_flag[0]))
  return {"anomaly_flag": return_flag}
