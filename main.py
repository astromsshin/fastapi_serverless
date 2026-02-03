from fastapi import FastAPI
from mangum import Mangum

# Initialize the FastAPI application
app = FastAPI()
# Wrap the app with Mangum to enable Lambda compatibility
handler = Mangum(app)

# Define a simple GET endpoint
@app.get("/")
async def hello_world():
  return {"message": "YSPACE - AWS Lambda + FastAPI for satellite anomaly checker"}
