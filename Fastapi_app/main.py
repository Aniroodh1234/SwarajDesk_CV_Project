import os
from dotenv import load_dotenv
load_dotenv()

## Groq api key and setting hugging face environment
Groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")

## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="You_current_project_name"


from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .inference import predict_issue_hybrid

app = FastAPI(title="SwarajDesk_CV_API (Hybrid VLM + ViT)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "API Running", "mode": "hybrid_vlm_vit_20_sectors"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = predict_issue_hybrid(image_bytes)
    return result

