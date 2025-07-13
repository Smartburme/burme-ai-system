from fastapi import FastAPI
from pydantic import BaseModel
from model import MyanmarXLMRoberta
from preprocessing import MyanmarTextPreprocessor
from config.settings import Config

app = FastAPI()
config = Config()
model = MyanmarXLMRoberta(config)
preprocessor = MyanmarTextPreprocessor()

class RequestData(BaseModel):
    text: str
    lang: str = "my"

@app.post("/predict")
async def predict(data: RequestData):
    cleaned_text = preprocessor.clean_text(data.text)
    inputs = preprocessor.tokenize(cleaned_text)
    with torch.no_grad():
        outputs = model(**inputs)
    return {"prediction": outputs.argmax().item()}
