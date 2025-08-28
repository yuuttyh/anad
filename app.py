from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Inicializa o FastAPI
app = FastAPI()

# Carrega o modelo DistilGPT2
generator = pipeline("text-generation", model="distilgpt2")

# Define o formato de entrada
class Prompt(BaseModel):
    text: str
    max_length: int = 50

@app.post("/generate")
def generate_text(prompt: Prompt):
    try:
        response = generator(prompt.text, max_length=prompt.max_length, num_return_sequences=1)
        return {"response": response[0]['generated_text']}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "API DistilGPT2 está rodando!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
