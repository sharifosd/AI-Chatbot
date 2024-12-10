from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
from typing import Optional

app = FastAPI(
    title="University FAQ Chatbot API",
    description="API for university-related questions in English and Bengali",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading
print("Loading model and tokenizer...")
MODEL_PATH = "university_faq_model"

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True
    )
    
    # Convert model for inference
    model = FastLanguageModel.for_inference(model)
    
    # Setup tokenizer with chat template
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class Question(BaseModel):
    text: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class Response(BaseModel):
    answer: str
    error: Optional[str] = None

def clean_response(response: str) -> str:
    """Clean the model response by removing system and user prompts."""
    try:
        # Extract the assistant's response
        assistant_start = response.find("assistant\n\n") + len("assistant\n\n")
        cleaned_response = response[assistant_start:].strip()
        return cleaned_response
    except Exception:
        return response

def generate_response(question: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    try:
        # Format the conversation using the chat template
        conversation = [
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize the formatted prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_response(response)
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "University FAQ Chatbot API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/ask", response_model=Response)
async def ask_question(question: Question):
    try:
        answer = generate_response(
            question.text,
            temperature=question.temperature,
            max_tokens=question.max_tokens
        )
        return Response(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)