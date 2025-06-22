from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import Phi4, Qwen2_5

phi4 = Phi4('microsoft/Phi-4-mini-instruct')
coder = Qwen2_5('Qwen/Qwen2.5-Coder-3B-Instruct')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

class ChatRequest(BaseModel):
    instruction: str
    summary: dict

@app.post("/file")
async def analyze_file(request: CodeRequest):
    code = request.code
    desc = coder.desc_file(code)

    return {
        "source": code,
        "message": desc
    }

@app.post("/func")
async def analyze_func(request: CodeRequest):
    code = request.code
    desc = coder.desc_func(code)

    return {
        "source": code,
        "message": desc
    }

@app.post("/class")
async def analyze_class(request: CodeRequest):
    code = request.code
    desc = coder.desc_class(code)

    return {
        "source": code,
        "message": desc
    }

@app.post("/readme")
async def analyze_readme(request: CodeRequest):
    code = request.code
    desc = phi4.desc_readme(code)

    return {
        "source": code,
        "message": desc
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    inst = request.instruction
    summary = request.summary

    answer = coder.chat(inst, summary)
    
    return {
        "message": answer
    }