import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from rag_full_pipeline import transcribe_audio, chunk_text, embed_and_store, answer_question, TRANSCRIPTION_FILE

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    message: str
    chunks_created: int

db = None
full_text = ""

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    global db, full_text
    
    temp_path = f"temp_{file.filename}"
    
    try:
        # Save file
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Process: Transcribe -> Chunk -> Embed
        transcription = transcribe_audio(temp_path)
        chunks = chunk_text(transcription)
        db = embed_and_store(chunks)
        
        # Load transcription for queries
        if os.path.exists(TRANSCRIPTION_FILE):
            with open(TRANSCRIPTION_FILE, "r") as f:
                full_text = f.read()
        
        return UploadResponse(message="Success", chunks_created=len(chunks))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not db:
        raise HTTPException(400, "Upload a file first")
    
    answer = answer_question(db, request.question, full_text)
    return QueryResponse(answer=answer)
