import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import shutil
from dotenv import load_dotenv

# Import from rag_full_pipeline
from rag_full_pipeline import (
    transcribe_audio,
    chunk_text,
    embed_and_store,
    answer_question,
    TRANSCRIPTIONS_DIRECTORY,
    AUDIO_DIRECTORY
)

load_dotenv()

app = FastAPI(title="Audio RAG API", version="1.0")

# Response models
class QueryResponse(BaseModel):
    answer: str

# Global state
vector_db = None
transcriptions_dict = {}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and process audio file"""
    global vector_db, transcriptions_dict
    
    # Create directories
    os.makedirs(AUDIO_DIRECTORY, exist_ok=True)
    
    # Save file
    audio_path = os.path.join(AUDIO_DIRECTORY, file.filename)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Transcribe
        transcription = transcribe_audio(audio_path, file.filename)
        if not transcription:
            raise HTTPException(500, "Transcription failed")
        
        # Store transcription
        transcriptions_dict[file.filename] = transcription
        
        # Create chunks with metadata
        chunks = chunk_text(transcription, file.filename)
        
        # Rebuild vector DB with all files
        all_chunks = []
        for filename, text in transcriptions_dict.items():
            file_chunks = chunk_text(text, filename)
            all_chunks.extend(file_chunks)
        
        vector_db = embed_and_store(all_chunks)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/query", response_model=QueryResponse)
async def query_rag(question: str):
    """Query the RAG system"""
    global vector_db, transcriptions_dict
    
    if not vector_db:
        raise HTTPException(400, "Upload an audio file first")
    
    answer = answer_question(vector_db, question, transcriptions_dict)
    return QueryResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
