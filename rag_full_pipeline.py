import os
import re
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from timer import timer_decorator
import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Configuration
AUDIO_DIRECTORY = "./audio_files"  # Directory containing audio files
TRANSCRIPTIONS_DIRECTORY = "./transcriptions"  # Directory for transcriptions
PERSIST_DIRECTORY = "./my_rag_db"
COLLECTION_NAME = "rag_collection"
MODEL_SIZE = 'small'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables. "
        "Please create a .env file with GEMINI_API_KEY=your-key-here"
    )
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # Latest free-tier model

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. "
        "Please create a .env file with GROQ_API_KEY=your-key-here"
    )

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Initialize Groq
groq_client = Groq(api_key=GROQ_API_KEY)

@timer_decorator
def transcribe_audio(audio_path, audio_filename):
    """
    Transcribes audio file to text using Groq Whisper API.
    Returns transcription text.
    """
    # Create transcriptions directory if it doesn't exist
    os.makedirs(TRANSCRIPTIONS_DIRECTORY, exist_ok=True)
    
    # Transcription file based on audio filename
    transcription_file = os.path.join(
        TRANSCRIPTIONS_DIRECTORY, 
        f"{os.path.splitext(audio_filename)[0]}.txt"
    )
    
    # Check if transcription already exists
    if os.path.exists(transcription_file):
        print(f"  ✓ Using existing transcription: {audio_filename}")
        with open(transcription_file, "r", encoding="utf-8") as f:
            return f.read()

    print(f"  🎙️  Transcribing via Groq API: {audio_filename}")
    
    if not os.path.exists(audio_path):
        print(f"  ❌ Error: Audio file '{audio_path}' not found.")
        return None

    try:
        # Open audio file
        with open(audio_path, "rb") as audio_file:
            # Use Groq Whisper API for transcription
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_filename, audio_file.read()),
                model="whisper-large-v3-turbo",  # Groq's fastest Whisper model
                response_format="verbose_json",  # Get detailed output with timestamps
                temperature=0.0
            )
        
        # Extract segments with timestamps
        output_lines = []
        if hasattr(transcription, 'segments') and transcription.segments:
            for segment in transcription.segments:
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')
                line = f"[{start:.2f}s -> {end:.2f}s] {text}"
                output_lines.append(line)
        else:
            # Fallback if no segments (shouldn't happen with verbose_json)
            output_lines.append(f"[0.00s -> 0.00s] {transcription.text}")
        
        complete_text = "\n".join(output_lines)
        
        # Save transcription
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write(complete_text)
        
        print(f"  ✓ Saved: {transcription_file}")
        return complete_text
        
    except Exception as e:
        print(f"  ❌ Transcription failed: {e}")
        return None

@timer_decorator
def chunk_text(text_content, source_filename):
    """
    Splits the raw text into smaller chunks with source file metadata.
    """
    # Intelligent Chunking with increased overlap for better context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Characters per chunk
        chunk_overlap=150,    # Increased overlap for better context preservation
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Create Document objects with metadata
    documents = []
    for chunk in splitter.split_text(text_content):
        timestamp = extract_timestamp(chunk)
        documents.append(Document(
            page_content=chunk,
            metadata={
                "source_file": source_filename,
                "timestamp": timestamp if timestamp else "unknown"
            }
        ))

    return documents

@timer_decorator
def embed_and_store(documents):
    """
    Embeds documents and stores them in a Chroma vector database.
    """
    print("--- Starting Embedding and Storage Process ---")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print("Removed old database to ensure a clean start.")

    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )

    print(f"Success! Vectors stored locally in '{PERSIST_DIRECTORY}'.")
    return db

def keyword_search(text, query, top_k=3):
    """
    Perform keyword-based search for named entities
    Returns chunks containing query keywords
    """
    # Extract potential keywords (remove common words)
    common_words = {'what', 'who', 'where', 'when', 'how', 'did', 'was', 'were', 
                    'is', 'are', 'the', 'a', 'an', 'mention', 'happen', 'start', 'about'}
    
    query_lower = query.lower()
    keywords = [word for word in re.findall(r'\b\w+\b', query_lower) 
                if word not in common_words and len(word) > 2]
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    # Score each chunk
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for keyword in keywords if keyword in chunk_lower)
        if score > 0:
            scored_chunks.append((chunk, score))
    
    # Sort by score and return top k
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]

def extract_timestamp(chunk):
    """
    Extract timestamp from chunk text.
    Format: [0.00s -> 5.23s] text
    Returns the start timestamp as a string or None
    """
    import re
    match = re.match(r'\[(\d+\.\d+)s\s*->\s*(\d+\.\d+)s\]', chunk)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        # Format as MM:SS
        start_minutes = int(start_time // 60)
        start_seconds = int(start_time % 60)
        return f"{start_minutes}:{start_seconds:02d}"
    return None

def answer_question(db, query, full_text_dict=None):
    """
    Answers a single question using hybrid search (semantic + keyword) and Gemini LLM.
    Returns the answer with source file and timestamp citations.
    full_text_dict: dict mapping filename -> full text for keyword fallback
    """
    RELEVANCE_THRESHOLD = 0.9  # Optimized based on testing (lower = stricter)
    
    # Perform similarity search with scores
    results_with_scores = db.similarity_search_with_score(query, k=5)
    relevant_results = [(doc, score) for doc, score in results_with_scores if score <= RELEVANCE_THRESHOLD]
    
    context_chunks = []
    documents = []
    search_method = None
    
    if relevant_results:
        # Found relevant semantic matches
        context_chunks = [doc.page_content for doc, score in relevant_results[:3]]
        documents = [doc for doc, score in relevant_results[:3]]
        search_method = "semantic"
    else:
        # Try keyword fallback - search across all transcriptions
        if full_text_dict:
            keyword_matches = []
            for filename, text in full_text_dict.items():
                results = keyword_search(text, query, top_k=2)
                for result in results:
                    timestamp = extract_timestamp(result)
                    keyword_matches.append({
                        "content": result,
                        "filename": filename,
                        "timestamp": timestamp
                    })
            
            if keyword_matches:
                # Take top 3 keyword matches
                keyword_matches = keyword_matches[:3]
                context_chunks = [m["content"] for m in keyword_matches]
                # Create pseudo-documents for keyword results
                documents = [Document(
                    page_content=m["content"],
                    metadata={
                        "source_file": m["filename"],
                        "timestamp": m["timestamp"] or "unknown"
                    }
                ) for m in keyword_matches]
                search_method = "keyword"
    
    # If no relevant context found, return "I don't know"
    if not context_chunks:
        return "\n💡 I don't know about that. No relevant information found in the audio.\n"
    
    # Extract source citations from documents (file + timestamp)
    sources = {}
    for doc in documents:
        source_file = doc.metadata.get("source_file", "unknown")
        timestamp = doc.metadata.get("timestamp", "unknown")
        
        if source_file not in sources:
            sources[source_file] = []
        if timestamp != "unknown" and timestamp not in sources[source_file]:
            sources[source_file].append(timestamp)
    
    # Build context string
    context = "\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
    
    # Create prompt for Gemini
    prompt = f"""You are a helpful AI assistant answering questions based on audio transcription content.

Context from the audio transcription:
{context}

User Question: {query}

Instructions:
- Answer the question based ONLY on the provided context
- Be conversational and natural in your response
- If the context doesn't contain enough information to fully answer the question, say so
- Do not make up or assume information not present in the context
- Keep your answer concise and relevant

Answer:"""
    
    try:
        # Generate response using Gemini
        response = gemini_model.generate_content(prompt)
        
        # Format the response with source citations
        answer_parts = [f"\n🤖 Answer (via {search_method} search):\n"]
        answer_parts.append(response.text)
        
        # Add source citations (file + timestamps)
        if sources:
            answer_parts.append("\n" + "─" * 60)
            answer_parts.append("📁 Sources:")
            for filename, timestamps in sources.items():
                if timestamps:
                    ts_str = ", ".join(timestamps)
                    answer_parts.append(f"   • {filename} at {ts_str}")
                else:
                    answer_parts.append(f"   • {filename}")
        
        answer = "\n".join(answer_parts) + "\n"
        return answer
        
    except Exception as e:
        # Fallback to raw context if LLM fails
        print(f"⚠️  LLM generation failed: {e}")
        answer_parts = [f"\n📋 Answer (found via {search_method} search - LLM unavailable):\n"]
        for i, chunk in enumerate(context_chunks, 1):
            answer_parts.append(f"\n[Match {i}]")
        return "\n".join(answer_parts)

@timer_decorator
def interactive_qa(db, transcriptions_dict=None):
    """
    Interactive Q&A session - allows users to keep asking questions.
    transcriptions_dict: dict mapping filename -> full text for keyword search
    """
    print("=" * 80)
    print("Ask questions about the audio transcription(s).")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    question_count = 0
    
    while True:
        try:
            # Get user input
            query = input(f"\n[Q{question_count + 1}] Your question: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n Session ended. You asked {question_count} question(s). Goodbye!")
                break
            
            # Skip empty queries
            if not query:
                print("  Please enter a valid question.")
                continue
            
            question_count += 1
            
            # Get and display answer
            answer = answer_question(db, query, transcriptions_dict)
            print(answer)
            
        except KeyboardInterrupt:
            print(f"\n\n Session interrupted. You asked {question_count} question(s). Goodbye!")
            break
        except Exception as e:
            print(f"\n Error processing question: {e}")
            continue

def main():
    print("=" * 80)
    print("Multi-File Audio RAG Chatbot")
    print("=" * 80)
    
    # Get all audio files from directory
    audio_files = []
    if os.path.exists(AUDIO_DIRECTORY):
        for file in os.listdir(AUDIO_DIRECTORY):
            if file.endswith(('.mp3', '.wav', '.m4a')):
                audio_files.append(file)
    
    if not audio_files:
        print(f"❌ No audio files found in '{AUDIO_DIRECTORY}'")
        print(f"   Please add .mp3, .wav, or .m4a files to this directory.")
        return
    
    print(f"\n📂 Found {len(audio_files)} audio file(s):")
    for i, file in enumerate(audio_files, 1):
        print(f"   {i}. {file}")
    print()
    
    all_documents = []
    transcriptions_dict = {}
    
    # Process each audio file
    for audio_file in audio_files:
        audio_path = os.path.join(AUDIO_DIRECTORY, audio_file)
        print(f"Processing: {audio_file}")
        
        # Step 1: Transcription
        transcription_text = transcribe_audio(audio_path, audio_file)
        
        if not transcription_text:
            print(f"  ⚠️  Skipping {audio_file} - transcription failed\n")
            continue
        
        # Store transcription for keyword search
        transcriptions_dict[audio_file] = transcription_text
        
        # Step 2: Chunking with metadata
        doc_chunks = chunk_text(transcription_text, audio_file)
        all_documents.extend(doc_chunks)
        
        print(f"  ✓ Created {len(doc_chunks)} chunks\n")
    
    if not all_documents:
        print("❌ No documents to process. Exiting.")
        return
    
    print(f"📊 Total chunks across all files: {len(all_documents)}\n")
    
    # Step 3: Embedding
    print("Embedding and storing in vector database...")
    vector_db = embed_and_store(all_documents)

    # Step 4: Q&A
    interactive_qa(vector_db, transcriptions_dict)

if __name__ == "__main__":
    main()
