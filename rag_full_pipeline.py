import os
import re
import shutil
from faster_whisper import WhisperModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from timer import timer_decorator

# Configuration
AUDIO_FILE_PATH = "jeffrey_interview.mp3"

                                   
TRANSCRIPTION_FILE = "transcription.txt"
PERSIST_DIRECTORY = "./my_rag_db"
COLLECTION_NAME = "rag_collection"
MODEL_SIZE = 'small'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

@timer_decorator
def transcribe_audio(audio_path):
    """
    Transcribes audio file to text using Faster Whisper.
    """
    # Check if transcription already exists
    if os.path.exists(TRANSCRIPTION_FILE):
        print(f"Content found in '{TRANSCRIPTION_FILE}'. Using existing transcription.")
        print("To re-transcribe, please delete this file.")
        with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
            return f.read()

    print(f"--- Starting Transcription for {audio_path} ---")
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return None

    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Detected language: {info.language} (Probability: {info.language_probability:.2f})")

    output_lines = []
    for segment in segments:
        #string with timestamps
        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        print(line)
        output_lines.append(line)


    complete_text = "\n".join(output_lines)

    with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
        f.write(complete_text)
    
    print(f"Transcription saved to '{TRANSCRIPTION_FILE}'.")
    return complete_text

@timer_decorator
def chunk_text(text_content):
    """
    Splits the raw text into smaller, manageable chunks.
    """
    print("--- Starting Chunking Process ---")

    # Intelligent Chunking with increased overlap for better context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Characters per chunk
        chunk_overlap=150,    # Increased overlap for better context preservation
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Create Document objects
    documents = [Document(page_content=x) for x in splitter.split_text(text_content)]

    print(f"Chunking complete. Created {len(documents)} document chunks.")
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

def answer_question(db, query, full_text=""):
    """
    Answers a single question using hybrid search (semantic + keyword).
    Returns the answer as a formatted string.
    """
    RELEVANCE_THRESHOLD = 0.9  # Optimized based on testing (lower = stricter)
    
    # Perform similarity search with scores
    results_with_scores = db.similarity_search_with_score(query, k=5)
    relevant_results = [(doc, score) for doc, score in results_with_scores if score <= RELEVANCE_THRESHOLD]
    
    if relevant_results:
        # Found relevant semantic matches
        answer_parts = [f"\n Answer (found {len(relevant_results)} relevant matches):\n"]
        for i, (doc, score) in enumerate(relevant_results[:3], 1):
            answer_parts.append(f"\n[Match {i}] (Relevance: {score:.4f})")
            answer_parts.append(doc.page_content)
            answer_parts.append("-" * 80)
        return "\n".join(answer_parts)
    else:
        # Try keyword fallback
        if full_text:
            keyword_results = keyword_search(full_text, query, top_k=3)
            if keyword_results:
                answer_parts = [f"\n Answer (found via keyword search):\n"]
                for i, chunk in enumerate(keyword_results, 1):
                    answer_parts.append(f"\n[Match {i}]")
                    answer_parts.append(chunk)
                    answer_parts.append("-" * 80)
                return "\n".join(answer_parts)
        
        return "\n I don't know about that. No relevant information found in the audio.\n"

@timer_decorator
def interactive_qa(db):
    """
    Interactive Q&A session - allows users to keep asking questions.
    """
    print("=" * 80)
    print("Ask questions about the audio transcription.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    # Load transcription for keyword search fallback
    if os.path.exists(TRANSCRIPTION_FILE):
        with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        full_text = ""
        print("️  Warning: Transcription file not found. Keyword fallback disabled.\n")
    
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
            answer = answer_question(db, query, full_text)
            print(answer)
            
        except KeyboardInterrupt:
            print(f"\n\n Session interrupted. You asked {question_count} question(s). Goodbye!")
            break
        except Exception as e:
            print(f"\n Error processing question: {e}")
            continue

def main():
    print(f"Using audio file: {AUDIO_FILE_PATH}")

    # Step 1: Transcription
    transcription_text = transcribe_audio(AUDIO_FILE_PATH)
    
    if not transcription_text:
         # If STT failed or file not found, try reading existing transcription.txt
        if os.path.exists(TRANSCRIPTION_FILE):
            print(f"STT failed or skipped, reading from existing '{TRANSCRIPTION_FILE}'...")
            with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
                transcription_text = f.read()
        else:
            print("Aborting: No transcription text available.")
            return

    # Step 2: Chunking
    doc_chunks = chunk_text(transcription_text)

    # Step 3: Embedding
    vector_db = embed_and_store(doc_chunks)

    # Step 4:  Q&A
    interactive_qa(vector_db)

if __name__ == "__main__":
    main()
