from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
import os
import argparse
import sys
from typing import List, Tuple, Dict
import time
import glob
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastapi import FastAPI, HTTPException, UploadFile, Form
import shutil

# Configure API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("Missing COHERE_API_KEY environment variable")

# Set Cohere API key
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# Check for Google API key but make it optional
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    print("Warning: GEMINI_API_KEY not found. Using Cohere embeddings only.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

app = FastAPI()

def get_document_loader(file_path):
    """Return appropriate document loader based on file extension."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return PyPDFLoader(file_path)
    elif file_ext == '.txt':
        return TextLoader(file_path)
    else:
        # Try to use UnstructuredFileLoader for other file types
        try:
            return UnstructuredFileLoader(file_path)
        except Exception as e:
            raise ValueError(f"Unsupported file type: {file_ext}. Error: {str(e)}")

def validate_chunk_params(chunk_size, chunk_overlap):
    """Validate chunk size and overlap parameters."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap must be a non-negative integer")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size")
    return True

def get_embeddings(use_gemini=True):
    """Return appropriate embeddings model based on availability."""
    if use_gemini and GOOGLE_API_KEY:
        print("Using Gemini embeddings")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        print("Using Cohere embeddings")
        return CohereEmbeddings(model="embed-english-v3.0")

def create_vectorstore(docs, embeddings, vector_db_type, save_path):
    """Create a vector store based on specified type."""
    if vector_db_type.lower() == "faiss":
        print(f"Creating FAISS vector database")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(save_path)
        return db, save_path
    elif vector_db_type.lower() == "chroma":
        print(f"Creating Chroma vector database")
        # Ensure path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        db = Chroma.from_documents(docs, embeddings, persist_directory=save_path)
        db.persist()
        return db, save_path
    else:
        raise ValueError(f"Unsupported vector database type: {vector_db_type}")

def process_document(file_path, output_dir="indexed_docs", chunk_size=1000, chunk_overlap=150, vector_db_type="faiss"):
    """Process document, create embeddings, and save the index."""
    start_time = time.time()
    
    try:
        print(f"Processing document: {file_path}")
        # Validate parameters
        validate_chunk_params(chunk_size, chunk_overlap)
        
        # Get base filename without extension for saving the index
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process the document
        try:
            loader = get_document_loader(file_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content extracted from document")
                
        except Exception as e:
            raise RuntimeError(f"Error loading document: {str(e)}")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            raise ValueError("Document splitting resulted in no chunks")
            
        print(f"Split into {len(docs)} chunks")
        
        # Define embedding
        try:
            embeddings = get_embeddings(use_gemini=True)
            
            # Create vector database
            save_path = os.path.join(output_dir, file_name)
            db, save_path = create_vectorstore(docs, embeddings, vector_db_type, save_path)
            
        except Exception as e:
            raise RuntimeError(f"Error creating embeddings: {str(e)}")
        
        end_time = time.time()
        
        print(f"Index saved to {save_path}")
        print(f"Embedding creation completed in {end_time - start_time:.2f} seconds")
        
        return save_path
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return None

def process_multiple_documents(file_paths, output_dir="indexed_docs", chunk_size=1000, chunk_overlap=150, combined_name="combined_index", vector_db_type="faiss"):
    """Process multiple documents into a single index, create embeddings, and save the index."""
    start_time = time.time()
    
    try:
        print(f"Processing {len(file_paths)} documents into a combined index")
        # Validate parameters
        validate_chunk_params(chunk_size, chunk_overlap)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process all documents
        all_docs = []
        
        for file_path in file_paths:
            try:
                print(f"Loading document: {file_path}")
                loader = get_document_loader(file_path)
                documents = loader.load()
                
                if not documents:
                    print(f"Warning: No content extracted from {file_path}, skipping...")
                    continue
                    
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                docs = text_splitter.split_documents(documents)
                
                if not docs:
                    print(f"Warning: Document splitting resulted in no chunks for {file_path}, skipping...")
                    continue
                    
                print(f"Added {len(docs)} chunks from {file_path}")
                all_docs.extend(docs)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                print("Skipping this file but continuing with others...")
        
        if not all_docs:
            raise ValueError("No valid documents or chunks were found")
            
        print(f"Combined {len(all_docs)} total chunks from {len(file_paths)} documents")
        
        # Define embedding
        try:
            embeddings = get_embeddings(use_gemini=True)
            
            # Create vector database
            save_path = os.path.join(output_dir, combined_name)
            db, save_path = create_vectorstore(all_docs, embeddings, vector_db_type, save_path)
            
        except Exception as e:
            raise RuntimeError(f"Error creating embeddings: {str(e)}")
        
        end_time = time.time()
        
        print(f"Combined index saved to {save_path}")
        print(f"Combined embedding creation completed in {end_time - start_time:.2f} seconds")
        
        return save_path
        
    except Exception as e:
        print(f"Error processing combined documents: {str(e)}")
        return None

def find_files(paths, recursive=False):
    """Find all files from given paths, optionally recursively."""
    all_files = []
    for path in paths:
        if os.path.isfile(path):
            all_files.append(path)
        elif os.path.isdir(path):
            if recursive:
                # Recursively find all files in directory
                for root, _, files in os.walk(path):
                    for file in files:
                        all_files.append(os.path.join(root, file))
            else:
                # Only include files directly in the directory
                entries = [os.path.join(path, entry) for entry in os.listdir(path)]
                all_files.extend([f for f in entries if os.path.isfile(f)])
        elif '*' in path:
            # Handle glob patterns
            matches = glob.glob(path, recursive=recursive)
            all_files.extend([f for f in matches if os.path.isfile(f)])
    
    return all_files

@app.post("/process-document")
async def process_document_endpoint(
    file: UploadFile,
    output_dir: str = Form("indexed_docs"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150),
    vector_db_type: str = Form("faiss")
):
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        save_path = process_document(
            file_path=file_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_db_type=vector_db_type
        )
        if save_path:
            return {"message": f"Document processed successfully. Index saved at {save_path}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process document.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/process-multiple-documents")
async def process_multiple_documents_endpoint(
    files: List[UploadFile],
    output_dir: str = Form("indexed_docs"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150),
    combined_name: str = Form("combined_index"),
    vector_db_type: str = Form("faiss")
):
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_paths = []
        for file in files:
            file_path = os.path.join(output_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_paths.append(file_path)
        
        save_path = process_multiple_documents(
            file_paths=file_paths,
            output_dir=output_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            combined_name=combined_name,
            vector_db_type=vector_db_type
        )
        if save_path:
            return {"message": f"Documents processed successfully. Combined index saved at {save_path}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process documents.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.get("/list-indices")
async def list_indices(output_dir: str = "indexed_docs"):
    try:
        if not os.path.exists(output_dir):
            raise HTTPException(status_code=400, detail="Output directory does not exist.")
        
        indices = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))]
        return {"indices": indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing indices: {str(e)}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Create document embeddings and save index')
    parser.add_argument('--file', '-f', required=False, type=str, help='Path to the input file (Legacy option)')
    parser.add_argument('--files', nargs='+', help='Paths to multiple input files or directories')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively process directories')
    parser.add_argument('--output', '-o', type=str, default="indexed_docs", help='Output directory for saving index')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for text splitting')
    parser.add_argument('--chunk-overlap', type=int, default=150, help='Chunk overlap for text splitting')
    parser.add_argument('--combine', '-c', action='store_true', help='Combine all files into a single index')
    parser.add_argument('--combined-name', type=str, default="combined_index", help='Name for the combined index')
    parser.add_argument('--vector-db', type=str, choices=['faiss', 'chroma'], default='faiss', 
                       help='Vector database type to use (faiss or chroma)')
    
    args = parser.parse_args()
    
    # Handle the files to process
    files_to_process = []
    
    # Handle legacy --file argument
    if args.file:
        files_to_process.append(args.file)
    
    # Handle new --files argument
    if args.files:
        files_to_process = find_files(args.files, recursive=args.recursive)
    
    if not files_to_process:
        print("Error: No input files specified. Use --file or --files options.")
        return 1
    
    # Check if files exist
    files_to_process = [f for f in files_to_process if os.path.exists(f)]
    if not files_to_process:
        print("Error: None of the specified files exist.")
        return 1
    
    # Process in combined mode if requested
    if args.combine:
        print(f"Processing {len(files_to_process)} files in combined mode")
        save_path = process_multiple_documents(
            files_to_process,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            combined_name=args.combined_name,
            vector_db_type=args.vector_db
        )
        
        if save_path:
            print(f"\nSuccessfully created combined index at: {save_path}")
            print(f"Combined {len(files_to_process)} files into a single index")
            return 0
        else:
            print("\nFailed to create combined index")
            return 1
    
    # Otherwise, process files individually (original behavior)
    # Track processing results
    results = {
        'success': [],
        'failed': []
    }
    
    total_files = len(files_to_process)
    print(f"Starting to process {total_files} file(s) individually")
    
    # Process each file
    for i, file_path in enumerate(files_to_process):
        print(f"\n[{i+1}/{total_files}] Processing: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"  Error: File {file_path} does not exist.")
            results['failed'].append((file_path, "File does not exist"))
            continue
        
        try:
            save_path = process_document(
                file_path, 
                output_dir=args.output,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                vector_db_type=args.vector_db
            )
            
            if save_path:
                results['success'].append((file_path, save_path))
            else:
                results['failed'].append((file_path, "Processing failed"))
                
        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}")
            results['failed'].append((file_path, str(e)))
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files processed: {total_files}")
    print(f"Successfully processed: {len(results['success'])}")
    print(f"Failed to process: {len(results['failed'])}")
    
    if results['success']:
        print("\nSuccessfully processed files:")
        for file_path, save_path in results['success']:
            print(f"  - {file_path} → {save_path}")
    
    if results['failed']:
        print("\nFailed to process files:")
        for file_path, error in results['failed']:
            print(f"  - {file_path} ({error})")
    
    print("\nYou can now query these indices using query_embeddings.py")
    
    if results['failed']:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
