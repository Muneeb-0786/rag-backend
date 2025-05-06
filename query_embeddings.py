from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
import os
import argparse
from typing import List, Tuple
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import textwrap
import numpy as np
try:
    import google.generativeai as genai
    from google.api_core.exceptions import InvalidArgument
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

# Configure API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Missing COHERE_API_KEY environment variable")

# Google API Key for Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_AVAILABLE and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# LLM configuration
LLM_MODEL = 'command-r'  # Cohere's model
TEMPERATURE = 0.0

# Set API key
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# Utility function for wrapping text
def word_wrap(text, width=80):
    """Wrap text to specified width"""
    return "\n".join(textwrap.wrap(text, width=width))

# === Step 2: Query Functions ===
def augment_query_generated(query, model="gemini-2.0-flash", num_variations=2):
    """Generate augmented queries using Gemini to help with retrieval"""
    if not GEMINI_AVAILABLE:
        print("Warning: Google Generative AI package not available. Skipping query augmentation.")
        return [query]
    
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not set. Skipping query augmentation.")
        return [query]
    
    try:
        system_prompt = """You are a helpful expert research assistant. 
        
        Given a user question about information in documents, please generate {num_variations} alternative ways to ask this question.
        
        These alternative formulations should:
        1. Capture different potential ways the information might be described in documents
        2. Use domain-specific terminology that might appear in the document
        3. Consider different aspects or angles of the original question
        4. Include potential keywords that would help in document retrieval
        
        Format your response as a numbered list with each query on a separate line.
        """
        
        gemini = genai.GenerativeModel(model)
        response = gemini.generate_content(
            [
                system_prompt.format(num_variations=num_variations),
                query
            ]
        )
        
        content = response.text
        # Parse the numbered list into separate queries
        augmented_queries = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() and '. ' in line):
                augmented_queries.append(line.split('. ', 1)[1])
        
        # Ensure we have at least one augmented query
        if not augmented_queries:
            augmented_queries = [content]
            
        return augmented_queries
    except Exception as e:
        print(f"Error generating augmented queries: {e}")
        return [query]  # Return original query if there's an error


def retrieve_and_rerank_documents(db, query, n_results=5, max_return=5, use_augmentation=True, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Retrieve relevant documents for a given query using multiple query expansion and reranking"""
    if use_augmentation:
        # Generate augmented queries
        augmented_queries = augment_query_generated(query)
        
        # Create a list of queries including the original
        queries = [query] + augmented_queries
        
        print("\nUsing the following queries for retrieval:")
        for i, q in enumerate(queries):
            print(f"Query {i+1}: {word_wrap(q)}")
        print('-'*50)
    else:
        queries = [query]
        print("\nUsing original query for retrieval.")
    
    # Get results for all queries
    results = []
    doc_objects = []
    for q in queries:
        docs = db.similarity_search(q, k=n_results)
        doc_objects.extend(docs)
        results.append([doc.page_content for doc in docs])
    
    # Deduplicate the retrieved documents
    unique_documents = []
    unique_doc_objects = []
    seen_content = set()
    
    for doc in doc_objects:
        if doc.page_content not in seen_content:
            unique_documents.append(doc.page_content)
            unique_doc_objects.append(doc)
            seen_content.add(doc.page_content)
   
    if CROSS_ENCODER_AVAILABLE:
        # Load cross-encoder for reranking
        try:
            cross_encoder = CrossEncoder(model)
            
            # Create pairs for cross-encoder scoring
            pairs = []
            for doc in unique_documents:
                pairs.append([query, doc])

            # Calculate cross-encoder scores
            scores = cross_encoder.predict(pairs)

            print("\nCross-encoder scores:")
            for i, score in enumerate(scores):
                print(f"Document {i+1}: {score:.4f}")

            # Get indices sorted by score in descending order
            ranked_indices = np.argsort(scores)[::-1]
            
            print("\nReranked order:")
            for i, idx in enumerate(ranked_indices[:max_return]):
                print(f"Rank {i+1}: Document index {idx+1} (Score: {scores[idx]:.4f})")

            # Rerank the documents and limit to max_return
            reranked_doc_objects = [unique_doc_objects[i] for i in ranked_indices[:max_return]]
            print(f"\nReturning top {max_return} documents based on reranking.")
            
            return reranked_doc_objects
        except Exception as e:
            print(f"Error during reranking: {e}")
            print("Falling back to original retrieval order.")
            return unique_doc_objects[:max_return]
    else:
        print("\nCross-encoder not available. Skipping reranking.")
        return unique_doc_objects[:max_return]


def load_from_index(
    index_path: str, 
    retrieval_type: str = "similarity", 
    chain_type: str = "stuff", 
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    use_compression: bool = False,
    use_reranking: bool = False,
    use_augmentation: bool = False
):
    print(f"Loading index from: {index_path}")
    
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    search_kwargs = {"k": k}
    if retrieval_type == "mmr":
        if fetch_k <= k:
            fetch_k = k * 2
        search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})
        print(f"Using MMR retrieval with k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
    elif retrieval_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = 0.5
        print(f"Using similarity with score threshold=0.5, k={k}")
    else:
        print(f"Using standard similarity search with k={k}")
        
    llm = ChatCohere(model=LLM_MODEL, temperature=TEMPERATURE)
    
    system_content = """You are an intelligent assistant that provides accurate, well-cited answers based on the provided context. also you are allowed to add ans from the internet also mention it that its from the internet.
You are not allowed to make up information or provide false information and also instead of using check source you can use the following format to mention the source of the information you are providing.

Always cite the source using the format [source_name, page_number, line_number] when referencing information from the provided context."""
    
    human_template = """Given the following context:
{context}

Please answer the following question:
{question}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", human_template),
    ])
    
    if use_compression:
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=db.as_retriever(search_type=retrieval_type, search_kwargs=search_kwargs)
        )
        print("Using compression retriever")
    else:
        retriever = db.as_retriever(search_type=retrieval_type, search_kwargs=search_kwargs)
        print("Using standard retriever")

    if use_reranking or use_augmentation:
        print(f"Using {'augmented queries and ' if use_augmentation else ''}{'reranking' if use_reranking else ''}")
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        return_generated_question=True,
    )
    
    return qa_chain, db

class IndexChat:
    def __init__(
        self, 
        index_path: str, 
        retrieval_type: str = "similarity", 
        chain_type: str = "stuff",
        k: int = 4, 
        fetch_k: int = 20, 
        lambda_mult: float = 0.5,
        use_compression: bool = False,
        use_reranking: bool = False,
        use_augmentation: bool = False,
        n_results: int = 10,
        max_return: int = 5
    ):
        self.index_path = index_path
        self.retrieval_type = retrieval_type
        self.chain_type = chain_type
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.use_compression = use_compression
        self.use_reranking = use_reranking
        self.use_augmentation = use_augmentation
        self.n_results = n_results
        self.max_return = max_return
        self.chat_history = []
        
        self.qa, self.db = load_from_index(
            index_path, 
            retrieval_type=retrieval_type, 
            chain_type=chain_type,
            k=k, 
            fetch_k=fetch_k, 
            lambda_mult=lambda_mult,
            use_compression=use_compression,
            use_reranking=use_reranking,
            use_augmentation=use_augmentation
        )
    
    def ask(self, question: str) -> Tuple[str, str, List]:
        if not question:
            return "", "", []
        
        # If using reranking or augmentation, use our custom retrieval
        if self.use_reranking or self.use_augmentation:
            try:
                # Get custom retrieved documents
                source_documents = retrieve_and_rerank_documents(
                    db=self.db, 
                    query=question, 
                    n_results=self.n_results, 
                    max_return=self.max_return,
                    use_augmentation=self.use_augmentation
                )
                
                # Use the returned documents with the QA chain
                result = self.qa({"question": question, "chat_history": self.chat_history})
                answer = result["answer"]
                generated_question = result.get("generated_question", question)
                
                self.chat_history.append((question, answer))
                if len(self.chat_history) > 5:
                    self.chat_history = self.chat_history[-5:]
                
                return answer, generated_question, source_documents
            except Exception as e:
                print(f"Error in custom retrieval: {e}")
                print("Falling back to standard retrieval")
        
        # Standard retrieval path
        result = self.qa({"question": question, "chat_history": self.chat_history})
        answer = result["answer"]
        generated_question = result.get("generated_question", question)
        source_documents = result["source_documents"]
        
        self.chat_history.append((question, answer))
        if len(self.chat_history) > 5:
            self.chat_history = self.chat_history[-5:]
        
        return answer, generated_question, source_documents

    def reset_chat(self):
        self.chat_history = []
        return "Chat history has been reset."

def display_sources(source_docs, detailed: bool = False):
    if not source_docs:
        print("No sources found.")
        return
    
    print("\n" + "="*50)
    print("SOURCES:")
    print("="*50)
    for i, doc in enumerate(source_docs, 1):
        print(f"Source {i}:")
        if detailed:
            print(f"  Content: {doc.page_content}")
        else:
            print(f"  Content: {doc.page_content[:150]}...")
        print(f"  Metadata: {doc.metadata}")
        print("-"*40)

def main():
    parser = argparse.ArgumentParser(description='Query document from saved embeddings')
    parser.add_argument('--index', '-i', type=str, help='Path to the saved index directory')
    parser.add_argument('--retrieval', '-r', type=str, default="similarity", 
                        choices=["similarity", "mmr", "similarity_score_threshold"],
                        help='Type of retrieval to use')
    parser.add_argument('--chain-type', '-c', type=str, default="stuff",
                        choices=["stuff", "map_reduce"],
                        help='Chain type to use')
    parser.add_argument('--k', '-k', type=int, default=4,
                        help='Number of documents to retrieve')
    parser.add_argument('--fetch-k', type=int, default=20,
                        help='Number of documents to fetch for MMR (must be > k)')
    parser.add_argument('--lambda-mult', type=float, default=0.5,
                        help='Diversity factor for MMR (0-1, higher = more diverse)')
    parser.add_argument('--use-compression', '-u', action='store_true',
                        help='Use contextual compression for retrieval')
    parser.add_argument('--detailed-sources', '-d', action='store_true',
                        help='Show full source document content')
    parser.add_argument('--use-reranking', action='store_true',
                        help='Use cross-encoder reranking for retrieval')
    parser.add_argument('--use-augmentation', action='store_true',
                        help='Use query augmentation for improved retrieval')
    parser.add_argument('--n-results', type=int, default=10,
                        help='Number of initial results to retrieve for reranking')
    parser.add_argument('--max-return', type=int, default=5,
                        help='Maximum number of documents to return after reranking')
    
    args = parser.parse_args()
    
    index_path = args.index
    if not index_path:
        default_dir = "indexed_docs"
        if os.path.exists(default_dir) and os.listdir(default_dir):
            indices = os.listdir(default_dir)
            print("Available indices:")
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {idx}")
            
            selection = input("Select index number (or enter full path): ")
            try:
                index_num = int(selection)
                if 1 <= index_num <= len(indices):
                    index_path = os.path.join(default_dir, indices[index_num-1])
                else:
                    print("Invalid selection.")
                    return
            except ValueError:
                index_path = selection
        else:
            index_path = input("Enter path to saved index directory: ")
    
    if not os.path.exists(index_path):
        print(f"Error: Index directory {index_path} does not exist.")
        print("You need to create embeddings first using create_embeddings.py")
        return
    
    chat = IndexChat(
        index_path, 
        retrieval_type=args.retrieval, 
        chain_type=args.chain_type,
        k=args.k, 
        fetch_k=args.fetch_k, 
        lambda_mult=args.lambda_mult,
        use_compression=args.use_compression,
        use_reranking=args.use_reranking,
        use_augmentation=args.use_augmentation,
        n_results=args.n_results,
        max_return=args.max_return
    )

    print(f"\nLoaded index: {chat.index_path}")
    print(f"Retrieval method: {chat.retrieval_type}, Chain type: {chat.chain_type}, k={chat.k}")
    if chat.retrieval_type == "mmr":
        print(f"MMR settings: fetch_k={chat.fetch_k}, lambda_mult={chat.lambda_mult}")
    if chat.use_reranking:
        print(f"Using reranking with n_results={chat.n_results}, max_return={chat.max_return}")
    if chat.use_augmentation:
        print("Using query augmentation for improved retrieval")
    
    print("\nCommands:")
    print("  'exit' - Quit the application")
    print("  'sources' - Show sources from last question")
    print("  'reset' - Reset chat history")
    
    last_sources = []
    
    while True:
        query = input("\nYou: ").strip()
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'sources':
            display_sources(last_sources, detailed=args.detailed_sources)
            continue
        elif query.lower() == 'reset':
            result = chat.reset_chat()
            print(f"\nSystem: {result}")
            continue
            
        answer, db_query, sources = chat.ask(query)
        last_sources = sources
        
        print("\nAI:", answer)
        print("\nGenerated query:", db_query)
        print(f"\n[Found {len(sources)} relevant sources. Type 'sources' to view them.]")

if __name__ == "__main__":
    main()
