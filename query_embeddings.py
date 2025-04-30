from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
import os
import argparse
from typing import List, Tuple
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Configure API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Missing COHERE_API_KEY environment variable")

# LLM configuration
LLM_MODEL = 'command-r'  # Cohere's model
TEMPERATURE = 0.0

# Set API key
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

def load_from_index(
    index_path: str, 
    retrieval_type: str = "similarity", 
    chain_type: str = "stuff", 
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    use_compression: bool = False
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

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        return_generated_question=True,
    )
    
    return qa_chain

class IndexChat:
    def __init__(
        self, 
        index_path: str, 
        retrieval_type: str = "similarity", 
        chain_type: str = "stuff",
        k: int = 4, 
        fetch_k: int = 20, 
        lambda_mult: float = 0.5,
        use_compression: bool = False
    ):
        self.index_path = index_path
        self.retrieval_type = retrieval_type
        self.chain_type = chain_type
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.use_compression = use_compression
        self.chat_history = []
        
        self.qa = load_from_index(
            index_path, 
            retrieval_type=retrieval_type, 
            chain_type=chain_type,
            k=k, 
            fetch_k=fetch_k, 
            lambda_mult=lambda_mult,
            use_compression=use_compression
        )
    
    def ask(self, question: str) -> Tuple[str, str, List]:
        if not question:
            return "", "", []
        
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
        use_compression=args.use_compression
    )

    print(f"\nLoaded index: {chat.index_path}")
    print(f"Retrieval method: {chat.retrieval_type}, Chain type: {chat.chain_type}, k={chat.k}")
    if chat.retrieval_type == "mmr":
        print(f"MMR settings: fetch_k={chat.fetch_k}, lambda_mult={chat.lambda_mult}")
    
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
