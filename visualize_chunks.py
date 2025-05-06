import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Union, Optional
import textwrap
import tempfile
import webbrowser
from html import escape
import json
from datetime import datetime

try:
    import google.generativeai as genai
    from google.api_core.exceptions import InvalidArgument
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class ChunkVisualizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.0-pro")
            print("Gemini model initialized for visualization")
        else:
            if not GEMINI_AVAILABLE:
                print("Warning: Gemini not available. Install with: pip install google-generativeai")
            if not self.api_key:
                print("Warning: Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            self.model = None

    def visualize_2d(self, query_embedding, chunk_embeddings, 
                     chunk_texts, prominent_indices=None, query_text=None):
        """
        Create a 2D visualization of query and document chunks using TSNE
        """
        # Ensure we have embeddings and they're properly formatted
        if query_embedding is None or chunk_embeddings is None or len(chunk_embeddings) == 0:
            print("Error: Missing embeddings for visualization")
            return None
            
        # Make sure query_embedding is properly shaped
        query_embedding = np.array(query_embedding).reshape(1, -1)
            
        # Combine query embedding with chunk embeddings
        all_embeddings = np.vstack([query_embedding, chunk_embeddings])
        
        # Ensure all embeddings have the same dimensionality
        if len(set([emb.shape[0] for emb in all_embeddings])) > 1:
            print("Error: Embeddings have inconsistent dimensions")
            return None
        
        # Apply TSNE dimensionality reduction - adjust perplexity for small datasets
        perplexity = min(30, max(5, len(all_embeddings) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        
        try:
            reduced_embeddings = tsne.fit_transform(all_embeddings)
        except Exception as e:
            print(f"TSNE error: {e}")
            return None
        
        # Extract query and chunk coordinates
        query_point = reduced_embeddings[0]
        chunk_points = reduced_embeddings[1:]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot non-prominent chunks
        for i, point in enumerate(chunk_points):
            if prominent_indices is None or i not in prominent_indices:
                plt.scatter(point[0], point[1], c='lightgrey', s=50, alpha=0.6)
                plt.annotate(f"Chunk {i+1}", (point[0], point[1]), fontsize=8, alpha=0.7)
        
        # Plot prominent chunks with different color
        if prominent_indices is not None:
            for i in prominent_indices:
                if i < len(chunk_points):
                    point = chunk_points[i]
                    plt.scatter(point[0], point[1], c='green', s=100, alpha=0.8)
                    plt.annotate(f"Chunk {i+1}", (point[0], point[1]), fontsize=10, 
                                alpha=1.0, fontweight='bold')
        
        # Plot query point
        plt.scatter(query_point[0], query_point[1], c='red', s=200, marker='*')
        plt.annotate("Query", (query_point[0], query_point[1]), fontsize=12, 
                    fontweight='bold', color='red')
        
        # Draw lines from query to prominent chunks
        if prominent_indices is not None:
            for i in prominent_indices:
                if i < len(chunk_points):
                    plt.plot([query_point[0], chunk_points[i][0]], 
                            [query_point[1], chunk_points[i][1]], 
                            'r--', alpha=0.5)
        
        # Set title and labels
        plt.title(f"Query and Document Chunks Visualization\n{query_text if query_text else ''}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        
        # Save to temp file and display
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            plt.savefig(tmp.name)
            plt.close()
            return tmp.name

    def generate_gemini_visualization(self, query, prominent_chunks, other_chunks, max_chunks=10):
        """
        Use Gemini to create a visual representation of how the query relates to the documents
        """
        if not GEMINI_AVAILABLE or not self.model:
            print("Gemini visualization not available. Please install google-generativeai package and set API key.")
            return None
        
        # Ensure we have chunks data
        if not prominent_chunks:
            print("No prominent chunks available for visualization")
            return None
            
        # Limit the number of chunks to avoid overwhelming the model
        if other_chunks and len(other_chunks) > max_chunks:
            other_chunks = other_chunks[:max_chunks]
            truncated = True
        else:
            truncated = False
            
        # Create prompt for Gemini
        system_prompt = """
        Create a visual representation showing how the user query relates to retrieved document chunks.
        
        Create an SVG diagram that:
        1. Places the query at the center
        2. Shows prominent/relevant chunks near the query with stronger connections
        3. Shows less relevant chunks further away with dotted connections
        4. Includes brief summaries of each chunk's content
        5. Uses colors to indicate relevance (green for high relevance, yellow for medium, grey for low)
        
        Return ONLY valid SVG code without any explanation or markdown - just the raw SVG that can be embedded in HTML.
        The SVG should be self-contained with appropriate viewport and size settings.
        """
        
        content_prompt = f"""
        Query: {query}
        
        Prominent Chunks:
        {json.dumps([c[:200] + '...' if len(c) > 200 else c for c in prominent_chunks])}
        
        Other Chunks:
        {json.dumps([c[:100] + '...' if len(c) > 100 else c for c in other_chunks]) if other_chunks else "None"}
        
        {f'Note: Only showing first {max_chunks} of {len(other_chunks)} other chunks.' if truncated else ''}
        """
        
        try:
            response = self.model.generate_content([system_prompt, content_prompt])
            svg_content = response.text
            
            # Ensure it's valid SVG
            if svg_content.startswith('<svg') or svg_content.strip().startswith('<svg'):
                # Create HTML file with the SVG
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Query-Document Visualization</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .container {{ margin: 0 auto; max-width: 1200px; }}
                        .header {{ text-align: center; margin-bottom: 20px; }}
                        .visualization {{ border: 1px solid #ddd; padding: 10px; }}
                        .query {{ color: #d00; font-weight: bold; }}
                        .timestamp {{ color: #777; font-size: 0.8em; text-align: right; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>Query-Document Visualization</h1>
                            <p class="query">"{escape(query)}"</p>
                            <p>Showing relationships between query and {len(prominent_chunks)} prominent chunks
                            {f' and {len(other_chunks)} other chunks' if other_chunks else ''}.</p>
                        </div>
                        <div class="visualization">
                            {svg_content}
                        </div>
                        <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </body>
                </html>
                """
                
                # Save to temp HTML file and open in browser
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp:
                    tmp.write(html_content)
                    tmp_path = tmp.name
                
                webbrowser.open(f'file://{tmp_path}')
                return tmp_path
            else:
                print("Received invalid SVG content from Gemini")
                return None
                
        except Exception as e:
            print(f"Error generating visualization with Gemini: {e}")
            return None

def create_chunk_summary(chunks, max_length=100):
    """Create a summary for each chunk"""
    return [textwrap.shorten(chunk, width=max_length, placeholder="...") for chunk in chunks]

def visualize_query_and_chunks(query, query_embedding, all_chunks, all_embeddings, 
                               prominent_indices, use_gemini=True):
    """Main function to visualize query and document chunks"""
    # Safety checks
    if query_embedding is None or all_embeddings is None or len(all_embeddings) == 0:
        print("Cannot visualize: missing embeddings")
        return None
        
    if not all_chunks or len(all_chunks) == 0:
        print("Cannot visualize: no document chunks provided")
        return None
        
    if len(all_embeddings) != len(all_chunks):
        print(f"Warning: Number of embeddings ({len(all_embeddings)}) doesn't match number of chunks ({len(all_chunks)})")
        # Truncate to the shorter length
        min_len = min(len(all_embeddings), len(all_chunks))
        all_embeddings = all_embeddings[:min_len]
        all_chunks = all_chunks[:min_len]
    
    # Initialize visualizer
    visualizer = ChunkVisualizer()
    
    # Extract text from chunks if they are document objects
    chunk_texts = []
    for chunk in all_chunks:
        if hasattr(chunk, 'page_content'):
            chunk_texts.append(chunk.page_content)
        else:
            chunk_texts.append(str(chunk))
    
    # Get prominent chunks
    prominent_chunks = []
    for i in prominent_indices:
        if i < len(chunk_texts):
            prominent_chunks.append(chunk_texts[i])
    
    if not prominent_chunks:
        print("Warning: No prominent chunks identified")
        # If no prominent chunks, use the first few as prominent
        prominent_chunks = chunk_texts[:min(3, len(chunk_texts))]
        prominent_indices = list(range(min(3, len(chunk_texts))))
    
    # Get other chunks
    other_indices = [i for i in range(len(chunk_texts)) if i not in prominent_indices]
    other_chunks = [chunk_texts[i] for i in other_indices]
    
    try:
        # Create TSNE visualization
        tsne_file = visualizer.visualize_2d(
            query_embedding, 
            all_embeddings, 
            chunk_texts, 
            prominent_indices,
            query
        )
        
        # Open the TSNE visualization
        if tsne_file:
            webbrowser.open(f'file://{tsne_file}')
            print(f"TSNE visualization saved to: {tsne_file}")
    except Exception as e:
        print(f"Error creating TSNE visualization: {e}")
        tsne_file = None
    
    # Create Gemini visualization if requested
    gemini_file = None
    if use_gemini:
        try:
            gemini_file = visualizer.generate_gemini_visualization(
                query, 
                prominent_chunks, 
                other_chunks
            )
            if gemini_file:
                print(f"Gemini visualization saved to: {gemini_file}")
        except Exception as e:
            print(f"Error creating Gemini visualization: {e}")
    
    return {
        "tsne_file": tsne_file,
        "gemini_file": gemini_file if use_gemini else None,
        "prominent_chunks": create_chunk_summary(prominent_chunks),
        "other_chunks": create_chunk_summary(other_chunks)
    }

if __name__ == "__main__":
    # Test with sample data
    query = "What is the capital of France?"
    query_embedding = np.random.rand(384)  # Random embedding for testing
    all_chunks = [
        "Paris is the capital of France and has a population of over 2 million people.",
        "France is a country in Western Europe with a population of 67 million.",
        "The Eiffel Tower is located in Paris and was built in 1889.",
        "London is the capital of the United Kingdom.",
        "Berlin is the capital of Germany.",
        "Italy's capital is Rome."
    ]
    all_embeddings = np.random.rand(len(all_chunks), 384)  # Random embeddings for testing
    prominent_indices = [0, 2]  # Chunks about Paris
    
    result = visualize_query_and_chunks(
        query, 
        query_embedding, 
        all_chunks, 
        all_embeddings, 
        prominent_indices
    )
    
    print("Visualization complete!")
