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
                     chunk_texts, prominent_indices=None, query_text=None,
                     augmented_queries=None, augmented_embeddings=None,
                     is_reranked=False):
        """
        Create a 2D visualization of query and document chunks using TSNE
        """
        # Ensure we have embeddings and they're properly formatted
        if query_embedding is None or chunk_embeddings is None or len(chunk_embeddings) == 0:
            print("Error: Missing embeddings for visualization")
            return None
            
        # Make sure query_embedding is properly shaped
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Process augmented query embeddings if available
        aug_embeddings_array = None
        if augmented_queries and augmented_embeddings:
            aug_embeddings_array = np.array(augmented_embeddings)
            
        # Combine query embedding with augmented query embeddings and chunk embeddings
        if aug_embeddings_array is not None and len(aug_embeddings_array) > 0:
            all_embeddings = np.vstack([query_embedding, aug_embeddings_array, chunk_embeddings])
            aug_count = len(aug_embeddings_array)
        else:
            all_embeddings = np.vstack([query_embedding, chunk_embeddings])
            aug_count = 0
            
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
        
        # Extract query, augmented queries, and chunk coordinates
        query_point = reduced_embeddings[0]
        
        if aug_count > 0:
            aug_points = reduced_embeddings[1:aug_count+1]
            chunk_points = reduced_embeddings[aug_count+1:]
        else:
            aug_points = []
            chunk_points = reduced_embeddings[1:]
        
        # Create plot with larger figure for better visualization of the chunk cloud
        plt.figure(figsize=(15, 10))
        
        # Plot all chunks as a cloud with varying transparency based on distance from query
        for i, point in enumerate(chunk_points):
            # Calculate distance to query (for transparency)
            dist = np.linalg.norm(point - query_point)
            max_dist = np.max([np.linalg.norm(p - query_point) for p in chunk_points])
            alpha = max(0.2, 1 - (dist / max_dist * 0.8))  # Scale transparency
            
            # Choose color based on whether it's prominent and reranked
            if prominent_indices is not None and i in prominent_indices:
                if is_reranked:
                    # Green with red border for reranked prominent chunks
                    plt.scatter(point[0], point[1], c='lightgreen', s=120, alpha=alpha, 
                                edgecolor='red', linewidth=2)
                    plt.annotate(f"Chunk {i+1} ↑", (point[0], point[1]), fontsize=10, 
                                alpha=1.0, fontweight='bold', color='darkgreen')
                else:
                    # Green for prominent chunks
                    plt.scatter(point[0], point[1], c='green', s=100, alpha=alpha)
                    plt.annotate(f"Chunk {i+1}", (point[0], point[1]), fontsize=10, 
                                alpha=1.0, fontweight='bold')
            else:
                # Light gray for non-prominent chunks
                plt.scatter(point[0], point[1], c='lightgrey', s=70, alpha=alpha * 0.8)
                plt.annotate(f"{i+1}", (point[0], point[1]), fontsize=8, alpha=alpha * 0.9)
        
        # Calculate which chunks are closest to each augmented query
        aug_query_closest_chunks = {}
        if len(aug_points) > 0:
            for i, aug_point in enumerate(aug_points):
                # Calculate distances to all chunks
                distances = [np.linalg.norm(aug_point - point) for point in chunk_points]
                # Get indices of the 3 closest chunks
                closest_indices = np.argsort(distances)[:3]
                aug_query_closest_chunks[i] = closest_indices
        
        # Plot augmented queries as yellow stars
        for i, point in enumerate(aug_points):
            plt.scatter(point[0], point[1], c='gold', s=150, marker='*', alpha=0.9, 
                       edgecolor='black', linewidth=0.5)
            plt.annotate(f"Aug Q{i+1}", (point[0], point[1]), fontsize=9, 
                        alpha=0.9, color='darkorange', fontweight='bold',
                        xytext=(10, -5), textcoords='offset points')
            
            # Draw dashed line to original query
            plt.plot([query_point[0], point[0]], [query_point[1], point[1]], 
                     'y--', alpha=0.6, linewidth=1.5)
            
            # Draw light dashed lines to the closest chunks
            if i in aug_query_closest_chunks:
                for chunk_idx in aug_query_closest_chunks[i]:
                    plt.plot([point[0], chunk_points[chunk_idx][0]], 
                            [point[1], chunk_points[chunk_idx][1]], 
                            'y:', alpha=0.4, linewidth=1.0)
                    
                    # Add small callout showing which augmented query influenced which chunks
                    # Only add for first closest chunk for clarity
                    if chunk_idx == aug_query_closest_chunks[i][0]:
                        # Calculate midpoint for label placement
                        mid_x = (point[0] + chunk_points[chunk_idx][0]) / 2
                        mid_y = (point[1] + chunk_points[chunk_idx][1]) / 2
                        plt.annotate(f"Q{i+1}→{chunk_idx+1}", (mid_x, mid_y), 
                                    fontsize=7, color='orange', alpha=0.7,
                                    bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.3))
        
        # Plot original query point (red star)
        plt.scatter(query_point[0], query_point[1], c='red', s=250, marker='*', 
                   edgecolor='black', linewidth=1)
        plt.annotate("Query", (query_point[0], query_point[1]), fontsize=12, 
                    fontweight='bold', color='red')
        
        # Draw lines from query to prominent chunks
        if prominent_indices is not None:
            for i in prominent_indices:
                if i < len(chunk_points):
                    # Use thicker line for reranked chunks
                    if is_reranked:
                        plt.plot([query_point[0], chunk_points[i][0]], 
                                [query_point[1], chunk_points[i][1]], 
                                'r-', alpha=0.6, linewidth=2)
                    else:
                        plt.plot([query_point[0], chunk_points[i][0]], 
                                [query_point[1], chunk_points[i][1]], 
                                'r--', alpha=0.5, linewidth=1.5)
        
        # Add reranking indicator in the plot
        if is_reranked:
            plt.text(0.02, 0.02, "⟳ Reranking Applied", transform=plt.gca().transAxes,
                    fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        # Create a figure legend explaining all elements
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, 
                      label='Original Query'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, 
                      label='Augmented Queries'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                      label='Prominent Chunks'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey', markersize=10, 
                      label='Other Chunks'),
            plt.Line2D([0], [0], linestyle=':', color='gold', 
                     label='Aug Query → Chunk Connection')
        ]
        if is_reranked:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                          markeredgecolor='red', markersize=10, label='Reranked Chunks')
            )
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add explanation of augmented query connections
        augmented_text = ""
        if augmented_queries and aug_query_closest_chunks:
            augmented_text = "\nAugmented Query Influences:"
            for i, indices in aug_query_closest_chunks.items():
                if i < len(augmented_queries):
                    query_excerpt = augmented_queries[i][:30] + "..." if len(augmented_queries[i]) > 30 else augmented_queries[i]
                    chunks_str = ", ".join([f"Chunk {idx+1}" for idx in indices])
                    augmented_text += f"\nAug Q{i+1}: \"{query_excerpt}\" → {chunks_str}"
        
        # Set title and labels
        title = f"Query and Document Chunks Visualization\n{query_text if query_text else ''}"
        if is_reranked:
            title += "\n(Results reranked to improve relevance)"
        
        # Add augmented query explanation below main visualization
        plt.figtext(0.5, 0.01, augmented_text, ha="center", fontsize=9, 
                   bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
                
        plt.title(title, fontsize=10)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the augmented query info at bottom
        
        # Save to temp file and display
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            plt.savefig(tmp.name, dpi=300)  # Higher DPI for better quality
            plt.close()
            return tmp.name

    def generate_gemini_visualization(self, query, prominent_chunks, other_chunks, 
                                     max_chunks=10, augmented_queries=None,
                                     is_reranked=False):
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
        1. Places the original query at the center as a red star
        2. If provided, shows augmented queries as yellow stars connected to the original query with dashed lines
        3. IMPORTANT: For each augmented query, clearly show which document chunks it helped retrieve with connecting lines
        4. Shows prominent/relevant chunks near the query with stronger connections
        5. Shows less relevant chunks further away with dotted connections
        6. Includes brief summaries of each chunk's content
        7. Uses colors to indicate relevance (green for high relevance, yellow for medium, grey for low)
        8. If reranking was applied, add a visual indicator (like recycling arrows ⟳) and make reranked chunks stand out
        9. Clearly label which chunks were retrieved due to augmented queries vs. the original query
        
        Return ONLY valid SVG code without any explanation or markdown - just the raw SVG that can be embedded in HTML.
        The SVG should be self-contained with appropriate viewport and size settings.
        """
        
        content_prompt = f"""
        Original Query: {query}
        
        Augmented Queries: {json.dumps(augmented_queries) if augmented_queries else "None"}
        
        Prominent Chunks:
        {json.dumps([c[:300] + '...' if len(c) > 300 else c for c in prominent_chunks])}
        
        Other Chunks:
        {json.dumps([c[:150] + '...' if len(c) > 150 else c for c in other_chunks]) if other_chunks else "None"}
        
        Reranking Applied: {"Yes" if is_reranked else "No"}
        
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
                        .augmented-queries {{ color: #a60; margin-bottom: 15px; background-color: #fff8e1; padding: 10px; border-radius: 5px; }}
                        .augmented-query {{ margin: 5px 0; font-style: italic; }}
                        .augmented-mapping {{ margin-top: 10px; background-color: #fffbe8; padding: 8px; border-left: 3px solid gold; }}
                        .reranking {{ color: #060; background-color: #e6ffe6; padding: 8px; border-radius: 5px; 
                                   border-left: 4px solid green; margin: 10px 0; }}
                        .timestamp {{ color: #777; font-size: 0.8em; text-align: right; }}
                        .legend {{ margin: 15px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
                        .legend-item {{ display: inline-block; margin-right: 20px; }}
                        .star {{ color: gold; }}
                        .red-star {{ color: red; }}
                        .highlight {{ background-color: #ff06; padding: 0 3px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>Query-Document Visualization</h1>
                            <p class="query"><span class="red-star">★</span> Original Query: "{escape(query)}"</p>
                            {f'''
                            <div class="augmented-queries">
                                <p><span class="star">★</span> Augmented Queries:</p>
                                {''.join([f'<p class="augmented-query">{i+1}. "{escape(q)}"</p>' for i, q in enumerate(augmented_queries)])}
                                <div class="augmented-mapping">
                                    <p><strong>How Augmented Queries Help:</strong></p>
                                    <p>Yellow stars represent alternative formulations of your query that help retrieve additional relevant information that might be missed by the original query alone. Lines between stars and chunks show which chunks were influenced by each augmented query.</p>
                                </div>
                            </div>
                            ''' if augmented_queries else ''}
                            {f'''
                            <div class="reranking">
                                <p>⟳ Reranking applied to improve result relevance</p>
                            </div>
                            ''' if is_reranked else ''}
                            <p>Showing relationships between query and {len(prominent_chunks)} prominent chunks
                            {f' and {len(other_chunks)} other chunks' if other_chunks else ''}.</p>
                            <div class="legend">
                                <div class="legend-item"><span class="red-star">★</span> Original Query</div>
                                <div class="legend-item"><span class="star">★</span> Augmented Queries</div>
                                <div class="legend-item">● Prominent Chunks</div>
                                <div class="legend-item">○ Other Chunks</div>
                                {f'<div class="legend-item">⟳ Reranked Results</div>' if is_reranked else ''}
                            </div>
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
                               prominent_indices, use_gemini=True, augmented_queries=None,
                               is_reranked=False, augmented_embeddings=None):
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
    
    # If augmented embeddings not provided but we have a db with embedding function,
    # try to create them directly
    if augmented_queries and augmented_embeddings is None:
        try:
            # Try direct embedding using standard approach
            augmented_embeddings = []
            from langchain_cohere import CohereEmbeddings
            embeddings = CohereEmbeddings(model="embed-english-v3.0")
            for q in augmented_queries:
                try:
                    emb = embeddings.embed_query(q)
                    augmented_embeddings.append(emb)
                except Exception as e:
                    print(f"Could not embed augmented query: {e}")
                    # If we fail, just use a copy of the original query embedding with some noise
                    noise = np.random.normal(0, 0.1, query_embedding.shape)
                    modified_emb = query_embedding + noise
                    augmented_embeddings.append(modified_emb[0])
        except Exception as e:
            print(f"Failed to create augmented embeddings: {e}")
            augmented_embeddings = None
    
    try:
        # Create TSNE visualization
        tsne_file = visualizer.visualize_2d(
            query_embedding, 
            all_embeddings, 
            chunk_texts, 
            prominent_indices,
            query,
            augmented_queries,
            augmented_embeddings,
            is_reranked
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
                other_chunks,
                augmented_queries=augmented_queries,
                is_reranked=is_reranked
            )
            if gemini_file:
                print(f"Gemini visualization saved to: {gemini_file}")
        except Exception as e:
            print(f"Error creating Gemini visualization: {e}")
    
    return {
        "tsne_file": tsne_file,
        "gemini_file": gemini_file if use_gemini else None,
        "prominent_chunks": create_chunk_summary(prominent_chunks),
        "other_chunks": create_chunk_summary(other_chunks),
        "augmented_queries": augmented_queries,
        "is_reranked": is_reranked
    }

if __name__ == "__main__":
    # Test with sample data
    query = "What is the capital of France?"
    augmented_queries = [
        "Which city serves as the capital of France?",
        "Tell me about the main city and administrative center of France"
    ]
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
        prominent_indices,
        augmented_queries=augmented_queries,
        is_reranked=True  # Test with reranking enabled
    )
    
    print("Visualization complete!")
    # Test with augmented query highlighting
    print("Added visualization for augmented query landing positions")
