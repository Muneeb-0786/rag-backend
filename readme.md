# Any File RAG 🚀

A versatile Retrieval Augmented Generation (RAG) system that can process, analyze, and answer questions about content from multiple file formats, including:

- 📄 PDF documents
- 📝 Text files
- 📋 Word documents (.docx)
- 🎵 Audio files (.mp3)
- 🎬 YouTube videos (via automatic transcription)

## Features

- **Multi-format Support**: Process various file types through a unified interface
- **YouTube Integration**: Automatically download audio from YouTube videos and transcribe
- **Audio Transcription**: Convert speech to text using AssemblyAI
- **Efficient Processing**: Handles large files through chunking and batched processing
- **Smart Memory Management**: Uses lightweight embeddings for very large documents
- **Session Management**: Store and retrieve conversations with specific documents
- **RESTful API**: Easy integration with any frontend application

## How It Works

1. **Upload a file or YouTube URL**: The system processes the content and creates a vector database
2. **Ask questions**: Submit questions about the document content
3. **Get AI-powered answers**: The system uses Google's Gemini for:
   - Direct summarization based on the entire content
   - RAG-powered answers using relevant context retrieval

## API Endpoints

- `POST /api/upload`: Upload a file for processing
- `POST /api/youtube`: Process a YouTube video
- `POST /api/query`: Ask questions about previously processed content
- `GET /api/sessions`: List active sessions

## Setup

### Prerequisites

- Python 3.8+
- AssemblyAI API key (for audio transcription)
- Google Gemini API key (for AI generation)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/any-file-rag.git
   cd any-file-rag
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

5. Run the application:
   ```
   python any-rag.py
   ```

## Dependencies

- Flask: Web framework
- LangChain: RAG pipeline infrastructure
- FAISS: Vector database for efficient similarity search
- AssemblyAI: Audio transcription API
- Google Generative AI: Gemini models for text generation
- yt-dlp: YouTube video download
- PyPDF2: PDF parsing
- python-docx: Word document parsing
- sentence-transformers: Text embeddings

## Usage Examples

### Process a PDF document

```python
import requests

files = {'file': open('document.pdf', 'rb')}
data = {'question': 'What are the key findings in this document?'}
response = requests.post('http://localhost:5000/api/upload', files=files, data=data)
print(response.json())
```

### Process a YouTube video

```python
import requests
import json

headers = {'Content-Type': 'application/json'}
data = {
    'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'question': 'What is the main message of this video?'
}
response = requests.post('http://localhost:5000/api/youtube', headers=headers, data=json.dumps(data))
print(response.json())
```

### Ask follow-up questions

```python
import requests
import json

headers = {'Content-Type': 'application/json'}
data = {
    'session_id': '123e4567-e89b-12d3-a456-426614174000',  # Use session_id from previous response
    'question': 'Can you elaborate on the second point?'
}
response = requests.post('http://localhost:5000/api/query', headers=headers, data=json.dumps(data))
print(response.json())
```

## Performance Considerations

- The system uses batched processing for large documents to manage memory usage
- For extremely large files (>1MB), the content is split into smaller segments
- Memory usage is optimized through garbage collection between batches
- Session data is ephemeral by default but vector databases are saved to disk

## Demo

![Any File RAG Demo](https://via.placeholder.com/800x400?text=Any+File+RAG+Demo)

*Add a screenshot or GIF of your application in action. You can replace the placeholder URL above with an actual image link once available.*

## API Reference

### GET `/api/health`
Health check endpoint to verify the API is running.

**Response:**
```json
{
  "status": "ok",
  "timestamp": 1714168028.45,
  "version": "1.0.0"
}
```

### POST `/api/upload`
Upload a file for processing.

**Request:**
- Form data with:
  - `file`: The file to process
  - `question` (optional): Initial question to answer about the document

**Response:**
```json
{
  "session_id": "2ee25e17-9670-450c-b5c7-612aa20e9a7f",
  "status": "processing",
  "message": "File uploaded and processing started"
}
```

### GET `/api/sessions/{session_id}/status`
Check the status of a processing session.

**Response:**
```json
{
  "session_id": "2ee25e17-9670-450c-b5c7-612aa20e9a7f",
  "processing_complete": true,
  "created_at": 1714168028.45,
  "file_name": "example.pdf",
  "content_length": 125000,
  "processing_time": 5.67
}
```

### GET `/api/sessions/{session_id}/result`
Get the complete processing result for a session.

**Response:**
```json
{
  "session_id": "2ee25e17-9670-450c-b5c7-612aa20e9a7f",
  "file_name": "example.pdf",
  "content_length": 125000,
  "content_preview": "This is the beginning of the document...",
  "summary": "This document discusses...",
  "initial_answer": "The document is about...",
  "processing_time": 5.67
}
```

### POST `/api/sessions/{session_id}/query`
Query a processed document.

**Request:**
```json
{
  "query": "What are the main points in this document?",
  "detailed": true
}
```

**Response:**
```json
{
  "query": "What are the main points in this document?",
  "answer": "The main points in the document are...",
  "processing_time": 0.45,
  "sources": [
    {
      "id": 1,
      "content": "Relevant excerpt from the document...",
      "metadata": {}
    }
  ]
}
```

### DELETE `/api/sessions/{session_id}`
Delete a session and its associated files.

**Response:**
```json
{
  "status": "success",
  "message": "Session deleted"
}
```

### GET `/api/sessions`
List all active sessions.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "2ee25e17-9670-450c-b5c7-612aa20e9a7f",
      "file_name": "example.pdf",
      "created_at": 1714168028.45,
      "last_accessed": 1714168120.78,
      "processing_complete": true,
      "has_error": false
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **File Upload Errors**
   - Check that your file is one of the supported formats: PDF, TXT, DOCX, MP3, MP4
   - Ensure the file size is under the 100MB limit

2. **API Key Issues**
   - Verify that your AssemblyAI and Gemini API keys are correctly set in the `.env` file
   - Check that your API keys are active and have sufficient credits

3. **Memory Errors**
   - When processing very large files, ensure your system has adequate RAM (8GB+ recommended)
   - Try splitting extremely large documents into smaller parts

4. **YouTube Processing Issues**
   - Make sure you have yt-dlp properly installed
   - Check that the YouTube URL is valid and the video is publicly accessible

### Logging

The application logs important events to the console. If you're experiencing issues, check the logs for error messages and stack traces.

## Development

### Project Structure

```
any-file-rag/
├── any-rag.py          # Main application file
├── requirements.txt    # Python dependencies
├── uploads/            # Uploaded file storage
├── sessions/           # Session vector databases
└── .env                # Environment variables
```

### Testing

Run the test suite with:
```
python -m pytest tests/
```

### Local Development

For development purposes, you can run the server with debug mode enabled:
```
python any-rag.py --debug
```

## Future Improvements

- [ ] Add user authentication for secure API access
- [ ] Implement Docker containerization for easy deployment
- [ ] Add support for more file formats (e.g., Excel, PowerPoint)
- [ ] Create a web-based UI for easier interaction
- [ ] Implement caching to improve response times for repeated queries
- [ ] Add support for real-time collaboration on documents

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.