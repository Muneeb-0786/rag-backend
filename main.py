from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from query_embeddings import app as query_app
from create_embeddings import app as create_app

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-applications
app.mount("/query", query_app)
app.mount("/create", create_app)
