import os
import json
import subprocess
import faiss
import numpy as np
from langchain_core.documents import Document
from webvtt import WebVTT
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
import concurrent.futures
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional
import requests
from sentence_transformers import SentenceTransformer
import uvicorn
import re
import traceback
import logging
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging to output to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Paths for saving knowledge base
INDEX_PATH = "/app/data/faiss_index.bin"
METADATA_PATH = "/app/data/metadata.json"

# Read the IP address and port from environment variables
inference_server_ip = os.getenv("INFERENCE_SERVER_IP")
inference_server_port = os.getenv("INFERENCE_SERVER_PORT")

if not ip_address or not port:
    raise ValueError("INFERENCE_SERVER_IP and INFERENCE_SERVER_PORT must be set in the environment")

# Construct the full URL
inference_server_url = f"http://{inference_server_ip}:{inference_server_port}/v1/chat/completions"

# # Ensure the directory exists
# os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

# # Create index file if it doesn't exist
# if not os.path.exists(INDEX_PATH):
#     with open(INDEX_PATH, 'wb') as f:
#         pass  # Create an empty binary file

# # Create metadata file if it doesn't exist
# if not os.path.exists(METADATA_PATH):
#     with open(METADATA_PATH, 'w') as f:
#         json.dump([], f)  # Create an empty JSON array

#model_name = "sentence-transformers/all-mpnet-base-v2"
#model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
#model_name = "BAAI/bge-large-en-v1.5"

# Load embedding model
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embedding_function  = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def verify_channel_url(channel_url):
    youtube_url_pattern = r"^https://www\.youtube\.com/@[a-zA-Z0-9_-]+(/videos)?$"
    if not re.match(youtube_url_pattern, channel_url):
        raise HTTPException(status_code=400, detail="Invalid YouTube channel URL.")
    print(f"Successfully verified channel URL: {channel_url}")
    return True

# Step 1: Fetch Latest Video IDs using yt-dlp
def get_latest_video_ids(channel_url, num_videos=10):
    try:
        print(f"Attempting to fetch the latest {num_videos} videos from: {channel_url}")
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--get-id", channel_url],
            capture_output=True,
            text=True,
            check=True
        )
        video_ids = result.stdout.strip().split("\n")[:num_videos]
        print(f"Found {len(video_ids)} videos: {video_ids}")
        return video_ids
    except subprocess.CalledProcessError as e:
        print(f"Error fetching video IDs: {e.stderr}")
        raise HTTPException(status_code=500, detail="Failed to fetch video IDs from the channel.")


# Step 2: Download Transcripts with Multithreading
def download_transcripts(video_ids, output_folder="transcripts"):
    os.makedirs(output_folder, exist_ok=True)

    def download_video(video_id):
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        output_file = os.path.join(output_folder, f"{video_id}.vtt")
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "--write-auto-subs",
                    "--skip-download",
                    "--sub-lang", "en",
                    "-o", output_file,
                    video_url
                ],
                check=True
            )
            print(f"Transcript downloaded: {output_file}")
        except subprocess.CalledProcessError:
            print(f"Error downloading transcript for {video_id}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_video, video_ids)


# Step 3: Reformat Transcripts
def reformat_transcripts(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".vtt"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".vtt", "_formatted.txt"))
            with open(output_path, "w") as output:
                for caption in WebVTT().read(input_path):
                    output.write("[" + caption.start + "] " + caption.text.replace('\n', ' ') + "\n")
            print(f"Reformatted: {output_path}")


# Step 4: Chunkify Transcripts
def chunkify_transcripts(folder):
    chunks = []
    for file_name in os.listdir(folder):
        if file_name.endswith("_formatted.txt"):
            with open(os.path.join(folder, file_name), "r") as file:
                transcript = file.read()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            for chunk_text in splitter.split_text(transcript):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {"video_id": file_name.replace("_formatted.txt", "")}
                })
    return chunks


# Step 5: Knowledge Base Management
def save_knowledge_base(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    print("Knowledge base saved.")


def load_knowledge_base():
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        print("Knowledge base loaded.")
        return index, metadata
    else:
        # Embed a sample text to check the dimensionality
        sample_embeddings = embedding_function.embed_documents(["test text"])  # Embed sample text
        embedding_dim = len(sample_embeddings[0])  # Get the dimension of the first embedding
        print(f"Embedding dimensionality: {embedding_dim}")  # Check the dimensionality

        # Initialize the FAISS index with the correct dimensionality
        index = faiss.IndexFlatL2(embedding_dim)

        metadata = []
        print("No existing knowledge base found. Initialized a new one.")
        return index, metadata


def update_knowledge_base(chunks, index, metadata):
    """Update the FAISS index with new data."""
    # Extract existing video IDs from metadata
    existing_ids = {entry["metadata"]["video_id"] for entry in metadata}

    # All chunks uses the same video_id, so we just have to ensure this video has not been added before
    new_chunks = []
    if len(chunks) > 0:
        if chunks[0]["metadata"]["video_id"] not in existing_ids:
            new_chunks = chunks

    if new_chunks:
        # Embed new texts using Hugging Face embeddings
        texts = [chunk["text"] for chunk in new_chunks]
        
        # Generate embeddings for the texts
        embeddings = np.array(embedding_function.embed_documents(texts))

        # Add new embeddings to FAISS index
        embedding_dim = embeddings.shape[1]  # Get the dimensionality of the embeddings
        if index.ntotal == 0:
            index = faiss.IndexFlatL2(embedding_dim)  # Create a new FAISS index with the correct dimension
        index.add(embeddings)

        # Extend metadata
        metadata.extend(new_chunks)
        print(f"Added {len(new_chunks)} new chunks to knowledge base.")

    else:
        print("No new data to add.")

    docstore = InMemoryDocstore({
        str(i): Document(
            page_content=chunk["text"],
            metadata={"video_id": chunk["metadata"]["video_id"]}
        )
        for i, chunk in enumerate(metadata)
    })

    index_to_docstore_id = {i: str(i) for i in range(len(metadata))}

    print(f"Number of vectors in FAISS index after updating knowledge base: {index.ntotal}")
    return index, metadata, docstore, index_to_docstore_id

# Step 6: Query the Knowledge Base
def query_and_analyze_knowledge_base(index, metadata, query):
    try:
        # Create the docstore and vectorstore
        docstore = InMemoryDocstore({
            str(i): Document(
                page_content=entry["text"],
                metadata={"video_id": entry["metadata"]["video_id"]}
            )
            for i, entry in enumerate(metadata)
        })

        index_to_docstore_id = {i: str(i) for i in range(len(metadata))}

        # Create the FAISS vectorstore
        vectorstore = FAISS(
            embedding_function=embedding_function,
            index=index,  # Correctly passing the FAISS index object
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        #print(f"Number of vectors in FAISS index: {index.ntotal}")

        #retrieved_docs = vectorstore.similarity_search(query, k=3)
        retrieved_docs = vectorstore.similarity_search(query, k=1)
        context_with_links = []
        for doc in retrieved_docs:
            video_id = doc.metadata["video_id"]
            timestamp_str = doc.page_content.split("[")[1].split("]")[0]
            timestamp_parts = list(map(float, timestamp_str.split(":")))
            timestamp_seconds = int(timestamp_parts[0] * 3600 + timestamp_parts[1] * 60 + timestamp_parts[2])
            youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={timestamp_seconds}s"
            context_with_links.append(f"{doc.page_content}\nLink: {youtube_link}")

        context = "\n\n".join(context_with_links)
        #print(f"context: {context}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""
            Context: 
            {context}

            Question: {query}

            Provide a detailed analysis in the form of a list, where each point starts on a new line. Clearly identify contradictions, agreements, or supporting statements. Include the YouTube links and timestamps where relevant information is found. Use structured formatting for clarity.
            """}
        ]
        #logging.info(f"messages: {messages}")

        response = requests.post(
            inference_server_url,
            headers={"Content-Type": "application/json"},
            json={"model": "llama-3.2-1b-instruct", "messages": messages, "temperature": 0}
        )

        if response.status_code == 200:
            analysis = response.json()["choices"][0]["message"]["content"]
            print("\nAnalysis:")
            print(analysis)
        else:
            print(f"Error querying LM Studio: {response.status_code}, {response.text}")
    except Exception as e:
        # Log the detailed error traceback
        print(f"Error during query_and_analyze_knowledge_base: {str(e)}")
        traceback.print_exc()  # This prints the full traceback to the console

# Initialize FastAPI app
app = FastAPI(title="ChannelGPT API")


class Query(BaseModel):
    text: str
    channel_id: Optional[str]


@app.post("/query")
async def query_endpoint(query: Query = Body(...)):
    try:
        if not query.text:
            raise HTTPException(status_code=400, detail="Query text is required")

        channel_url = query.channel_id
        verify_channel_url(channel_url)

        index, metadata = load_knowledge_base()
        if index is None or index.ntotal == 0:
            print("Knowledge base is empty. Fetching new data...")
            index = faiss.IndexFlatL2(512)
            metadata = []

        video_ids = get_latest_video_ids(channel_url, 10)
        if not video_ids:
            return {"analysis": "No videos found for the provided channel URL."}

        download_transcripts(video_ids)
        reformat_transcripts("transcripts", "transcripts_formatted")
        chunks = chunkify_transcripts("transcripts_formatted")
        print(f"Number of chunks created: {len(chunks)}")
        print(f"First 5 chunks: {chunks[:5]}")

        index, metadata, docstore, index_to_docstore_id = update_knowledge_base(chunks, index, metadata)
        save_knowledge_base(index, metadata)

        if index.ntotal == 0:
            return {"analysis": "No content available in the knowledge base. Please check the channel URL and try again."}

        from io import StringIO
        import sys

        old_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result

        query_and_analyze_knowledge_base(index, metadata, query.text)

        sys.stdout = old_stdout
        response = result.getvalue()

        return {"analysis": response}

    except Exception as e:
        # Log the detailed error traceback
        print(f"Error in query_endpoint: {str(e)}")
        traceback.print_exc()  # This prints the full traceback to the console
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
