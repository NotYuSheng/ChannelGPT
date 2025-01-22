import concurrent.futures
import json
import os
import logging
import re
import subprocess
import traceback
from typing import Optional, Tuple, Dict, List

import faiss
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, models
from webvtt import WebVTT

# Configure logging to output to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Paths for saving knowledge base
INDEX_PATH = "/app/data/faiss_index.bin"
METADATA_PATH = "/app/data/metadata.json"
INDEX_TO_DOCSTORE_ID_PATH = "/app/data/index_to_docstore_id.json"
DOCSTORE_PATH = "/app/data/docstore.json"

# Define paths
#MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"
# Local directory to save the model
#LOCAL_MODEL_PATH = "./models/bge-base-en-v1.5"
LOCAL_MODEL_PATH = "./models/distiluse-base-multilingual-cased-v1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Check if the model is saved locally; download if necessary
if not os.path.exists(LOCAL_MODEL_PATH):
    logging.info(f"Model not found at {LOCAL_MODEL_PATH}. Downloading and converting...")
    word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    transformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    transformer_model.save(LOCAL_MODEL_PATH)
    logging.info(f"Model successfully saved at {LOCAL_MODEL_PATH}")
else:
    logging.info(f"Model already exists at {LOCAL_MODEL_PATH}. No action needed.")

# Load embedding function
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embedding_function = HuggingFaceEmbeddings(
    model_name=LOCAL_MODEL_PATH,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize embedding_dim
sample_embeddings = embedding_function.embed_documents(["test text"])
embedding_dim = len(sample_embeddings[0])
logging.info(f"Embedding dimensionality: {embedding_dim}")

def verify_channel_url(channel_url: str) -> bool:
    """Verify if the provided URL is a valid YouTube channel URL."""
    youtube_url_pattern = r"^https://www\.youtube\.com/@[a-zA-Z0-9_-]+(/videos)?$"
    if not re.match(youtube_url_pattern, channel_url):
        raise HTTPException(status_code=400, detail="Invalid YouTube channel URL.")
    logging.info(f"Successfully verified channel URL: {channel_url}")
    return True

def get_video_details(video_ids: list, channel_handle: str) -> list:
    """Fetch video titles, upload dates, and channel handles using yt-dlp."""
    video_details = []
    for video_id in video_ids:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            result = subprocess.run(
                ["yt-dlp", 
                 "--print", "title", 
                 "--print", "upload_date", 
                 video_url],
                capture_output=True,
                text=True,
                check=True
            )
            output_lines = result.stdout.strip().split("\n")
            video_title = output_lines[0]
            upload_date = output_lines[1]
            video_details.append({
                "video_id": video_id,
                "title": video_title,
                "upload_date": upload_date,
                "channel_handle": channel_handle
            })
            logging.info(f"Fetched details for video ID {video_id}: Title: {video_title}, "
                         f"Upload Date: {upload_date}, channel_handle: {channel_handle}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error fetching details for {video_id}: {e.stderr}", exc_info=True)
            continue
    return video_details

def get_latest_video_ids(channel_url: str, num_videos: int = 10) -> list:
    """Fetch the latest video IDs from a YouTube channel using yt-dlp."""
    try:
        logging.info(f"Attempting to fetch the latest {num_videos} videos from: {channel_url}")
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--get-id", channel_url],
            capture_output=True,
            text=True,
            check=True
        )
        video_ids = result.stdout.strip().split("\n")[:num_videos]
        logging.info(f"Found {len(video_ids)} videos: {video_ids}")
        return video_ids
    except subprocess.CalledProcessError as e:
        logging.error(f"Error fetching video IDs: {e.stderr}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch video IDs from the channel.")

def clean_transcript(input_file: str, output_file: str) -> None:
    """Clean up overlapping and repeated content in the transcript, retaining timestamps."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = []
        last_text = ""
        timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})")

        for line in lines:
            # Extract timestamp and content
            match = timestamp_pattern.search(line)
            if match:
                timestamp = match.group(1)
                text = line.replace(timestamp, "").strip()

                # Add line if it's not a repetition
                if text and text != last_text:
                    cleaned_lines.append(f"{timestamp} {text}")
                    last_text = text

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(cleaned_lines))

        logging.info(f"Cleaned transcript saved to: {output_file}")

    except Exception as e:
        logging.error(f"Error cleaning transcript: {input_file}", exc_info=True)

def download_transcripts(video_ids: list, output_folder: str = "transcripts") -> None:
    """Download transcripts for given video IDs using yt-dlp and clean them."""
    os.makedirs(output_folder, exist_ok=True)

    def download_and_clean(video_id):
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        raw_output_file = os.path.join(output_folder, f"{video_id}.vtt")
        cleaned_output_file = os.path.join(output_folder, f"{video_id}_cleaned.txt")
        
        try:
            # Download transcript
            subprocess.run(
                [
                    "yt-dlp",
                    "--write-auto-subs",
                    "--skip-download",
                    "--sub-lang", "en",
                    "-o", os.path.join(output_folder, f"{video_id}"),
                    video_url
                ],
                check=True
            )
            logging.info(f"Transcript downloaded: {raw_output_file}")
            
            # Check if the file exists before cleaning
            if os.path.exists(raw_output_file):
                clean_transcript(raw_output_file, cleaned_output_file)
            else:
                logging.error(f"Transcript file not found: {raw_output_file}")

        except subprocess.CalledProcessError:
            logging.error(f"Error downloading transcript for {video_id}", exc_info=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_and_clean, video_ids)

def reformat_transcripts(input_folder: str, output_folder: str) -> None:
    """Reformat VTT transcripts to plain text files with timestamps."""
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".vtt"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".vtt", "_formatted.txt"))
            with open(output_path, "w") as output:
                for caption in WebVTT().read(input_path):
                    output.write("[" + caption.start + "] " + caption.text.replace('\n', ' ') + "\n")
            logging.info(f"Reformatted: {output_path}")

def chunkify_transcripts(folder: str, video_details: list) -> List[Dict[str, dict | str]]:
    """Chunkify reformatted transcripts into smaller segments for embedding."""
    chunks = []
    # Initialize an empty dictionary to store the video details
    video_details_map = {}

    # Iterate through each item in the video_details list
    for detail in video_details:
        # Extract the video_id from the current detail
        video_id = detail["video_id"]

        # Add the detail to the dictionary with video_id as the key
        video_details_map[video_id] = detail

    logging.info(f"video_details_map: {video_details_map}")
    for file_name in os.listdir(folder):
        if file_name.endswith("_formatted.txt"):
            video_id = file_name.replace("_formatted.txt", "")[:-3]
            with open(os.path.join(folder, file_name), "r") as file:
                transcript = file.read()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            for chunk_text in splitter.split_text(transcript):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "video_id": video_id,
                        "title": video_details_map[video_id]["title"],
                        "upload_date": video_details_map[video_id]["upload_date"],
                        "channel_handle": video_details_map[video_id]["channel_handle"]
                    }
                })
    return chunks

def save_knowledge_base(index: faiss.IndexFlatL2, metadata: List[dict], index_to_docstore_id: Dict[int, str], docstore: InMemoryDocstore) -> None:
    """Save the FAISS index, metadata, index_to_docstore_id, and docstore to disk."""
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    with open(INDEX_TO_DOCSTORE_ID_PATH, "w") as f:
        json.dump(index_to_docstore_id, f)
    
    # Serialize docstore content
    serialized_docstore = {k: {"page_content": v.page_content, "metadata": v.metadata} for k, v in docstore._dict.items()}
    with open(DOCSTORE_PATH, "w") as f:
        json.dump(serialized_docstore, f)
    
    logging.info("Knowledge base saved.")

def load_knowledge_base() -> Tuple[faiss.IndexFlatL2, List[dict], Dict[int, str], InMemoryDocstore]:
    """Load the FAISS index, metadata, index_to_docstore_id, and docstore from disk."""
    if all(os.path.exists(path) for path in [INDEX_PATH, METADATA_PATH, INDEX_TO_DOCSTORE_ID_PATH, DOCSTORE_PATH]):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        with open(INDEX_TO_DOCSTORE_ID_PATH, "r") as f:
            index_to_docstore_id = json.load(f)
        
        # Deserialize docstore content
        with open(DOCSTORE_PATH, "r") as f:
            serialized_docstore = json.load(f)
        docstore_data = {k: Document(page_content=v["page_content"], metadata=v["metadata"]) for k, v in serialized_docstore.items()}
        docstore = InMemoryDocstore(docstore_data)
        
        logging.info("Knowledge base loaded.")
        return index, metadata, index_to_docstore_id, docstore
    else:
        # Initialize a new knowledge base if no files are found
        index = faiss.IndexFlatL2(embedding_dim)
        metadata = []
        index_to_docstore_id = {}
        docstore = InMemoryDocstore({})
        logging.info("No existing knowledge base found. Initialized a new one.")
        return index, metadata, index_to_docstore_id, docstore

def update_knowledge_base(
    chunks: List[Dict[str, dict | str]], 
    index: faiss.IndexFlatL2, 
    metadata: List[dict], 
    index_to_docstore_id: Dict[int, str],
    docstore: InMemoryDocstore, 
    video_details: list    
) -> Tuple[faiss.IndexFlatL2, list, dict, InMemoryDocstore]:
    """Update the FAISS index with new data."""
    # Extract existing video IDs from metadata
    existing_ids = {entry["metadata"]["video_id"] for entry in metadata}

    # All chunks use the same video_id, so we just have to ensure this video has not been added before
    new_chunks = []
    if len(chunks) > 0:
        if chunks[0]["metadata"]["video_id"] not in existing_ids:
            new_chunks = chunks

    if new_chunks:
        # Embed new texts using HuggingFaceEmbeddings
        texts = [chunk["text"] for chunk in new_chunks]
        logging.info(f"Generating embeddings for {len(texts)} new chunks...")
        embeddings = np.array(embedding_function.embed_documents(texts)).astype('float32')

        # Initialize the FAISS index if it's empty
        if index.ntotal == 0:
            embedding_dim = embeddings.shape[1]  # Get dimensionality from embeddings
            index = faiss.IndexFlatL2(embedding_dim)
            logging.info(f"Initialized FAISS index with dimensionality: {embedding_dim}")

        # Add new embeddings to FAISS index
        index.add(embeddings)
        logging.info(f"Added {len(new_chunks)} vectors to FAISS index. Total vectors: {index.ntotal}")

        # Extend metadata
        metadata.extend(new_chunks)
        logging.info(f"Added {len(new_chunks)} new chunks to metadata.")
    else:
        logging.info("No new data to add.")

    # Create a lookup dictionary for video details by video_id
    video_lookup = {detail["video_id"]: {"title": detail["title"], "upload_date": detail["upload_date"], "channel_handle": detail["channel_handle"]} for detail in video_details}

    # Update docstore with new chunks
    docstore_data = docstore._dict.copy()
    starting_index = len(docstore_data)  # Start from the current size of the docstore

    for i, chunk in enumerate(new_chunks):
        video_id = chunk["metadata"]["video_id"]
        if video_id in video_lookup:
            title = video_lookup[video_id]["title"]
            upload_date = video_lookup[video_id]["upload_date"]
            channel_handle = video_lookup[video_id]["channel_handle"]
            docstore_data[str(starting_index + i)] = Document(
                page_content=chunk["text"],
                metadata={
                    "video_id": video_id,
                    "title": title,
                    "upload_date": upload_date,
                    "channel_handle": channel_handle
                }
            )

    # Create a new InMemoryDocstore with updated data
    docstore = InMemoryDocstore(docstore_data)

    # Update index_to_docstore_id with new mappings
    index_to_docstore_id.update({index.ntotal - len(new_chunks) + i: str(len(metadata) - len(new_chunks) + i) for i in range(len(new_chunks))})

    return index, metadata, index_to_docstore_id, docstore

def query_and_analyze_knowledge_base(index: faiss.IndexFlatL2, metadata: List[dict], query: str, channel_handle: str) -> str:
    """Query the knowledge base and analyze the results, filtering by channel handle."""
    try:
        # Filter metadata to only include entries from the specified channel handle
        filtered_metadata = []
        logging.info(f"Filtering metadata for channel handle: {channel_handle}")
        logging.info(f"Total metadata entries before filtering: {len(metadata)}")

        for entry in metadata:
            channel = entry["metadata"]["channel_handle"]
            #logging.info(f"channel: {channel}")
            if channel == channel_handle:
                filtered_metadata.append(entry)

        logging.info(f"Total metadata entries after filtering: {len(filtered_metadata)}")

        if not filtered_metadata:
            logging.warning(f"No data found for channel: {channel_handle}")
            return f"No data found for channel: {channel_handle}"

        # Create the docstore and vectorstore
        docstore = InMemoryDocstore({
            str(i): Document(
                page_content=entry["text"],
                metadata={
                    "channel_handle": entry["metadata"]["channel_handle"],
                    "video_id": entry["metadata"]["video_id"],
                    "title": entry["metadata"]["title"],
                    "upload_date": entry["metadata"]["upload_date"]
                }
            )
            for i, entry in enumerate(filtered_metadata)
        })
        logging.info(f"Docstore created with {len(docstore._dict)} entries.")

        index_to_docstore_id = {i: str(i) for i in range(len(filtered_metadata))}
        logging.info(f"Index to docstore ID map created with {len(index_to_docstore_id)} entries.")

        # Create the FAISS vectorstore
        vectorstore = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        logging.info(f"Number of vectors in FAISS index: {vectorstore.index.ntotal}")
        logging.info(f"Query: {query}")
        
        # Perform similarity search to retrieve the top-k most similar documents
        retrieved_docs = vectorstore.similarity_search(query, k=3) # k specifies the number of results to return
        context_with_links = []

        for doc in retrieved_docs:
            video_id = doc.metadata["video_id"]
            title = doc.metadata["title"]
            upload_date = doc.metadata["upload_date"]
            
            # Reformat the upload date
            upload_date_formatted = datetime.strptime(upload_date, "%Y%m%d").strftime("%B %d, %Y")
    
            
            timestamp_str = doc.page_content.split("[")[1].split("]")[0]
            timestamp_parts = list(map(float, timestamp_str.split(":")))
            timestamp_seconds = int(timestamp_parts[0] * 3600 + timestamp_parts[1] * 60 + timestamp_parts[2])
            youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={timestamp_seconds}s"
            
            # Include upload date in the formatted output
            context_with_links.append(f"Title: {title}\n{doc.page_content}\nUpload Date: {upload_date_formatted}\nLink: {youtube_link}")
            
            # Log the title, upload date, and link
            logging.info(f"Title: {title}\nUpload Date: {upload_date_formatted}\n{doc.page_content}\nLink: {youtube_link}")

        context = "\n\n".join(context_with_links)

        if context:
            logging.info("Video Context:")
            logging.info(context)
        else:
            logging.info("No context could be generated from the query.")

        return context

    except Exception as e:
        logging.error(f"Error during query_and_analyze_knowledge_base: {str(e)}", exc_info=True)
        traceback.print_exc()

# Initialize FastAPI app
app = FastAPI(title="ChannelGPT API")

class Query(BaseModel):
    text: str
    channel_handle: Optional[str]

@app.post("/query")
async def query_endpoint(query: Query = Body(...)) -> dict:
    """Endpoint for querying the knowledge base."""
    try:
        if not query.text:
            raise HTTPException(status_code=400, detail="Query text is required")

        channel_url = f"https://www.youtube.com/@{query.channel_handle}"
        verify_channel_url(channel_url)

        video_ids = get_latest_video_ids(channel_url, 10)
        if not video_ids:
            return {"analysis": "No videos found for the provided channel URL."}

        # Fetch video details
        video_details = get_video_details(video_ids, query.channel_handle)
        logging.info(f"video_details: {video_details}")

        index, metadata, index_to_docstore_id, docstore = load_knowledge_base()

        download_transcripts(video_ids)
        reformat_transcripts("transcripts", "transcripts_formatted")
        chunks = chunkify_transcripts("transcripts_formatted", video_details)
        logging.info(f"Number of chunks created: {len(chunks)}")
        logging.info(f"First 5 chunks: {chunks[:5]}")

        index, metadata, index_to_docstore_id, docstore = update_knowledge_base(chunks, index, metadata, index_to_docstore_id, docstore, video_details)
        
        save_knowledge_base(index, metadata, index_to_docstore_id, docstore)

        if index.ntotal == 0:
            return {"analysis": "No content available in the knowledge base. Please check the channel URL and try again."}

        response = query_and_analyze_knowledge_base(index, metadata, query.text, query.channel_handle)

        return {"analysis": response}

    except Exception as e:
        # Log the detailed error traceback
        logging.error(f"Error in query_endpoint: {str(e)}", exc_info=True)
        traceback.print_exc()  # This prints the full traceback to the console
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    # Remove uvicorn.run() and rely on Docker CMD
    pass
