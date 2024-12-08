import os
import json
import subprocess
import faiss
import numpy as np
from googleapiclient.discovery import build
from langchain_core.documents import Document
from webvtt import WebVTT
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from config import OPENAI_API_KEY, YOUTUBE_API_KEY, CHANNEL_ID
from langchain.schema import HumanMessage
import concurrent.futures
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional

# API keys
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Paths for saving knowledge base
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"


# Step 1: Fetch Latest Video IDs
def get_latest_video_ids(channel_id, num_videos):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.search().list(
            part="id",
            channelId=channel_id,
            order="date",
            maxResults=num_videos,
            type="video"
        ).execute()
        
        if not response.get("items"):
            print(f"Warning: No videos found for channel ID {channel_id}")
            return []
            
        return [item["id"]["videoId"] for item in response.get("items", [])]
    except Exception as e:
        print(f"Error with YouTube API: {str(e)}")
        if "quota" in str(e).lower():
            print("YouTube API quota exceeded. Please check your quota limits.")
        elif "invalid" in str(e).lower():
            print("Invalid YouTube API key. Please check your API key configuration.")
        raise HTTPException(status_code=500, detail=f"YouTube API error: {str(e)}")


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

    # Use ThreadPoolExecutor to download transcripts in parallel
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
    """Create chunks from transcripts, ensuring the metadata field is included."""
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
    return None, []


def update_knowledge_base(chunks, index, metadata):
    """Update the FAISS index with new data."""
    # Extract existing video IDs from metadata
    existing_ids = {entry["metadata"]["video_id"] for entry in metadata}

    # Filter out chunks with duplicate video IDs
    new_chunks = [chunk for chunk in chunks if chunk["metadata"]["video_id"] not in existing_ids]

    if new_chunks:
        # Embed new texts
        texts = [chunk["text"] for chunk in new_chunks]
        embedding_model = OpenAIEmbeddings()
        embeddings = np.array(embedding_model.embed_documents(texts))

        # Add new embeddings to FAISS index
        if index.ntotal == 0:
            index = faiss.IndexFlatL2(1536)
        index.add(embeddings)

        # Extend metadata
        metadata.extend(new_chunks)
        print(f"Added {len(new_chunks)} new chunks to knowledge base.")
    else:
        print("No new data to add.")

    # Update the docstore
    docstore = InMemoryDocstore({
        str(i): Document(
            page_content=chunk["text"],
            metadata={"video_id": chunk["metadata"]["video_id"]} if "metadata" in chunk and "video_id" in chunk[
                "metadata"] else {}
        )
        for i, chunk in enumerate(metadata)
    })

    # Create the index-to-docstore mapping
    index_to_docstore_id = {i: str(i) for i in range(len(metadata))}

    return index, metadata, docstore, index_to_docstore_id


# Step 6: Query the Knowledge Base
def query_knowledge_base(index, metadata, query):
    """Query the FAISS vector store and provide results with clickable YouTube URLs."""
    embedding_model = OpenAIEmbeddings()
    docstore = InMemoryDocstore({
        str(i): Document(
            page_content=entry["text"],
            metadata={"video_id": entry["metadata"]["video_id"]} if "metadata" in entry and "video_id" in entry[
                "metadata"] else {}
        )
        for i, entry in enumerate(metadata)
    })
    index_to_docstore_id = {i: str(i) for i in range(len(metadata))}

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    try:
        retrieved_docs = vectorstore.similarity_search(query, k=3)
        print("\nRetrieved Documents:")
        for doc in retrieved_docs:
            text = doc.page_content
            video_id = doc.metadata.get("video_id", "Unknown")

            # Extract timestamp from the text
            timestamp_match = text.split("]")[0].strip("[")  # Extract "[hh:mm:ss.sss]"
            timestamp_parts = timestamp_match.split(":")
            time_in_seconds = (
                    int(timestamp_parts[0]) * 3600 +
                    int(timestamp_parts[1]) * 60 +
                    int(float(timestamp_parts[2]))
            ) if len(timestamp_parts) == 3 else 0

            # Construct YouTube URL with timestamp
            url = f"https://www.youtube.com/watch?v={video_id}&t={time_in_seconds}s"

            print(f"- {text[:100]} (Video ID: {video_id})")
            print(f"  Link: {url}")

    except ValueError as e:
        print(f"Error during similarity search: {e}")

def query_and_analyze_knowledge_base(index, metadata, query):
    embedding_model = OpenAIEmbeddings()
    docstore = InMemoryDocstore({
        str(i): Document(
            page_content=entry["text"],
            metadata={"video_id": entry["metadata"]["video_id"]} if "metadata" in entry and "video_id" in entry[
                "metadata"] else {}
        )
        for i, entry in enumerate(metadata)
    })
    index_to_docstore_id = {i: str(i) for i in range(len(metadata))}

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    try:
        # Retrieve relevant chunks
        retrieved_docs = vectorstore.similarity_search(query, k=10)

        # Generate context with YouTube links
        context_with_links = []
        for doc in retrieved_docs:
            video_id = doc.metadata["video_id"].split(".")[0]
            timestamp_str = doc.page_content.split("[")[1].split("]")[0]
            timestamp_parts = list(map(float, timestamp_str.split(":")))
            timestamp_seconds = int(timestamp_parts[0] * 3600 + timestamp_parts[1] * 60 + timestamp_parts[2])
            youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={timestamp_seconds}s"
            context_with_links.append(f"{doc.page_content}\nLink: {youtube_link}")

        # Prepare the context for LLM
        context = "\n\n".join(context_with_links)

        llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Convert the prompt into the appropriate format
        messages = [
            HumanMessage(content=f"""
            Context: 
            {context}

            Question: {query}

            Provide a detailed analysis in the form of a list, where each point starts on a new line. Clearly identify contradictions, agreements, or supporting statements. Include the YouTube links and timestamps where relevant information is found. Use structured formatting for clarity.
            """)
        ]

        # Call the LLM
        analysis = llm(messages)

        print("\nAnalysis:")
        print(analysis.content)

    except ValueError as e:
        print(f"Error during similarity search: {e}")


# Use the new multithreaded function in the main workflow
def main():
    index, metadata = load_knowledge_base()
    if index is None:
        index = faiss.IndexFlatL2(1536)
        metadata = []

    channel_id = CHANNEL_ID
    video_ids = get_latest_video_ids(channel_id, 10)
    download_transcripts(video_ids)  # Updated function
    reformat_transcripts("transcripts", "transcripts_formatted")
    chunks = chunkify_transcripts("transcripts_formatted")

    index, metadata, docstore, index_to_docstore_id = update_knowledge_base(chunks, index, metadata)
    save_knowledge_base(index, metadata)

    query = "diddy sexual assault?"
    query_and_analyze_knowledge_base(index, metadata, query)


# Initialize FastAPI app
app = FastAPI(title="PiersGPT API")

class Query(BaseModel):
    text: str
    channel_id: Optional[str] = None

@app.post("/query")
async def query_endpoint(query: Query = Body(...)):
    try:
        if not query.text:
            raise HTTPException(status_code=400, detail="Query text is required")

        # Use the provided channel_id or fallback to CHANNEL_ID
        channel_id = query.channel_id or CHANNEL_ID

        # Load or create knowledge base for the channel
        index, metadata = load_knowledge_base()
        if index is None:
            index = faiss.IndexFlatL2(1536)
            metadata = []

        # Update knowledge base with latest videos
        video_ids = get_latest_video_ids(channel_id, 10)
        download_transcripts(video_ids)
        reformat_transcripts("transcripts", "transcripts_formatted")
        chunks = chunkify_transcripts("transcripts_formatted")

        index, metadata, docstore, index_to_docstore_id = update_knowledge_base(chunks, index, metadata)
        save_knowledge_base(index, metadata)

        # Capture the output from query_and_analyze_knowledge_base
        from io import StringIO
        import sys

        # Redirect stdout to capture output
        old_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result

        # Run the query
        query_and_analyze_knowledge_base(index, metadata, query.text)

        # Restore stdout and capture the result
        sys.stdout = old_stdout
        response = result.getvalue()

        return {"analysis": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

# todo:
# convert user id to channel id
# clean up code a bit to modularize it
# host it, take screenshots and post it on github and X
