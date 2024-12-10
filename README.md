# ChannelGPT

ChannelGPT is an AI-powered content analysis tool that allows users to query and analyze any YouTube channel's content using natural language. The application processes video transcripts and uses advanced language models to provide detailed analysis and insights.

SEE DEMO VIDEO: https://youtu.be/NSxQu9Pn-Cc
## Features

- Analyze any YouTube channel by providing its Channel ID
- Query video content using natural language
- AI-powered analysis of transcripts
- User-friendly web interface
- RESTful API endpoints
- Real-time content updates

## Prerequisites

- Python 3.8+
- OpenAI API key
- YouTube API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ChannelGPT
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a configuration file:
   - Copy `config.template.py` to `config.py`
   - Add your API keys to `config.py`:
     ```python
     OPENAI_API_KEY = "your_openai_api_key"
     YOUTUBE_API_KEY = "your_youtube_api_key"
     CHANNEL_ID = "UCatt7TBjfBkiJWx8khav_Gg"
     ```

## Running the Application

The application consists of two components that need to be running simultaneously:

1. Start the FastAPI backend server:
```bash
uvicorn main:app --reload --port 8001
```
to reload server after changes:
```bash
uvicorn main:app --reload
```

2. In a new terminal, start the Gradio web interface:
```bash
python app.py
```

The application will be available at:
- FastAPI backend: http://localhost:8001
- Gradio interface: http://localhost:7860 (default)

## Usage

1. Access the Gradio web interface in your browser
2. Enter the YouTube Channel ID you want to analyze
   - You can find a channel's ID by:
     - Going to the channel's page
     - Right-clicking and viewing page source
     - Searching for "channelId"
     - Or using online tools like [Comment Picker](https://commentpicker.com/youtube-channel-id.php)
3. Enter your query in the text box
4. Click "Submit" to get AI-powered analysis
5. Use example queries for inspiration

## API Endpoints

- POST `/query`
  - Endpoint for querying the knowledge base
  - Request body: `{"channel_id": "channel_id_here", "text": "your query here"}`
  - Returns analysis based on video transcripts

## Project Structure

- `main.py`: FastAPI backend and core functionality
- `app.py`: Gradio web interface
- `config.py`: Configuration and API keys
- `requirements.txt`: Project dependencies

## Security Note

Never commit your `config.py` file with actual API keys. It's included in `.gitignore` for security.

## Algorithm for the code is defined below:
- Initialization:
    - Set API keys for OpenAI and YouTube in the environment.
    - Define paths for saving the knowledge base (FAISS index and metadata).

- Fetch Latest Video IDs:
  - Use the YouTube API to fetch the latest video IDs from a specified channel. This involves:
  Verifying the existence of the channel.
  - Retrieving the most recent videos based on the channel ID and sorting them by date.

- Download Transcripts:
  - For each video ID, download the corresponding video transcripts using yt-dlp. This is done concurrently using a thread pool to speed up the process.
 
- Reformat Transcripts:
  - Convert the downloaded VTT files into a more usable text format. This involves:
    - Reading each VTT file.
    - Extracting timestamps and text, and formatting them into a cleaner structure.

- Chunkify Transcripts:
  - Break down the formatted transcripts into smaller chunks. This is necessary for processing large texts and improving the manageability of data. Each chunk includes:
  The text of the chunk.
  - Metadata containing the video ID.

- Knowledge Base Management:
  - Load Knowledge Base: Load existing FAISS index and metadata if available.
  - Update Knowledge Base: Add new chunks to the FAISS index and update the metadata. This includes:
    - Filtering out chunks from videos already present in the metadata.
    - Embedding the text of new chunks using OpenAI's embeddings.
    - Adding new embeddings to the FAISS index.
    - Updating the in-memory document store with new chunks.
  - Save Knowledge Base: Save the updated FAISS index and metadata to disk.
  
- Query the Knowledge Base:
  - Perform a similarity search in the FAISS vector store using a query. This involves:
    - Embedding the query using OpenAI's model.
    - Retrieving the most similar documents.
    - Formatting the results to include clickable YouTube links with timestamps.

- API and Web Interface:
  - FastAPI Setup: Define an API endpoint that accepts queries. This endpoint handles:
    - Validation of the channel ID.
    - Loading or updating the knowledge base with the latest videos.
    - Performing the query and returning the analysis.
  - Gradio Interface: Although not explicitly detailed in the provided code, a Gradio interface would interact with this API to provide a user-friendly interface for submitting queries and displaying results.

- Execution:
  - The main function orchestrates the above steps when run locally, using a predefined channel ID.
  - When deployed as a web service, the API endpoint can dynamically accept different channel IDs and queries from users.