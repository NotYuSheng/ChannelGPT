# ChannelGPT

ChannelGPT is an AI-powered content analysis tool that allows users to query and analyze any YouTube channel's content using natural language. The application processes video transcripts and uses advanced language models to provide detailed analysis and insights.

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