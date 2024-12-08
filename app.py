import gradio as gr
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8001/query"


def query_knowledge_base(channel_id, query_text):
    """Send query to FastAPI backend and return the response"""
    try:
        # Prepare the request payload
        payload = {
            "text": query_text,
            "channel_id": channel_id if channel_id.strip() else None
        }

        # Make the request to the FastAPI backend
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Extract and return the analysis
        result = response.json()
        if "detail" in result and "YouTube API error" in result["detail"]:
            return f"Error with YouTube API: {result['detail']}\nPlease check your YouTube API key configuration."
        return result["analysis"]
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the server. Please make sure the FastAPI server is running on port 8001 (python main.py)"
    except requests.exceptions.RequestException as e:
        if hasattr(e.response, 'json'):
            try:
                error_detail = e.response.json().get('detail', str(e))
                if "YouTube API error" in error_detail:
                    return f"Error with YouTube API: {error_detail}\nPlease check your YouTube API key configuration."
                return f"Error: {error_detail}"
            except:
                pass
        return f"Error: {str(e)}"


# Create Gradio interface
demo = gr.Interface(
    fn=query_knowledge_base,
    inputs=[
        gr.Textbox(
            lines=1,
            placeholder="Enter YouTube Channel ID (optional - will use default if empty)",
            label="Channel ID"
        ),
        gr.Textbox(
            lines=2,
            placeholder="Enter your query about the channel's content...",
            label="Query"
        )
    ],
    outputs=gr.Textbox(
        lines=10,
        label="Analysis"
    ),
    title="ChannelGPT - YouTube Content Analysis",
    description="Provide a detailed analysis in the form of a list, where each point starts on a new line. Clearly identify contradictions, agreements, or supporting statements. Include the YouTube links and timestamps where relevant information is found. Use structured formatting for clarity.",
    examples=[
        ["", "What are the main topics discussed in recent videos?"],
        ["", "What are the most controversial opinions expressed?"],
        ["", "Summarize the recent interviews"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Starting Gradio server. Make sure to start the FastAPI server first with: python main.py")
    demo.launch()  # share=True creates a public URL
