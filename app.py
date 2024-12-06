import gradio as gr
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8001/query"

def query_knowledge_base(query_text):
    """Send query to FastAPI backend and return the response"""
    try:
        response = requests.post(
            API_URL,
            json={"text": query_text}
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["analysis"]
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}\nMake sure the FastAPI server is running on port 8001."

# Create Gradio interface
demo = gr.Interface(
    fn=query_knowledge_base,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter your query about Piers Morgan's content...",
        label="Query"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="Analysis"
    ),
    title="PiersGPT - Content Analysis",
    description="Ask questions about Piers Morgan's content and get AI-powered analysis based on video transcripts.",
    examples=[
        ["What does Piers think about Trump?"],
        ["Tell me about recent interviews"],
        ["What are his views on current events?"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public URL 