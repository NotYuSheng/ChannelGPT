import requests

class Tools:
    def __init__(self):
        pass

    def query_knowledge_base(self, channel_handle: str, query_text: str) -> str:
        """
        When given a YouTube channel handle, query knowledge base for video contexts
        """
        api_url = "http://192.168.1.107:8001/query"
        payload = {"text": query_text, "channel_handle": channel_handle}

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("analysis")
        except Exception as e:
            return f"Error: {str(e)}"

# Test the tool
if __name__ == "__main__":
    tool = Tools()
    channel_handle = "MrBeast"
    query_text = "What topics does this channel cover?"
    result = tool.query_knowledge_base(channel_handle, query_text)
    print(f"Result: {result}")
