import requests

class Tools:
    def __init__(self):
        pass

    def query_knowledge_base(self, channel_handle: str, query_text: str) -> str:
        """
        Query knowledge base for relevant video contexts
        :param channel_handle: The handle of the YouTube channel to query.
        :param query_text: The natural language query to send to the knowledge base.
        :return: The query result or an error message.
        """
        api_url = "http://192.168.133.130:8001/query"

        payload = {
            "text": query_text,
            "channel_handle": channel_handle
        }

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            if "detail" in result and "YouTube API error" in result["detail"]:
                return f"Error with YouTube API: {result['detail']}\nPlease check your YouTube API key configuration."
            return result.get("analysis", "No analysis found.")
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to the server. Please ensure the FastAPI server is running."
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

# Test the tool
if __name__ == "__main__":
    tool = Tools()
    channel_handle = "MrBeast"
    query_text = "What topics does this channel cover?"
    result = tool.query_knowledge_base(channel_handle, query_text)
    print(f"Result: {result}")
