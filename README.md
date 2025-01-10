# ChannelGPT

This project integrates **ChannelGPT** with **Open WebUI**, enabling natural language querying and analysis of YouTube channel content. 

To promote flexibility and reduce dependency on external services, the reliance on proprietary models like **OpenAI** has been replaced with free, open-source alternatives such as **vLLM** and **LM Studio**, allowing for independent, localized deployment. The **Google API Client Discovery** has been replaced by **yt-dlp**, a web scraping tool that extracts video data directly from YouTube’s web interface. For content analysis and embedding generation, models from **Hugging Face** are utilized, ensuring high-quality insights and accurate query handling.

<a href="#"><img alt="last-commit" src="https://img.shields.io/github/last-commit/NotYuSheng/ChannelGPT?color=red"></a>

> [!WARNING]  
> This project is incomplete and still a work in progress.

## How It Works

1. **Building the Knowledge Base**  
   The system collects data from a YouTube channel, such as video titles, descriptions, and transcripts, using **yt-dlp**. This data is processed using advanced models to create a "knowledge base"—a kind of searchable database that understands the content.

2. **Hosting the Backend Server**  
   A **FastAPI backend server** hosts the knowledge base. This server stores all the processed information and handles requests to search for relevant answers.

3. **Querying the Backend Server**  
   The **Open WebUI** provides a user-friendly interface where users can ask questions about the YouTube channel. When a user submits a question, the agent in Open WebUI sends the query to the FastAPI backend server. The server searches its knowledge base for the most relevant information and sends back an answer.

4. **Providing Answers**  
   The Open WebUI agent displays the answer to the user. This allows users to quickly get insights about a YouTube channel’s content without manually searching through videos.

## Features

- **Ask Questions Easily:** Users can type questions in plain language, and the system will provide answers based on YouTube channel content.
- **No Paid Services Needed:** The system uses free, open-source models like **vLLM**, **LM Studio**, and **Hugging Face models**, so there's no need to rely on expensive APIs.
- **Automatic Data Collection:** Video information is automatically gathered from YouTube using **yt-dlp**.
- **Backend Server with Knowledge Base:** The backend server stores and processes YouTube data, making it fast and easy to find answers.
- **Web Interface for Users:** The **Open WebUI** provides a simple interface that anyone can use to ask questions and get answers.

## Embedding Models Considered

| Model Name                                   | Description                                   | Status  |
|---------------------------------------------|---------------------------------------------|---------|
| `sentence-transformers/all-mpnet-base-v2`   | General-purpose embedding model from Sentence Transformers | ❌ Untested |
| `distiluse-base-multilingual-cased-v1`      | Multilingual embedding model from Sentence Transformers   | ❌ Untested |
| `BAAI/bge-large-en-v1.5`                    | Large English embedding model from BAAI                  | ❌ Untested |
| `BAAI/bge-base-en-v1.5`                     | Base English embedding model from BAAI                   | ✅ Tested   |

---

### Reference
This project is inspired by the original [ChannelGPT repository](https://github.com/a2ashraf/ChannelGPT) by [a2ashraf](https://github.com/a2ashraf). Significant modifications have been made, including:

- Replacing the **Gradio frontend** with **Open WebUI**.
- Replacing reliance on OpenAI models with **vLLM** or **LM Studio**.
- Replacing the Google API Client Discovery with **yt-dlp** for web scraping YouTube data.
- Using **Hugging Face models** for embedding channel content.

For a detailed presentation, refer to the [Google Slides](https://docs.google.com/presentation/d/1-fByxUlOslhKEuLHqnWeTu0N_QsHAox3wgATqQPI1qo/edit#slide=id.g32aa57a467b_0_68).
