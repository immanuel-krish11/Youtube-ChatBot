# 🎥 YouTube Transcript ChatBot

This is a Streamlit-based chatbot that lets you **ask questions about any YouTube video**, using its transcript as the knowledge base. It uses **LangChain**, **FAISS**, and **OpenAI's GPT-4o-mini** for Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 🔎 Fetches transcripts from YouTube videos
- 📚 Splits transcript into chunks and creates semantic embeddings
- 🧠 Stores and retrieves transcript context using FAISS vector search
- 🤖 Answers user questions using OpenAI's GPT-4o-mini, grounded in video content
- 💬 Interactive chat interface with Streamlit

---

## 🧠 How It Works

1. **Transcript Extraction**: The app pulls the transcript from a YouTube video.
2. **Text Splitting**: Breaks long text into smaller overlapping chunks.
3. **Embeddings & Storage**: Converts each chunk into a vector using OpenAI Embeddings and stores it in a FAISS index.
4. **Retrieval**: When a user asks a question, the most relevant chunks are retrieved.
5. **Answer Generation**: A prompt is sent to OpenAI’s GPT-4o-mini, including the retrieved context and user question.
6. **Streaming Answer**: The answer is shown in the chat UI.

---

## 🛠️ Tech Stack

| Component      | Tool/Library            |
|----------------|-------------------------|
| Frontend       | Streamlit               |
| LLM            | OpenAI GPT-4o-mini      |
| Embeddings     | `text-embedding-3-small` (OpenAI) |
| Vector Store   | FAISS                   |
| Framework      | LangChain               |
| Transcripts    | `youtube-transcript-api` |
| Env Management | python-dotenv           |

## Here's a Demo of this ChatBot - 

<img width="853" height="790" alt="Screenshot 2025-10-15 at 3 11 54 PM" src="https://github.com/user-attachments/assets/22f56634-180b-4201-b12b-88f650989161" />
