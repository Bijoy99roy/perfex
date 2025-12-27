# Perfex

Perfex is a Rust based cli for experimenting with LLM providers (OpenAI, Gemini, Groq) and vector database operations using LanceDB providing RAG(Retrieval Augumented Generation) capabilities with local files.

## Features

- [x] Chat and streaming chat with multiple LLM providers
- [x] Embedding generation (OpenAI)
- [ ] PDF document reading
- [ ] Vector database (LanceDB) ingestion
- [ ] Ask question from ingested files


## Usage

1. **Set API keys** (as needed):
   ```bash
   export OPENAI_API_KEY=your_openai_key
   export GEMINI_API_KEY=your_gemini_key
   export GROQ_API_KEY=your_groq_key
   ```

2. **Build and run**
    ```bash
    cargo run
    ```