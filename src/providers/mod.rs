use anyhow::{Result, anyhow};
use async_trait::async_trait;

#[async_trait]
pub trait LLMProvider {
    async fn chat(&self, prompt: &str) -> Result<String>;
    async fn chat_stream(&self, prompt: &str) -> Result<()>;
    async fn embed(&self, _inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Err(anyhow!("Embeddings not supported for this provider"))
    }
}

pub mod gemini;
pub mod groq;
pub mod openai;
