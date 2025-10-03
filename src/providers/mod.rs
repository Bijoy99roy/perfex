use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait LLMProvider {
    async fn chat(&self, prompt: &str) -> Result<String>;
    async fn chat_stream(&self, prompt: &str) -> Result<()>;
}

pub mod openai;
