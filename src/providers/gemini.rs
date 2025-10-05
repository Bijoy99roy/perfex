use crate::providers::LLMProvider;
use anyhow::{Ok, Result, anyhow};
use async_trait::async_trait;
use futures::TryStreamExt;
use gemini_rust::{Gemini, Model, Part};

pub struct GeminiProvider {
    client: Gemini,
    _model: String,
}

impl GeminiProvider {
    pub fn new(api_key: &str, model: Model) -> Self {
        let gemini =
            Gemini::with_model(api_key, model.clone()).expect("Failed to create Gemini client");
        Self {
            client: gemini,
            _model: model.to_string(),
        }
    }
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    async fn chat(&self, prompt: &str) -> Result<String> {
        let response = self
            .client
            .generate_content()
            .with_user_message(prompt)
            .execute()
            .await
            .map_err(|e| anyhow!("Gemini content generation error: {:?}", e))?;

        Ok(response.text().to_string())
    }

    async fn chat_stream(&self, prompt: &str) -> Result<()> {
        let mut stream = self
            .client
            .generate_content()
            .with_user_message(prompt)
            .execute_stream()
            .await
            .map_err(|e| anyhow!("Gemini streaming error: {:?}", e))?;
        println!(">>> Response: \n");
        while let Some(chunk) = stream.try_next().await? {
            for candidate in chunk.candidates {
                if let Some(parts) = candidate.content.parts {
                    for part in parts {
                        if let Part::Text { text, .. } = part {
                            print!("{}", text);
                        }
                    }
                }
            }
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }

        println!("\n");
        Ok(())
    }
}
