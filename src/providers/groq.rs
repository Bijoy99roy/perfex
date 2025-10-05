use anyhow::{Ok, Result};
use async_trait::async_trait;
use futures::TryStreamExt;
use groqai::{ChatCompletionRequest, ChatMessage, GroqClient, GroqClientBuilder, Role};

use crate::providers::LLMProvider;
pub struct GroqProvider {
    client: GroqClient,
    model: String,
}

impl GroqProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        let builder = GroqClientBuilder::new(api_key.to_string())
            .expect("Failed to create GroqClientBuilder")
            .build()
            .expect("Failed to build GroqClient");

        Self {
            client: builder,
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl LLMProvider for GroqProvider {
    async fn chat(&self, prompt: &str) -> Result<String> {
        let message = vec![ChatMessage::new_text(Role::User, prompt)];
        let response = self
            .client
            .chat(&self.model)
            .messages(message)
            .send()
            .await?;

        match &response.choices[0].message.content {
            groqai::MessageContent::Text(text) => Ok(text.clone()),
            _ => Err(anyhow::anyhow!("Unexpected message content type")),
        }
    }

    async fn chat_stream(&self, prompt: &str) -> Result<()> {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::new_text(Role::User, prompt)],
            model: self.model.clone(),
            ..Default::default()
        };
        let mut response = self.client.chat_completions_stream(request).await?;
        println!(">>> Response: \n");
        while let Some(result) = response.try_next().await? {
            for choices in result.choices {
                if let Some(delta) = choices.delta.content {
                    if let groqai::MessageContent::Text(text) = delta {
                        print!("{}", text);
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
