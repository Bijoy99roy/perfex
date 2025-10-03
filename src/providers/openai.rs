use super::LLMProvider;
use anyhow::Result;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use async_trait::async_trait;
use futures::StreamExt;
pub struct OpenAIClient {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAIClient {
    pub fn new(model: &str) -> Self {
        Self {
            client: Client::new(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAIClient {
    async fn chat(&self, prompt: &str) -> Result<String> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages([ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt)
                    .build()?,
            )])
            .build()?;

        let response = self.client.chat().create(request).await?;

        Ok(response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_else(|| "(no response)".to_string()))
    }

    async fn chat_stream(&self, prompt: &str) -> Result<()> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages([ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt)
                    .build()?,
            )])
            .build()?;
        let mut stream = self.client.chat().create_stream(request).await?;
        println!(">>> Response: \n");

        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    println!("printing chunk: {:?}", chunk);
                    if let Some(content) = &chunk.choices[0].delta.content {
                        print!("{}", content);
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();
                    }
                }
                Err(err) => eprintln!("Stream error: {err}"),
            }
        }
        println!();
        Ok(())
    }
}
