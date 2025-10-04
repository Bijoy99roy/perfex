mod providers;

use std::{env, process};

use anyhow::Result;
use gemini_rust::Model;
use inquire::{Select, Text};
use providers::{LLMProvider, openai::OpenAIClient};

use crate::providers::gemini::GeminiProvider;

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Select::new("Select Model Provider:", vec!["OpenAI", "Gemini"]).prompt()?;

    let client: Box<dyn LLMProvider + Send + Sync> = match provider {
        "OpenAI" => {
            if !env::var("OPENAI_API_KEY").is_ok() {
                eprintln!(
                    "ERROR: OPENAI_API_KEY is missing. Please set it before running the program."
                );
                process::exit(1);
            }

            Box::new(OpenAIClient::new("gpt-4o"))
        }
        "Gemini" => {
            let gemini_api_key = match env::var("GEMINI_API_KEY") {
                Ok(val) => val,
                Err(_) => {
                    eprintln!(
                        "ERROR: GEMINI_API_KEY is missing. Please set it before running the program."
                    );
                    process::exit(1);
                }
            };
            Box::new(GeminiProvider::new(
                &gemini_api_key,
                Model::Gemini25FlashLite,
            ))
        }
        _ => unreachable!(),
    };

    loop {
        let prompt = Text::new("Enter a prompt(type 'exit' to quit):").prompt()?;
        if prompt.to_lowercase() == "exit" {
            break;
        }
        println!("{}", Model::Gemini25FlashLite.to_string());
        let response = client.chat_stream(&prompt).await?;
        // println!("\n>>> Resposne: \n{}\n", response);
    }
    Ok(())
}
