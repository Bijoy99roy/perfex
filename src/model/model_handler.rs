use crate::providers::{LLMProvider, openai::OpenAIClient, gemini::GeminiProvider, groq::GroqProvider};
use gemini_rust::Model;
use std::{env, process};
pub fn load_model(provider: &str) -> Box<dyn LLMProvider + Send + Sync>{
    match provider {
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
        "Groq" => {
            let groq_api_key = match env::var("GROQ_API_KEY") {
                Ok(val) => val,
                Err(_) => {
                    eprintln!(
                        "ERROR: GROQ_API_KEY is missing. Please set it before running the program."
                    );
                    process::exit(1);
                }
            };
            Box::new(GroqProvider::new(&groq_api_key, "llama-3.3-70b-versatile"))
        }
        _ => unreachable!(),
    }
}