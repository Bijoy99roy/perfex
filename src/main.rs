mod providers;

use anyhow::Result;
use inquire::{Select, Text};
use providers::{LLMProvider, openai::OpenAIClient};

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Select::new("Select Model Provider:", vec!["OpenAI"]).prompt()?;

    let client: Box<dyn LLMProvider + Send + Sync> = match provider {
        "OpenAI" => Box::new(OpenAIClient::new("gpt-4o")),
        _ => unreachable!(),
    };

    loop {
        let prompt = Text::new("Enter a prompt(type 'exit' to quit):").prompt()?;
        if prompt.to_lowercase() == "exit" {
            break;
        }
        let response = client.chat_stream(&prompt).await?;
        // println!("\n>>> Resposne: \n{}\n", response);
    }
    Ok(())
}
