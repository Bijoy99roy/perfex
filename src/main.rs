mod document_reader;
mod providers;
mod vectordb;
use std::{env, process};

use anyhow::Result;
use gemini_rust::Model;
use inquire::{Select, Text};
use providers::{LLMProvider, openai::OpenAIClient};

use crate::{
    document_reader::pdf::read_pdf,
    providers::{gemini::GeminiProvider, groq::GroqProvider},
    vectordb::lancedb::{create_table, execute_query, make_schema, prepare_data},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider =
        Select::new("Select Model Provider:", vec!["OpenAI", "Gemini", "Groq"]).prompt()?;

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
    };

    loop {
        let prompt = Text::new("Enter a prompt(type 'exit' to quit):").prompt()?;
        if prompt.to_lowercase() == "exit" {
            break;
        }
        let multiple_embeddings = client
            .embed(vec!["Hello world".to_string(), "How are you?".to_string()])
            .await?;

        println!(
            "Got {} embeddings. 1st 10 values: {:?}",
            multiple_embeddings.len(),
            &multiple_embeddings[0][..multiple_embeddings[0].len().min(10)]
        );
        let response = client.chat_stream(&prompt).await?;
        // println!("\n>>> Resposne: \n{}\n", response);

        // Test data ingestion with dummy data
        let dim = 4;
        let schema = make_schema(dim);

        let ids = vec!["A", "B", "C"];
        let contents = vec!["Content A", "Content B", "Content C"];
        let titles = vec!["T1", "T2", "T3"];

        let embeddings = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.9, 0.8, 0.7, 0.6],
            vec![0.4, 0.4, 0.4, 0.4],
        ];

        let batches = prepare_data(ids, contents, titles, embeddings, dim, schema.clone());
        let table = create_table("./my_lancedb", "docs_embeddings", batches, schema).await?;

        // Dummy embedding vector for testing
        let query_vector = vec![0.6, 0.7, 0.8, 0.5];

        let limit = 1;

        let results = execute_query(&table, &query_vector, limit).await?;

        for batch in results {
            println!("{:?}", batch);
        }

        let pdf_data = read_pdf("src/client-rfp 1.pdf")?;
        println!("Pdf content: \n {:?}", pdf_data);
    }
    Ok(())
}
