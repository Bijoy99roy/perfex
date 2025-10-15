mod document_reader;
mod providers;
mod rag;
mod model;
mod utils;
mod vectordb;
use std::{env, process};

use anyhow::Result;
use gemini_rust::Model;
use inquire::{Select, Text};
use providers::{LLMProvider, openai::OpenAIClient};

use crate::{
    document_reader::pdf::read_pdf,
    providers::{gemini::GeminiProvider, groq::GroqProvider},
    rag::prompts::contruct_propmt,
    utils::document_splitter,
    vectordb::lancedb::{create_table, execute_query, make_schema, prepare_data},
    model::model_handler::load_model,
};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};

struct TaskModels {
    llm: Box<dyn LLMProvider + Send + Sync>,
    embedding: Option<Box<dyn LLMProvider + Send + Sync>>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let task_options =
        Select::new("Select what you want to do:", vec!["LLM Query", "RAG"]).prompt()?;
    
    let models = match task_options {
        "LLM Query"=>{
            let provider =
                Select::new("Select Model Provider:", vec!["OpenAI", "Gemini", "Groq"]).prompt()?;
            let llm = load_model(&provider);

            TaskModels{ llm, embedding: None}
        },
        "RAG" => {
            let provider =
                Select::new("Select Model Provider:", vec!["OpenAI", "Gemini", "Groq"]).prompt()?;
            let llm = load_model(&provider);
            let embedding_provider = "OpenAI".to_string();
            let embedding = load_model(&embedding_provider);
            TaskModels{ llm, embedding: Some(embedding)}
        },
        _=> unreachable!(),
    };
    
 
    let (embedding_model, table) = if task_options == "RAG" {
        let pdf_data = read_pdf("src/client-rfp 1.pdf")?;
        let (chunks, titles, ids) = document_splitter(&[pdf_data], 1000, 50);

        let embedding_model = models.embedding.as_ref().unwrap();
        let multiple_embeddings: Vec<Vec<f32>> = embedding_model.embed(chunks.clone()).await?;

        println!(
            "Got {} embeddings. 1st 10 values: {:?}",
            multiple_embeddings.len(),
            &multiple_embeddings[0][..multiple_embeddings[0].len().min(10)]
        );

        let dim = 1536;
        let schema = make_schema(dim);

        let batches =
            prepare_data(ids, chunks, titles, multiple_embeddings, dim, schema.clone());
        let table = create_table("./my_lancedb", "docs_embeddings", batches, schema).await?;

        (Some(embedding_model), Some(table))
    } else {
        (None, None)
    };
    loop {
        let prompt = Text::new("Enter a prompt(type 'exit' to quit):").prompt()?;
        if prompt.to_lowercase() == "exit" {
            break;
        }
        match task_options{
            "LLM Query"=>{
            models.llm.chat_stream(&prompt).await?;
        },
        "RAG" => {
            

        // TODO: Fix this atrocious code in next before weekend
        let embedding_model = embedding_model.as_ref().unwrap();
        let table = table.as_ref().unwrap();
        let query_vector:Vec<Vec<f32>> = embedding_model.embed(vec![prompt.clone()]).await?;

        let limit = 10;

        let results = execute_query(&table, &query_vector.first().unwrap(), limit).await?;

        let mut context = String::new();
        for batch in results {
            // println!("{:?}", batch);
            let content_col = batch
                .column(batch.schema().index_of("content").unwrap())
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                let content = content_col.value(i);
                context.push_str(content);
                // println!("Content: {}", content);
            }
        }
        let final_prompt = contruct_propmt(&prompt, &context);
        println!("{}", &final_prompt);
        let response = models.llm.chat_stream(&final_prompt).await?;
        },
        _=> unreachable!(),
        }
        

        
    }
    Ok(())
}
