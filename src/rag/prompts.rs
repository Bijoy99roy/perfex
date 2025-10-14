pub fn contruct_propmt(question: &str, context: &str) -> String {
    let system_prompt = r#"
Compose a comprehensive reply to the user query using the context given to you.
Make sure to follow below rules:
1. Please refrain from inventing answers.
2. Refuse to answer any question outside of provided context.
3. Understand the content very carefully to answer questions

Use the following parameters to answer the question:
---------

CONTEXT:
{context}

QUESTION:
{question}
"#;

    system_prompt
        .replace("{context}", context)
        .replace("{question}", question)
}
