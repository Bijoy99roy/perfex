use anyhow::Result;
use arrow_array::{ArrayRef, Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use gemini_rust::batch;
use std::sync::Arc;

use lancedb::{Table, connect, database::CreateTableMode};

pub fn make_schema(dims: i32) -> Schema {
    Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dims),
            false,
        ),
    ]))
}

pub fn prepare_data(
    ids: Vec<&str>,
    contents: Vec<&str>,
    titles: Vec<&str>,
    embeddings: Vec<Vec<f32>>,
    dims: i32,
) -> Vec<RecordBatch> {
    let ids_list = Arc::new(StringArray::from(ids)) as ArrayRef;
    let content_list = Arc::new(StringArray::from(contents)) as ArrayRef;
    let title_list = Arc::new(StringArray::from(titles)) as ArrayRef;

    let mut flat: Vec<f32> = vec![];

    for vec in &embeddings {
        assert_eq!(vec.len(), dims as usize, "Embedding dimension mismatch");
        flat.extend_from_slice(&vec[..]);
    }

    let embedding_list = Arc::new(Float32Array::from(flat)) as ArrayRef;
    vec![
        RecordBatch::try_from_iter(vec![
            ("id", ids_list),
            ("content", content_list),
            ("title", title_list),
            ("embedding", embedding_list),
        ])
        .expect("Failed to create RecordBatch"),
    ]
}

pub async fn create_table(
    db_path: &str,
    table_name: &str,
    initial_data: Vec<RecordBatch>,
    schema: SchemaRef,
) -> Result<()> {
    let db = connect(db_path).execute().await?;
    let batches = RecordBatchIterator::new(initial_data.into_iter().map(Ok), schema.clone());
    db.create_table(table_name, batches)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await?;
    Ok(())
}
