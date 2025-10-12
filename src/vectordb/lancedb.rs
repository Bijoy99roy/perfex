// use anyhow::{Ok, Result};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    types::Float32Type,
};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema, SchemaRef};
use futures::{StreamExt, TryStreamExt};
use std::sync::Arc;

use lancedb::{
    Connection, Result, Table, connect,
    database::CreateTableMode,
    query::{ExecutableQuery, QueryBase},
};

pub fn make_schema(dims: i32) -> Arc<Schema> {
    Arc::new(Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dims),
            false,
        ),
    ])))
}

pub fn prepare_data(
    ids: Vec<&str>,
    contents: Vec<&str>,
    titles: Vec<&str>,
    embeddings: Vec<Vec<f32>>,
    dims: i32,
    schema: SchemaRef,
) -> Vec<RecordBatch> {
    let ids_list = Arc::new(StringArray::from(ids)) as ArrayRef;
    let content_list = Arc::new(StringArray::from(contents)) as ArrayRef;
    let title_list = Arc::new(StringArray::from(titles)) as ArrayRef;

    let data: Vec<Option<Vec<Option<f32>>>> = embeddings
        .into_iter()
        .map(|row| Some(row.into_iter().map(Some).collect::<Vec<Option<f32>>>()))
        .collect();

    let embedding_list =
        Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(data, dims));
    println!("ids len {}", ids_list.len());
    println!("contents len {}", content_list.len());
    println!("titles len {}", title_list.len());
    println!("embeddings len {}", embedding_list.iter().len());
    vec![
        RecordBatch::try_new(
            schema.clone(),
            vec![ids_list, content_list, title_list, embedding_list],
        )
        .expect("Failed to create RecordBatch"),
    ]
}

pub async fn create_table(
    db_path: &str,
    table_name: &str,
    initial_data: Vec<RecordBatch>,
    schema: SchemaRef,
) -> Result<Table> {
    let db = connect(db_path).execute().await?;
    let batches = RecordBatchIterator::new(
        initial_data
            .into_iter()
            .map(|batch| -> std::result::Result<RecordBatch, ArrowError> { Ok(batch) }),
        schema.clone(),
    );

    let table = db
        .create_table(table_name, batches)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await?;
    Ok(table)
}

pub async fn execute_query(
    table: &Table,
    query_vector: &[f32],
    limit: usize,
) -> anyhow::Result<Vec<RecordBatch>> {
    let query = table.query().nearest_to(query_vector)?.limit(limit);

    let stream = query.execute().await?;

    let results = stream.try_collect().await?;
    Ok(results)
}
