use crate::document_reader::pdf::DocumentReader;
use uuid::Uuid;
pub fn document_splitter(
    data: &[DocumentReader],
    chunk_size: usize,
    overlap: usize,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut chunks = Vec::new();
    let mut titles = Vec::new();
    for document in data.iter() {
        let text = &document.contents;
        let title = &document.title;
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + chunk_size).min(chars.len());
            let chunk = chars[start..end].iter().collect();
            chunks.push(chunk);
            titles.push(title.clone());
            start = start + chunk_size - overlap;
        }
    }

    let uuids: Vec<String> = (0..chunks.len())
        .map(|_| Uuid::new_v4().to_string())
        .collect();

    (chunks, titles, uuids)
}
