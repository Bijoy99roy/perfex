use crate::document_reader::pdf::DocumentReader;

pub fn document_splitter(
    data: &[DocumentReader],
    chunk_size: usize,
    overlap: usize,
) -> Vec<String> {
    let mut chunks = Vec::new();

    for document in data.iter() {
        let text = &document.contents;
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + chunk_size).min(chars.len());
            let chunk = chars[start..end].iter().collect();
            chunks.push(chunk);

            start = start + chunk_size - overlap;
        }
    }

    chunks
}
