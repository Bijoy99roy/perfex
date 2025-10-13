use lopdf::Document;
#[derive(Debug)]
pub struct DocumentReader {
    pub title: String,
    pub contents: String,
}

pub fn read_pdf(path: &str) -> Result<DocumentReader, Box<dyn std::error::Error>> {
    let doc = Document::load(path)?;

    // let mut pages = Vec::new();
    let filename = path.split("/").last().unwrap_or("");
    let filename_no_ext = filename.split(".").nth(0).unwrap_or(filename);
    let title = filename_no_ext.replace(" ", "_");

    let mut pdf_document = DocumentReader {
        title: title,
        contents: "".to_string(),
    };

    let pages = doc.get_pages();
    let page_nums: Vec<u32> = pages.keys().cloned().collect();
    let full_text = doc.extract_text(&page_nums)?;
    pdf_document.contents = full_text;
    Ok(pdf_document)
}
