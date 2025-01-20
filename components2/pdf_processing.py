import pdfplumber
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

def read_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            pdf_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return pdf_text
    except Exception as e:
        return f"Lỗi khi đọc PDF: {str(e)}"

# Hàm để chia văn bản thành các chunk hợp lý
def chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


