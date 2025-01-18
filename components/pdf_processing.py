import pdfplumber

def read_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            pdf_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return pdf_text
    except Exception as e:
        return f"Lỗi khi đọc PDF: {str(e)}"


# Hàm phân tách văn bản thành các câu
def split_into_sentences(text):
    sentences = text.split(". ")  # Tách câu bằng dấu chấm và khoảng trắng
    return sentences


# Hàm để chia văn bản thành các chunk hợp lý
def chunk_text(text, chunk_size=500):
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= chunk_size:
            current_chunk += sentence + " "
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_length = sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


