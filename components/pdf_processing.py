import pdfplumber
import re

# Đọc file PDF
def read_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            pdf_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return pdf_text
    except Exception as e:
        return f"Lỗi khi đọc PDF: {str(e)}"

# Phân tách văn bản thành các câu
def split_into_sentences(text):
    # Tách câu dựa trên các dấu câu kết thúc
    sentence_endings = r"(?<=[.?!])\s+"
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Chia văn bản thành chunk với overlap dựa trên câu
def chunk_text(text, chunk_size=500, overlap_sentences=2):
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # Nếu câu có thể thêm vào chunk hiện tại
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Khi chunk đầy, lưu chunk và bắt đầu chunk mới với overlap
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Overlap: Lấy một số câu cuối làm khởi đầu chunk mới
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk.append(sentence)
            current_length = sum(len(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Thêm chunk cuối

    return chunks
