import pdfplumber

# Hàm đọc PDF
def read_pdf(file_path):
    try:
        # Mở file PDF bằng pdfplumber
        with pdfplumber.open(file_path) as pdf:
            # Trích xuất nội dung từ tất cả các trang
            pdf_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())

        return pdf_text
    except Exception as e:
        return f"Lỗi khi đọc PDF: {str(e)}"

# Đọc PDF
file_path = r"D:\Nam 3\Import PDF\sotxuathuyet.pdf"
pdf_text = read_pdf(file_path)

# Ghi nội dung ra file testReadPDF.txt
output_file_path = "testReadPDF.txt"
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(pdf_text)

print(f"Nội dung PDF đã được xuất ra file {output_file_path}")
