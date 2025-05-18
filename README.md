# Trích xuất văn bản tiếng Việt từ PDF (Extract Vietnamese Text from PDF)

## Tiếng Việt

Công cụ này giúp trích xuất văn bản tiếng Việt từ các file PDF, bao gồm cả PDF scan và PDF có bảng biểu, sử dụng thư viện Docling.

### Yêu cầu

- Python 3.8+
- Thư viện Docling

### Cài đặt

1. Kích hoạt môi trường ảo:
```
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

2. Cài đặt các gói cần thiết:
```
pip install -r requirements.txt
```

### Sử dụng

Trích xuất từ một file PDF đơn lẻ:
```
python extract_vietnamese_pdf.py đường_dẫn_đến_file.pdf
```

Trích xuất từ tất cả các file PDF trong một thư mục:
```
python extract_vietnamese_pdf.py đường_dẫn_đến_thư_mục
```

Các tùy chọn:
- `-o`, `--output`: Thư mục đầu ra (mặc định: cùng thư mục với file đầu vào)
- `-f`, `--format`: Định dạng đầu ra (markdown, html, json) (mặc định: markdown)

Ví dụ:
```
python extract_vietnamese_pdf.py tài_liệu.pdf -o kết_quả -f html
```

## English

This tool helps extract Vietnamese text from PDF files, including scanned PDFs and PDFs with tables, using the Docling library.

### Requirements

- Python 3.8+
- Docling library

### Installation

1. Activate the virtual environment:
```
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

2. Install required packages:
```
pip install -r requirements.txt
```

### Usage

Extract from a single PDF file:
```
python extract_vietnamese_pdf.py path_to_file.pdf
```

Extract from all PDF files in a directory:
```
python extract_vietnamese_pdf.py path_to_directory
```

Options:
- `-o`, `--output`: Output directory (default: same as input)
- `-f`, `--format`: Output format (markdown, html, json) (default: markdown)

Example:
```
python extract_vietnamese_pdf.py document.pdf -o results -f html
``` 