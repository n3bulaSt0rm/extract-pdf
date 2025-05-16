# PDF Processor with Table Detection and Section Chunking

Công cụ trích xuất văn bản, bảng biểu từ tài liệu PDF (bao gồm cả PDF scan) và chia thành các phần dựa trên cấu trúc điều, mục. Bảng biểu được chuyển đổi sang định dạng Markdown.

## Tính năng chính

- Trích xuất văn bản từ PDF thông thường và PDF scan
- Phát hiện và trích xuất bảng biểu trong PDF
- Chuyển đổi bảng thành định dạng Markdown
- Duy trì vị trí tương đối của bảng trong tài liệu
- Chia văn bản thành các phần dựa trên "Điều X", "Mục X", "Phần X"
- Hỗ trợ tiếng Việt với xử lý OCR tối ưu

## Yêu cầu hệ thống

- Python 3.7+
- Tesseract OCR v5.0+ (với dữ liệu tiếng Việt)
- Poppler

## Cài đặt

1. Cài đặt Tesseract OCR:
   - Windows: Tải và cài đặt từ [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - Đảm bảo cài đặt dữ liệu ngôn ngữ tiếng Việt (vie.traineddata)

2. Cài đặt Poppler:
   - Windows: Tải và cài đặt từ [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases)

3. Cài đặt các thư viện Python:
   ```
   pip install -r requirements.txt
   ```

4. Điều chỉnh đường dẫn trong mã:
   - Mở file `main.py` và cập nhật đường dẫn Tesseract và Poppler nếu cần

## Sử dụng

1. Đặt file PDF vào thư mục `data/pdf`
2. Chỉnh sửa danh sách `files_to_process` trong `main.py` để chỉ định:
   - Đường dẫn file PDF đầu vào
   - Đường dẫn file văn bản đầu ra
   - Loại PDF (scan hay không)

3. Chạy chương trình:
   ```
   python main.py
   ```

4. Kết quả:
   - File văn bản đầy đủ trong thư mục `output`
   - Các chunk được chia nhỏ trong thư mục `output/[tên_file]_chunks`
   - Thông tin metadata về các chunk trong file `chunks_metadata.json`

## Ví dụ kết quả

Đối với mỗi tài liệu, chương trình sẽ tạo:

1. File văn bản đầy đủ với bảng biểu trong định dạng Markdown
2. Thư mục chứa các chunk theo điều/mục
3. File metadata chứa thông tin về mỗi chunk

Ví dụ nội dung chunk:

```
Điều 1: Phạm vi điều chỉnh

Thông tư này quy định việc...

| STT | Nội dung | Ghi chú |
| --- | -------- | ------- |
| 1   | Mục A    | Chi tiết|
| 2   | Mục B    | Chi tiết|
```

## Tùy chỉnh

- Điều chỉnh DPI ảnh trong hàm `pdf_to_images()` để cân bằng chất lượng và hiệu suất
- Thay đổi mẫu phát hiện điều/mục trong hàm `chunk_by_sections()` để phù hợp với cấu trúc tài liệu
- Tối ưu tham số OCR trong hàm `extract_text_from_image()` cho kết quả tốt hơn 