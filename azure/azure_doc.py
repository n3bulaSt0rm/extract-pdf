import os
import tempfile
import base64
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import json

class AzureDocumentProcessor:
    def __init__(self, endpoint: str = None, key: str = None):
        """
        Khởi tạo bộ xử lý tài liệu Azure Document Intelligence
        """
        self.endpoint = "https://document-exactors.cognitiveservices.azure.com/"
        self.key = "FrfoN2G4pLc3xqJahH1MBskg7ROmzEPfskLgD560yn9R8p3h2boMJQQJ99BEACYeBjFXJ3w3AAALACOGfLQd"
        
        if not self.endpoint or not self.key:
            raise ValueError("Thông tin xác thực Azure Document Intelligence không được cấu hình")
        
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint, 
            credential=AzureKeyCredential(self.key)
        )
    
    def analyze_document(self, file_path: str, model_id: str = "prebuilt-layout") -> AnalyzeResult:
        """
        Phân tích tài liệu PDF sử dụng Azure Document Intelligence
        
        Args:
            file_path: Đường dẫn đến file PDF
            model_id: ID của model phân tích (mặc định: prebuilt-layout)
        
        Returns:
            Kết quả phân tích từ Azure Document Intelligence
        """
        print(f"Đang phân tích tài liệu: {file_path}")
        print(f"Sử dụng model: {model_id}")
        
        with open(file_path, "rb") as f:
            document_bytes = f.read()
        
        # Tối ưu cho PDF thông thường (không scan)
        # Sử dụng prebuilt-layout để phân tích bố cục và bảng biểu tốt hơn
        # Không cần ocrHighResolution vì PDF thông thường đã có text layer
        poller = self.client.begin_analyze_document(
            model_id, 
            body=document_bytes,
            content_type="application/pdf",
            locale="vi-VN",
            features=["keyValuePairs", "languages"],
            string_index_type="unicodeCodePoint"
        )
        
        print("Đang chờ kết quả từ Azure Document Intelligence...")
        result = poller.result()
        return result
    
    def save_raw_tables(self, result: AnalyzeResult, output_path: str):
        """
        Lưu kết quả bảng gốc từ Azure Document Intelligence
        
        Args:
            result: Kết quả phân tích từ Azure Document Intelligence
            output_path: Đường dẫn để lưu file kết quả bảng
        """
        # Lưu thông tin bảng dưới dạng text
        with open(output_path, "w", encoding="utf-8") as f:
            if hasattr(result, 'tables') and result.tables:
                for i, table in enumerate(result.tables):
                    f.write(f"BẢNG {i+1}:\n")
                    f.write(f"Số hàng: {table.row_count}, Số cột: {table.column_count}\n")
                    
                    # Lưu dữ liệu ô dưới dạng nguyên bản
                    f.write("DỮ LIỆU BẢNG:\n")
                    for cell in table.cells:
                        f.write(f"Hàng {cell.row_index}, Cột {cell.column_index}: {cell.content}\n")
                    
                    f.write("\n" + "-"*50 + "\n\n")
            else:
                f.write("Không tìm thấy bảng nào trong tài liệu.")
        
        print(f"Đã lưu kết quả bảng gốc vào: {output_path}")
    
    def save_raw_text(self, result: AnalyzeResult, output_path: str):
        """
        Lưu kết quả văn bản gốc từ Azure Document Intelligence
        
        Args:
            result: Kết quả phân tích từ Azure Document Intelligence
            output_path: Đường dẫn để lưu file kết quả văn bản
        """
        # Lưu thông tin văn bản dưới dạng text
        with open(output_path, "w", encoding="utf-8") as f:
            if hasattr(result, 'paragraphs') and result.paragraphs:
                for paragraph in result.paragraphs:
                    f.write(f"{paragraph.content}\n")
            else:
                f.write("Không tìm thấy văn bản nào trong tài liệu.")
        
        print(f"Đã lưu kết quả văn bản gốc vào: {output_path}")

def main():
    """
    Hàm chính để xử lý tài liệu PDF
    """
    # Đường dẫn cố định đến file PDF cần xử lý
    pdf_path = "D:\\DATN_HUST\\test\\data\\raw\\pdf\\QCDT-2023-upload.pdf"
    
    # Kiểm tra sự tồn tại của file
    if not os.path.exists(pdf_path):
        print(f"Không tìm thấy file: {pdf_path}")
        return
    
    # Đường dẫn cố định đến các file kết quả
    tables_output = "D:\\DATN_HUST\\test\\azure\\original_table.txt"
    text_output = "D:\\DATN_HUST\\test\\azure\\original_text.txt"
    
    # Khởi tạo bộ xử lý tài liệu
    processor = AzureDocumentProcessor()
    
    print("Bắt đầu xử lý tài liệu PDF...")
    
    # Phân tích tài liệu
    result = processor.analyze_document(pdf_path)
    
    # Lưu kết quả gốc cho bảng và văn bản
    processor.save_raw_tables(result, tables_output)
    processor.save_raw_text(result, text_output)
    
    print(f"Đã hoàn thành xử lý tài liệu.")
    print(f"Kết quả bảng gốc từ Azure: {tables_output}")
    print(f"Kết quả văn bản gốc từ Azure: {text_output}")

if __name__ == "__main__":
    main()
