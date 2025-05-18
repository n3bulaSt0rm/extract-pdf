import os
import re
import torch
from pathlib import Path

# Import docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Import trực tiếp từ docling-ibm-models - sửa lại đường dẫn import cho đúng
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor

def extract_text_from_pdf(pdf_path, output_path, preserve_tables=True, handle_scanned=True):
    """
    Extract text from PDF files (including scanned PDFs with tables) using Docling.
    Tables are preserved in Markdown format, regular text is plain text.
    Optimized for Vietnamese language content.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path to the output file
        preserve_tables (bool): Whether to preserve tables in Markdown format
        handle_scanned (bool): Whether to enable OCR for scanned PDFs
    
    Returns:
        str: Path to the output file
    """
    print(f"Processing: {pdf_path}")
    
    # Configure GPU/CPU acceleration
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU acceleration.")
        print(f"CUDA version: {torch.version.cuda}")
        acc_device = AcceleratorDevice.CUDA
    else:
        print("CUDA is NOT available. Using CPU.")
        acc_device = AcceleratorDevice.CPU

    # Initialize the document converter with minimal configuration
    # Only adding GPU acceleration to the default setup
    try:
        # Cấu hình GPU đúng cách thông qua AcceleratorOptions
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=acc_device
        )
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        
        # Cấu hình để xử lý tốt bảng
        pipeline_options.do_table_structure = True
        
        # Khởi tạo DocumentConverter với cấu hình GPU
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        
        # Convert the PDF
        result = converter.convert(pdf_path)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get content in Markdown format to access tables
        markdown_content = result.document.export_to_markdown()
        
        if preserve_tables:
            # Process the markdown content to keep tables as markdown and simplify regular text
            # Define regex patterns for Markdown tables
            table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'
            
            # Function to process each match (table)
            def replace_with_table(match):
                return match.group(0)
            
            # Replace tables with placeholders
            processed_content = re.sub(table_pattern, replace_with_table, markdown_content)
            
            # Simplify headers (remove # symbols but keep structure)
            processed_content = re.sub(r'^(#+)\s+(.+)$', r'\2', processed_content, flags=re.MULTILINE)
            
            # Write the processed content to the output file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(processed_content)
        else:
            # Just write the markdown content as is
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
        
        print(f"Extracted content saved to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write error message to output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Error extracting content from {pdf_path}: {str(e)}")
        
        return output_path

def extract_text_from_image(image_path, output_path):
    """
    Extract text from scanned images using Docling OCR.
    
    Args:
        image_path (str): Path to the image file
        output_path (str): Path to the output file
    
    Returns:
        str: Path to the output file
    """
    print(f"Processing image: {image_path}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU acceleration.")
        print(f"CUDA version: {torch.version.cuda}")
        acc_device = AcceleratorDevice.CUDA
    else:
        print("CUDA is NOT available. Using CPU.")
        acc_device = AcceleratorDevice.CPU
    
    try:
        # Cấu hình GPU đúng cách
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=acc_device
        )
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        
        # Khởi tạo DocumentConverter với cấu hình GPU
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        
        # Convert the image
        result = converter.convert(image_path)
        
        # Get content in Markdown format
        markdown_content = result.document.export_to_markdown()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write the markdown content to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"Extracted content saved to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write error message to output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Error extracting content from {image_path}: {str(e)}")
        
        return output_path

# Hàm mới sử dụng trực tiếp docling-ibm-models
def extract_tables_with_ibm_models(pdf_path, output_path):
    """
    Trích xuất bảng từ PDF sử dụng trực tiếp các mô hình từ docling-ibm-models
    
    Args:
        pdf_path (str): Đường dẫn đến file PDF
        output_path (str): Đường dẫn đến file đầu ra
        
    Returns:
        str: Đường dẫn đến file đầu ra
    """
    print(f"Đang xử lý PDF bằng ibm-models trực tiếp: {pdf_path}")
    
    # Kiểm tra GPU
    if torch.cuda.is_available():
        print(f"Sử dụng GPU với CUDA {torch.version.cuda}")
        gpu_device = "cuda"
    else:
        print("Không có GPU, sử dụng CPU.")
        gpu_device = "cpu"
    
    try:
        # Khởi tạo mô hình layout để phát hiện vị trí bảng
        # artifact_path là None để sử dụng weights mặc định
        layout_predictor = LayoutPredictor(artifact_path=None, device=gpu_device)
        
        # Khởi tạo TableFormer để nhận dạng cấu trúc bảng
        tableformer_predictor = TFPredictor(device=gpu_device)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Đọc PDF và chuyển đổi thành hình ảnh
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path)
        
        tables_content = []
        
        for i, img in enumerate(images):
            print(f"Đang xử lý trang {i+1}/{len(images)}")
            
            # Lưu hình ảnh tạm thời
            temp_img_path = os.path.join(output_dir, f"temp_page_{i+1}.png")
            img.save(temp_img_path)
            
            # Phát hiện vị trí bảng bằng mô hình Layout
            layout_result = layout_predictor.predict(temp_img_path)
            
            # Tìm các vùng có nhãn là bảng
            table_regions = []
            for item in layout_result["prediction"]:
                if item["label"] == "table":
                    table_regions.append(item["bbox"])
            
            if table_regions:
                print(f"Đã phát hiện {len(table_regions)} bảng trên trang {i+1}")
                
                # Xử lý từng vùng bảng
                for j, table_bbox in enumerate(table_regions):
                    # Cắt hình ảnh chỉ lấy phần bảng
                    table_img = img.crop(table_bbox)
                    table_img_path = os.path.join(output_dir, f"table_{i+1}_{j+1}.png")
                    table_img.save(table_img_path)
                    
                    # Sử dụng TableFormer để nhận dạng cấu trúc bảng
                    table_result = tableformer_predictor.predict(table_img_path)
                    
                    # Chuyển đổi kết quả thành bảng Markdown
                    markdown_table = convert_table_to_markdown(table_result)
                    tables_content.append(f"## Bảng {i+1}.{j+1}\n\n{markdown_table}\n\n")
                    
                    # Xóa file tạm
                    os.remove(table_img_path)
            
            # Xóa file tạm
            os.remove(temp_img_path)
        
        # Ghi nội dung vào file đầu ra
        with open(output_path, "w", encoding="utf-8") as f:
            if tables_content:
                f.write("# Các bảng được trích xuất\n\n")
                f.write("".join(tables_content))
            else:
                f.write("Không tìm thấy bảng nào trong tài liệu PDF.")
        
        print(f"Kết quả đã được lưu vào: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Lỗi khi xử lý PDF: {str(e)}")
        
        # Tạo thư mục đầu ra nếu cần
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Ghi thông báo lỗi vào file đầu ra
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Lỗi khi trích xuất nội dung từ {pdf_path}: {str(e)}")
        
        return output_path

def convert_table_to_markdown(table_result):
    """
    Chuyển đổi kết quả TableFormer thành bảng Markdown
    
    Args:
        table_result (dict): Kết quả từ TableFormerPredictor
        
    Returns:
        str: Bảng ở định dạng Markdown
    """
    try:
        # Lấy cấu trúc bảng và nội dung từ kết quả
        table_structure = table_result["table_structure"]
        
        # Tạo bảng Markdown
        markdown_rows = []
        
        # Thêm header
        header_row = "| " + " | ".join(cell["content"] for cell in table_structure[0]) + " |"
        markdown_rows.append(header_row)
        
        # Thêm dòng ngăn cách
        separator = "| " + " | ".join(["---"] * len(table_structure[0])) + " |"
        markdown_rows.append(separator)
        
        # Thêm các dòng dữ liệu
        for row in table_structure[1:]:
            data_row = "| " + " | ".join(cell["content"] for cell in row) + " |"
            markdown_rows.append(data_row)
        
        return "\n".join(markdown_rows)
    except Exception as e:
        print(f"Lỗi khi chuyển đổi bảng sang Markdown: {str(e)}")
        return "Không thể chuyển đổi bảng sang định dạng Markdown."

def main():
    """
    Main function to extract text from PDF file.
    Uses fixed paths for input and output files.
    """
    # Define fixed paths
    INPUT_PATH = r"D:\DATN_HUST\test\data\raw\pdf\QCDT-2023-upload.pdf"
    OUTPUT_PATH = r"D:\DATN_HUST\test\output.txt"
    
    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file does not exist: {INPUT_PATH}")
        return
    
    # Process the PDF file using standard docling method
    try:
        extract_text_from_pdf(
            INPUT_PATH, 
            OUTPUT_PATH, 
            preserve_tables=True,
            handle_scanned=True
        )
        print(f"Docling conversion completed! Check the output at: {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main() 