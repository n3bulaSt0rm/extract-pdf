import os
import sys
import re
import torch
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Add docling_ibm_models to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
docling_ibm_models_path = os.path.join(current_dir, "docling_ibm_models")
sys.path.append(docling_ibm_models_path)

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

def perform_ocr(image, lang='vie'):
    """
    Thực hiện OCR trên ảnh để nhận dạng văn bản tiếng Việt
    
    Args:
        image: Ảnh PIL hoặc numpy array
        lang: Ngôn ngữ OCR (mặc định là tiếng Việt)
        
    Returns:
        str: Văn bản được nhận dạng
    """
    try:
        # Chuyển đổi numpy array thành PIL Image nếu cần
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Thực hiện OCR với cấu hình cho tiếng Việt
        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config='--psm 6'  # Assume uniform text block
        )
        
        # Xử lý kết quả OCR
        text = text.strip()
        return text if text else ""
        
    except Exception as e:
        print(f"Lỗi khi thực hiện OCR: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path, output_path, preserve_tables=True):
    """
    Extract text from PDF files (including scanned PDFs with tables) using Docling.
    Tables are preserved in Markdown format, regular text is plain text.
    Optimized for Vietnamese language content.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path to the output file
        preserve_tables (bool): Whether to preserve tables in Markdown format
    
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

# Hàm mới sử dụng trực tiếp docling-ibm-models
def extract_tables_with_ibm_models(pdf_path, output_path):
    """
    Extract tables from PDF using docling-ibm-models with enhanced capabilities.
    This function specializes in table detection and extraction using IBM's specialized models.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path to the output file
        
    Returns:
        str: Path to the output file
    """
    print(f"Processing PDF with enhanced IBM models: {pdf_path}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"Using GPU with CUDA {torch.version.cuda}")
        gpu_device = "cuda"
    else:
        print("No GPU available, using CPU.")
        gpu_device = "cpu"
    
    try:
        # Get the path to model weights
        current_dir = os.path.dirname(os.path.abspath(__file__))
        layout_model_path = os.path.join(current_dir, "model_artifacts", "layout")
        tableformer_model_path = os.path.join(current_dir, "model_artifacts", "tableformer", "accurate")
        
        # Verify model files exist
        layout_model_file = os.path.join(layout_model_path, "model.safetensors")
        tableformer_model_file = os.path.join(tableformer_model_path, "tableformer_accurate.safetensors")
        
        if not os.path.exists(layout_model_file):
            raise FileNotFoundError(f"Layout model file not found: {layout_model_file}")
        if not os.path.exists(tableformer_model_file):
            raise FileNotFoundError(f"TableFormer model file not found: {tableformer_model_file}")
        
        print(f"Using layout model from: {layout_model_path}")
        print(f"Using TableFormer model from: {tableformer_model_path}")
        
        # Initialize layout predictor with correct parameters
        layout_predictor = LayoutPredictor(
            artifact_path=layout_model_path,
            device=gpu_device,
            num_threads=4,
            base_threshold=0.3,
            blacklist_classes={"Form", "Key-Value Region"}
        )
        
        # Load TableFormer config
        with open(os.path.join(tableformer_model_path, "tm_config.json"), "r") as f:
            tableformer_config = json.load(f)
        
        # Update config with model path and other parameters
        tableformer_config["model"]["save_dir"] = tableformer_model_path
        tableformer_config["predict"] = {
            "max_steps": 100,
            "beam_size": 5,
            "bbox": True,
            "padding": True,
            "padding_size": 10,
            "profiling": False,
            "pdf_cell_iou_thres": 0.5,
            "table_cell_iou_thres": 0.5,
            "correct_overlapping_cells": True,
            "sort_row_col_indexes": True,
            "enable_ocr": True,
            "ocr_lang": "vie",
            "ocr_config": {
                "psm": 6,  # Assume uniform text block
                "oem": 1,  # Use LSTM OCR Engine Mode
                "dpi": 300,
                "config": "--oem 1 --psm 6 -l vie"
            }
        }
        
        # Initialize TableFormer predictor with config
        tableformer_predictor = TFPredictor(
            config=tableformer_config,
            device=gpu_device,
            num_threads=4
        )
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images with better quality
        images = convert_from_path(
            pdf_path,
            dpi=300,
            thread_count=4
        )
        
        tables_content = []
        page_metadata = []
        
        for i, img in enumerate(images):
            print(f"\nProcessing page {i+1}/{len(images)}")
            
            # Convert PIL Image to numpy array
            img_array = np.array(img)
            
            # Save temporary image with better quality
            temp_img_path = os.path.join(output_dir, f"temp_page_{i+1}.png")
            img.save(temp_img_path, quality=95)
            
            # Detect table regions using layout model
            layout_result = layout_predictor.predict(img_array)
            
            # Extract table regions with confidence scores and additional metadata
            table_regions = []
            for item in layout_result:
                if item["label"] == "Table" and item.get("confidence", 0) > 0.5:
                    table_regions.append({
                        "bbox": [item["l"], item["t"], item["r"], item["b"]],
                        "confidence": item.get("confidence", 0),
                        "type": "Standard",
                        "page": i + 1
                    })
            
            if table_regions:
                print(f"Detected {len(table_regions)} tables on page {i+1}")
                
                # Process each table region
                for j, table_info in enumerate(table_regions):
                    table_bbox = table_info["bbox"]
                    confidence = table_info["confidence"]
                    
                    # Crop image to table region with padding
                    padding = 10
                    x1, y1, x2, y2 = table_bbox
                    x1 = max(0, int(x1 - padding))
                    y1 = max(0, int(y1 - padding))
                    x2 = min(img.width, int(x2 + padding))
                    y2 = min(img.height, int(y2 + padding))
                    
                    table_img = img.crop((x1, y1, x2, y2))
                    table_img_path = os.path.join(output_dir, f"table_{i+1}_{j+1}.png")
                    table_img.save(table_img_path, quality=95)
                    
                    # Convert table image to numpy array for TableFormer
                    table_img_array = np.array(table_img)
                    
                    print(f"\nDebug - Processing table {i+1}.{j+1}:")
                    print(f"1. Table dimensions: {table_img.width}x{table_img.height}")
                    print(f"2. Table bbox: {table_bbox}")
                    
                    # Create iocr_page dictionary with required information
                    iocr_page = {
                        "image": table_img_array,
                        "tokens": [],
                        "width": table_img.width,
                        "height": table_img.height,
                        "ocr_config": tableformer_config["predict"]["ocr_config"]
                    }
                    
                    # Use TableFormer to recognize table structure
                    table_result = tableformer_predictor.predict(
                        iocr_page,
                        table_bbox,
                        table_img_array,
                        1.0,
                        None,
                        True
                    )
                    
                    # Debug output
                    print(f"3. TableFormer result:")
                    print(f"   - tf_output length: {len(table_result[0]) if table_result[0] else 0}")
                    print(f"   - matching_details keys: {table_result[1].keys() if table_result[1] else 'None'}")
                    
                    # Convert to markdown with enhanced metadata
                    markdown_table = convert_table_to_markdown(table_result)
                    
                    # Add enhanced table metadata
                    table_metadata = {
                        "page": i + 1,
                        "table_number": j + 1,
                        "confidence": confidence,
                        "bbox": table_bbox,
                        "type": table_info.get("type", "unknown"),
                        "dimensions": {
                            "width": x2 - x1,
                            "height": y2 - y1
                        },
                        "position": {
                            "x": x1,
                            "y": y1
                        }
                    }
                    
                    # Add table content with enhanced formatting
                    table_content = {
                        "content": f"""## Table {i+1}.{j+1}

**Type**: {table_info.get('type', 'Standard')}
**Confidence**: {confidence:.2f}
**Location**: Page {i+1}, Position ({x1:.1f}, {y1:.1f})

{markdown_table}

""",
                        "metadata": table_metadata
                    }
                    
                    tables_content.append(table_content)
                    
                    # Clean up temporary file
                    os.remove(table_img_path)
            
            # Clean up temporary file
            os.remove(temp_img_path)
        
        # Write content to output file with enhanced metadata
        with open(output_path, "w", encoding="utf-8") as f:
            if tables_content:
                f.write("# Extracted Tables\n\n")
                
                # Sort tables by page and table number
                sorted_tables = sorted(tables_content, 
                                    key=lambda x: (x["metadata"]["page"], 
                                                 x["metadata"]["table_number"]))
                
                # Write table contents
                for table in sorted_tables:
                    f.write(table["content"])
                
                # Add detailed metadata section
                f.write("\n## Table Metadata\n\n")
                f.write("| Page | Table | Type | Confidence | Position | Dimensions |\n")
                f.write("|------|-------|------|------------|----------|------------|\n")
                
                for table in sorted_tables:
                    meta = table["metadata"]
                    pos_str = f"({meta['position']['x']:.1f}, {meta['position']['y']:.1f})"
                    dim_str = f"{meta['dimensions']['width']:.1f}x{meta['dimensions']['height']:.1f}"
                    f.write(f"| {meta['page']} | {meta['table_number']} | {meta['type']} | {meta['confidence']:.2f} | {pos_str} | {dim_str} |\n")
            else:
                f.write("No tables found in the PDF document.")
        
        print(f"Results saved to: {output_path}")
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

def convert_table_to_markdown(table_result):
    """
    Chuyển đổi kết quả TableFormer thành bảng Markdown
    
    Args:
        table_result (tuple): Kết quả từ TableFormerPredictor (tf_output, matching_details)
        
    Returns:
        str: Bảng ở định dạng Markdown
    """
    try:
        # Lấy kết quả từ tuple
        tf_output, matching_details = table_result
        
        print("\nDebug - Chi tiết kết quả TableFormer:")
        print("1. Keys trong matching_details:", matching_details.keys())
        
        # Kiểm tra xem có table_cells không
        if "table_cells" in matching_details and matching_details["table_cells"]:
            cells = matching_details["table_cells"]
            print(f"2. Số lượng cells tìm thấy: {len(cells)}")
            print("3. Mẫu cell đầu tiên:", cells[0] if cells else "Không có cell")
            
            # Tạo ma trận để lưu nội dung bảng
            max_row = max(cell["row_id"] for cell in cells) + 1
            max_col = max(cell["column_id"] for cell in cells) + 1
            table_matrix = [["" for _ in range(max_col)] for _ in range(max_row)]
            
            print(f"4. Kích thước bảng: {max_row}x{max_col}")
            
            # Điền nội dung vào ma trận
            for cell in cells:
                row = cell["row_id"]
                col = cell["column_id"]
                
                # Lấy nội dung từ text_cell_bboxes nếu có
                content = ""
                if cell.get("text_cell_bboxes"):
                    print(f"5. Cell [{row},{col}] có text_cell_bboxes")
                    # Lấy nội dung từ docling_responses nếu có
                    if matching_details.get("docling_responses"):
                        print("6. Có docling_responses")
                        for response in matching_details["docling_responses"]:
                            if "text" in response:
                                content += response["text"] + " "
                                print(f"7. Text từ docling: {response['text']}")
                    # Nếu không có docling_responses, thử lấy từ tokens
                    elif cell.get("tokens"):
                        print("8. Có tokens")
                        content = " ".join(token.get("text", "") for token in cell["tokens"])
                        print(f"9. Text từ tokens: {content}")
                    # Nếu không có cả hai, thử lấy từ prediction
                    elif matching_details.get("prediction"):
                        print("10. Có prediction")
                        for pred in matching_details["prediction"]:
                            if pred.get("text"):
                                content += pred["text"] + " "
                                print(f"11. Text từ prediction: {pred['text']}")
                    content = content.strip()
                
                # Điền nội dung vào ô tương ứng
                if row < max_row and col < max_col:
                    table_matrix[row][col] = content
                    print(f"12. Đã điền nội dung vào ô [{row},{col}]: {content}")
            
            # Tạo bảng Markdown
            markdown_rows = []
            
            # Thêm header
            header_row = "| " + " | ".join(cell if cell else " " for cell in table_matrix[0]) + " |"
            markdown_rows.append(header_row)
            
            # Thêm dòng ngăn cách
            separator = "| " + " | ".join(["---"] * len(table_matrix[0])) + " |"
            markdown_rows.append(separator)
            
            # Thêm các dòng dữ liệu
            for row in table_matrix[1:]:
                data_row = "| " + " | ".join(cell if cell else " " for cell in row) + " |"
                markdown_rows.append(data_row)
            
            result = "\n".join(markdown_rows)
            print("13. Bảng Markdown được tạo:", result)
            return result
        else:
            print("Debug - Không tìm thấy table_cells trong matching_details")
            return "Không tìm thấy cấu trúc bảng."
        
    except Exception as e:
        print(f"Lỗi khi chuyển đổi bảng sang Markdown: {str(e)}")
        import traceback
        print(f"Debug - Traceback: {traceback.format_exc()}")
        return "Không thể chuyển đổi bảng sang định dạng Markdown."

def main():
    """
    Main function to extract both text and tables from PDF file.
    Uses fixed paths for input and output files.
    """
    # Define fixed paths
    INPUT_PATH = r"D:\DATN_HUST\test\data\raw\pdf\QCDT-2023-upload.pdf"
    TEXT_OUTPUT_PATH = r"D:\DATN_HUST\test\text_output.txt"
    TABLES_OUTPUT_PATH = r"D:\DATN_HUST\test\tables_output.txt"
    
    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file does not exist: {INPUT_PATH}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(TEXT_OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process text using standard docling method
        # print("\n=== Processing Text Content ===")
        # extract_text_from_pdf(
        #     INPUT_PATH, 
        #     TEXT_OUTPUT_PATH, 
        #     preserve_tables=False,  # We'll handle tables separately
        # )
        # print(f"Text extraction completed! Check the output at: {TEXT_OUTPUT_PATH}")
        
        # Process tables using IBM models
        print("\n=== Processing Tables ===")
        extract_tables_with_ibm_models(
            INPUT_PATH,
            TABLES_OUTPUT_PATH
        )
        print(f"Table extraction completed! Check the output at: {TABLES_OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main() 