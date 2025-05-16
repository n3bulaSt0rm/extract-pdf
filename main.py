#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import re
from pathlib import Path
from typing import Optional, List
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_text
import ftfy
from pyvi import ViTokenizer

def clean_special_chars(text: str) -> str:
    """
    Clean special characters and artifacts often found in extracted PDF text.
    
    Args:
        text (str): Input text with potential artifacts
        
    Returns:
        str: Cleaned text
    """
    # Replace common corruption patterns
    text = re.sub(r'[+]{2,}', '', text)
    text = re.sub(r'[¿©®]{1,}', '', text)
    text = re.sub(r'[E]{3,}', 'E', text)
    text = re.sub(r'[r]{3,}', 'r', text)
    text = re.sub(r'[k]{3,}', 'k', text)
    text = re.sub(r'[-]{2,}', ' — ', text)
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[1-9]{6,}', '', text)
    text = re.sub(r'[csxze]{4,}', '', text)
    
    # Remove other non-printable characters 
    # (Python regex doesn't support \p{} Unicode property patterns without the 're' module)
    text = re.sub(r'[^\w\s.,;:!?()[\]{}\'\"<>@#$%^&*+=|\\~`-]', ' ', text)
    
    return text.strip()

def clean_vietnamese_text(text: str) -> str:
    """
    Clean and normalize Vietnamese text.
    
    Args:
        text (str): Input Vietnamese text
        
    Returns:
        str: Cleaned and normalized Vietnamese text
    """
    # Fix encoding issues first using ftfy
    text = ftfy.fix_text(text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors in Vietnamese
    text = text.replace('ă', 'ă').replace('â', 'â').replace('ê', 'ê')
    text = text.replace('ô', 'ô').replace('ơ', 'ơ').replace('ư', 'ư')
    text = text.replace('Đ', 'Đ').replace('đ', 'đ')
    
    # Fix tone marks that might be misplaced or incorrectly recognized
    vietnamese_chars = {
        # Lower case
        'à': 'à', 'á': 'á', 'ả': 'ả', 'ã': 'ã', 'ạ': 'ạ',
        'ằ': 'ằ', 'ắ': 'ắ', 'ẳ': 'ẳ', 'ẵ': 'ẵ', 'ặ': 'ặ',
        'ầ': 'ầ', 'ấ': 'ấ', 'ẩ': 'ẩ', 'ẫ': 'ẫ', 'ậ': 'ậ',
        'è': 'è', 'é': 'é', 'ẻ': 'ẻ', 'ẽ': 'ẽ', 'ẹ': 'ẹ',
        'ề': 'ề', 'ế': 'ế', 'ể': 'ể', 'ễ': 'ễ', 'ệ': 'ệ',
        'ì': 'ì', 'í': 'í', 'ỉ': 'ỉ', 'ĩ': 'ĩ', 'ị': 'ị',
        'ò': 'ò', 'ó': 'ó', 'ỏ': 'ỏ', 'õ': 'õ', 'ọ': 'ọ',
        'ồ': 'ồ', 'ố': 'ố', 'ổ': 'ổ', 'ỗ': 'ỗ', 'ộ': 'ộ',
        'ờ': 'ờ', 'ớ': 'ớ', 'ở': 'ở', 'ỡ': 'ỡ', 'ợ': 'ợ',
        'ù': 'ù', 'ú': 'ú', 'ủ': 'ủ', 'ũ': 'ũ', 'ụ': 'ụ',
        'ừ': 'ừ', 'ứ': 'ứ', 'ử': 'ử', 'ữ': 'ữ', 'ự': 'ự',
        'ỳ': 'ỳ', 'ý': 'ý', 'ỷ': 'ỷ', 'ỹ': 'ỹ', 'ỵ': 'ỵ',
        # Upper case
        'À': 'À', 'Á': 'Á', 'Ả': 'Ả', 'Ã': 'Ã', 'Ạ': 'Ạ',
        'Ằ': 'Ằ', 'Ắ': 'Ắ', 'Ẳ': 'Ẳ', 'Ẵ': 'Ẵ', 'Ặ': 'Ặ',
        'Ầ': 'Ầ', 'Ấ': 'Ấ', 'Ẩ': 'Ẩ', 'Ẫ': 'Ẫ', 'Ậ': 'Ậ',
        'È': 'È', 'É': 'É', 'Ẻ': 'Ẻ', 'Ẽ': 'Ẽ', 'Ẹ': 'Ẹ',
        'Ề': 'Ề', 'Ế': 'Ế', 'Ể': 'Ể', 'Ễ': 'Ễ', 'Ệ': 'Ệ',
        'Ì': 'Ì', 'Í': 'Í', 'Ỉ': 'Ỉ', 'Ĩ': 'Ĩ', 'Ị': 'Ị',
        'Ò': 'Ò', 'Ó': 'Ó', 'Ỏ': 'Ỏ', 'Õ': 'Õ', 'Ọ': 'Ọ',
        'Ồ': 'Ồ', 'Ố': 'Ố', 'Ổ': 'Ổ', 'Ỗ': 'Ỗ', 'Ộ': 'Ộ',
        'Ờ': 'Ờ', 'Ớ': 'Ớ', 'Ở': 'Ở', 'Ỡ': 'Ỡ', 'Ợ': 'Ợ',
        'Ù': 'Ù', 'Ú': 'Ú', 'Ủ': 'Ủ', 'Ũ': 'Ũ', 'Ụ': 'Ụ',
        'Ừ': 'Ừ', 'Ứ': 'Ứ', 'Ử': 'Ử', 'Ữ': 'Ữ', 'Ự': 'Ự',
        'Ỳ': 'Ỳ', 'Ý': 'Ý', 'Ỷ': 'Ỷ', 'Ỹ': 'Ỹ', 'Ỵ': 'Ỵ'
    }

    for incorrect, correct in vietnamese_chars.items():
        text = text.replace(incorrect, correct)
    
    # Fix common patterns in Vietnamese text from PDFs
    text = re.sub(r'([a-zA-Z])- ([a-zA-Z])', r'\1\2', text)  # Remove unnecessary hyphens
    
    # Fix spacing around punctuation for Vietnamese
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    text = re.sub(r'([.,;:!?)])\s+', r'\1 ', text)
    text = re.sub(r'([([{])\s+', r'\1', text)
    
    # Fix corrupted text patterns
    text = clean_special_chars(text)
    
    # # Use ViTokenizer for better word segmentation
    # text = ViTokenizer.tokenize(text)
    # text = text.replace('_', ' ')  # Replace underscores with spaces
    
    return text.strip()

def fix_common_ocr_errors(text: str) -> str:
    """
    Fix common OCR errors in Vietnamese PDFs
    
    Args:
        text (str): Text with potential OCR errors
        
    Returns:
        str: Corrected text
    """
    # Fix common Vietnamese OCR errors
    corrections = {
        'tiên sĩ': 'tiến sĩ',
        'tiến Sĩ': 'tiến sĩ',
        'Tiến Sĩ': 'Tiến sĩ',
        'đào tạo': 'đào tạo',
        'đảo tạo': 'đào tạo',
        'đao tạo': 'đào tạo', 
        'tiêu luận': 'tiểu luận',
        'chuyên đê': 'chuyên đề',
        'chuyên đề tiên': 'chuyên đề tiến',
        'luân án': 'luận án',
        'quy chê': 'quy chế',
        'chuyền tiếp': 'chuyển tiếp',
        'Điêu': 'Điều',
        'tỏ chức': 'tổ chức',
        'trung binh': 'trung bình',
    }
    
    for error, correction in corrections.items():
        text = re.sub(r'\b' + re.escape(error) + r'\b', correction, text, flags=re.IGNORECASE)
    
    return text

def fix_document_identifiers(text: str) -> str:
    """
    Fix document identifiers like numbers, legal references, etc. that might have been split
    during extraction.
    
    Args:
        text (str): Text with potential split identifiers
        
    Returns:
        str: Text with fixed identifiers
    """
    # Fix common document identifier patterns
    
    # Fix number/year/code patterns (e.g., "11 / 2012 / QH13" -> "11/2012/QH13")
    text = re.sub(r'(\d+)\s*/\s*(\d+)\s*/\s*([A-Za-z0-9]+)', r'\1/\2/\3', text)
    text = re.sub(r'(\d+)\s+/\s+(\d+)\s+/\s+([A-Za-z0-9]+)', r'\1/\2/\3', text)
    
    # Fix spaced number/year patterns without slashes (e.g., "11 2012 QH13" -> "11/2012/QH13")
    text = re.sub(r'(\d{1,4})\s+(\d{4})\s+([A-Z]{1,3}\d{1,3})', r'\1/\2/\3', text)
    
    # Fix legal reference patterns (e.g., "Điều 1 ." -> "Điều 1.")
    text = re.sub(r'(Điều\s+\d+)\s+\.', r'\1.', text)
    text = re.sub(r'(Khoản\s+\d+)\s+\.', r'\1.', text)
    text = re.sub(r'(Mục\s+\d+)\s+\.', r'\1.', text)
    
    # Fix date patterns (e.g., "ngày 15 tháng 7 năm 2020" -> "ngày 15/7/2020")
    # text = re.sub(r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', r'ngày \1/\2/\3', text)
    
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    text = re.sub(r'([([{])\s+', r'\1', text)
    
    # Fix common Vietnamese legal document patterns
    text = re.sub(r'Điều\s+(\d+)\s*\.\s*', r'Điều \1. ', text)
    
    return text

def post_process_vietnamese_text(text: str) -> str:
    """
    Apply post-processing specific for Vietnamese PDFs
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Processed Vietnamese text
    """
    # Apply ftfy to fix encoding issues first
    text = ftfy.fix_text(text)
    
    # Fix common layout issues
    lines = text.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        # Clean the current line
        line = clean_vietnamese_text(line)
        
        # If line is empty or just contains special characters, skip it
        if not line.strip() or re.match(r'^[^\w]+$', line):
            processed_lines.append('')
            continue
        
        # Fix common OCR errors
        line = fix_common_ocr_errors(line)
            
        # Check if this line might be a continuation of the previous line
        if (i > 0 and line and not line[0].isupper() and 
            not line[0].isdigit() and not line.startswith('-') and
            processed_lines and processed_lines[-1] and 
            not processed_lines[-1].endswith('.') and 
            not processed_lines[-1].endswith('?') and
            not processed_lines[-1].endswith('!')):
            # Append to the previous line instead of adding a new line
            processed_lines[-1] = processed_lines[-1] + ' ' + line
        else:
            processed_lines.append(line)
    
    result = '\n'.join(processed_lines)
    
    # Further cleanup for specific Vietnamese text issues
    result = re.sub(r'\n{3,}', '\n\n', result)  # Normalize multiple newlines
    
    # Fix specific corruption patterns from the sample
    result = re.sub(r'\.{2,}[-¿+]{1,}', '...', result)
    result = re.sub(r'[©+221]{3,}', '', result)
    result = re.sub(r'[-¿]{2,}[cz]{2,}', '', result)
    
    # Remove any remaining corruption patterns
    result = re.sub(r'[^\w\s.,;:!?()[\]{}\'\"<>@#$%^&*+=|\\~`-]', '', result)
    
    # Fix document identifiers
    result = fix_document_identifiers(result)
    
    return result

def extract_text_from_pdf(
    input_path: str,
    output_path: Optional[str] = None,
    strategy: str = "hi_res",
    include_page_breaks: bool = True,
    language: str = "vie",  # Set default language to Vietnamese
    use_ocr: bool = True,   # Enable OCR for better accuracy
) -> str:
    """
    Extract text from a PDF file with high accuracy using unstructured.io.
    Optimized for Vietnamese text.
    
    Args:
        input_path (str): Path to the input PDF file
        output_path (Optional[str], optional): Path to save the extracted text. If None, returns the text.
        strategy (str, optional): The extraction strategy. Defaults to "hi_res".
            Options include:
            - "hi_res": Higher accuracy but slower processing
            - "fast": Faster processing but potentially lower accuracy
        include_page_breaks (bool, optional): Whether to include page breaks in the output. Defaults to True.
        language (str, optional): Language code for extraction. Defaults to "vie" for Vietnamese.
        use_ocr (bool, optional): Whether to use OCR. Defaults to True.
        
    Returns:
        str: The extracted text if output_path is None
    """
    print(f"Processing PDF: {input_path}")
    start_time = time.time()
    
    # Extract elements from PDF with settings optimized for Vietnamese
    elements = partition_pdf(
        filename=input_path,
        strategy=strategy,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        include_metadata=True,
        languages=[language],  # Use Vietnamese language for better accuracy
        ocr_languages=language,  # Use Vietnamese for OCR
    )
    
    # Add page break markers if requested
    if include_page_breaks:
        for i, element in enumerate(elements):
            if i > 0 and hasattr(elements[i], 'metadata') and hasattr(elements[i-1], 'metadata'):
                if elements[i].metadata.page_number != elements[i-1].metadata.page_number:
                    elements[i].text = f"\n\n--- Trang {elements[i].metadata.page_number} ---\n\n" + elements[i].text
    
    # Convert elements to text
    extracted_text = elements_to_text(elements)
    
    # Apply post-processing for Vietnamese text
    processed_text = post_process_vietnamese_text(extracted_text)
    
    end_time = time.time()
    print(f"Extraction completed in {end_time - start_time:.2f} seconds")
    
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        print(f"Extracted text saved to: {output_path}")
        return None
    
    return processed_text

def main():
    # Input and output paths
    input_pdf = r"D:\DATN_HUST\test\data\pdf\QCDT-2023-upload.pdf"
    output_txt = r"D:\DATN_HUST\test\output\QCDT-2023-upload.txt"
    
    # Process PDF with settings optimized for Vietnamese
    extract_text_from_pdf(
        input_path=input_pdf,
        output_path=output_txt,
        strategy="hi_res",
        include_page_breaks=True,
        language="vie",  # Set Vietnamese as the language
        use_ocr=True,    # Enable OCR for better accuracy
    )

if __name__ == "__main__":
    main()
