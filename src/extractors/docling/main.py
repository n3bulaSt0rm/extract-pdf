import os
import re
from docling.document_converter import DocumentConverter

def extract_text_from_pdf(pdf_path, output_path, preserve_tables=True):
    """
    Extract text from PDF files (including scanned PDFs with tables) using Docling.
    Tables are preserved in Markdown format, regular text is plain text.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path to the output file
        preserve_tables (bool): Whether to preserve tables in Markdown format
    
    Returns:
        str: Path to the output file
    """
    # Initialize the document converter
    converter = DocumentConverter()
    
    # Convert the PDF
    print(f"Processing: {pdf_path}")
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

def main():
    """
    Main function to extract text from PDF file.
    Uses fixed paths for input and output files.
    """
    # Define fixed paths
    INPUT_PATH = r"D:\DATN_HUST\test\data\pdf\QCDT-2023-upload.pdf"
    OUTPUT_PATH = r"D:\DATN_HUST\test\output\output.txt"
    
    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file does not exist: {INPUT_PATH}")
        return
    
    # Process the PDF file - preserve tables by default
    try:
        extract_text_from_pdf(INPUT_PATH, OUTPUT_PATH, preserve_tables=True)
        print(f"Conversion completed! Check the output at: {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main() 