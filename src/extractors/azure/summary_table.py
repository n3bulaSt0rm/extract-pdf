"""
Convert HTML tables to descriptive text using DeepSeek API

This script reads a text file containing HTML tables, extracts the tables,
converts each table to a descriptive text using DeepSeek API, and saves the result.
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# DeepSeek API configuration
DEEPSEEK_API_KEY = "sk-39e28a1cc59c49b28c3e1b16028949e0"
# Output directory configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def clean_document_content(content):
    """
    Clean the document content by:
    1. Removing lines with page markers (PageBreak, PageNumber, PageFooter, PageHeader)
       including empty lines around them
    2. Removing consecutive empty lines, keeping at most one
    
    Args:
        content (str): Original text content
        
    Returns:
        str: Cleaned text content without page markers
    """
    # Step 1: Remove lines with page markers and surrounding empty lines
    lines = content.split('\n')
    cleaned_lines = []
    i = 0
    
    while i < len(lines):
        # Check if current line contains a page marker
        is_page_marker = any(marker in lines[i] for marker in 
                            ['PageBreak', 'PageNumber', 'PageFooter', 'PageHeader'])
        
        if is_page_marker:
            # Skip one line before PageBreak if it's empty
            if 'PageBreak' in lines[i] and i > 0 and lines[i-1].strip() == '':
                if len(cleaned_lines) > 0:
                    cleaned_lines.pop()  # Remove the empty line before PageBreak
            
            # Skip the page marker line
            i += 1
            
            # Skip one line after PageNumber if it's empty
            if i < len(lines) and 'PageNumber' in lines[i-1] and lines[i].strip() == '':
                i += 1
                
            # Skip any consecutive empty lines between page markers
            while i < len(lines) and lines[i].strip() == '' and i+1 < len(lines) and any(
                marker in lines[i+1] for marker in ['PageBreak', 'PageNumber', 'PageFooter', 'PageHeader']):
                i += 1
        else:
            # Keep non-page-marker lines
            cleaned_lines.append(lines[i])
            i += 1
    
    # Step 2: Remove consecutive empty lines, keeping at most one
    condensed_lines = []
    prev_empty = False
    
    for line in cleaned_lines:
        is_empty = not line.strip()
        
        if is_empty and prev_empty:
            # Skip if this is a second consecutive empty line
            continue
        
        condensed_lines.append(line)
        prev_empty = is_empty
    
    # Rejoin content after cleaning
    content = '\n'.join(condensed_lines)
    
    return content


def extract_html_tables(text):
    """
    Extract HTML tables from text
    
    Args:
        text (str): Text that may contain HTML tables
        
    Returns:
        list: List of tuples (table_start_pos, table_end_pos, table_text)
    """
    table_patterns = []
    
    # Find all HTML tables in the text
    table_pattern = re.compile(r'<table>.*?</table>', re.DOTALL)
    for match in table_pattern.finditer(text):
        start_pos = match.start()
        end_pos = match.end()
        table_text = match.group(0)
        
        # Determine if this is a "--- Table N ---" marked table
        lines = text[:start_pos].split('\n')
        if lines and '--- Table ' in lines[-1]:
            # Add the marker to the table text
            table_marker = lines[-1]
            start_pos = start_pos - len(table_marker) - 1  # -1 for newline
            table_text = table_marker + '\n' + table_text
        
        table_patterns.append((start_pos, end_pos, table_text))
    
    return table_patterns


def convert_table_to_text(table_text, model):
    """
    Convert a table (markdown or HTML) to descriptive text using DeepSeek
    
    Args:
        table_text (str): Table in markdown or HTML format
        model: DeepSeek model to use for conversion
        
    Returns:
        str: Descriptive text of the table content
    """
    try:
        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["table"],
            template="""
            Biểu diễn bảng sau bằng văn bản bình thường, đảm bảo rõ ràng, dễ hiểu, không mất thông tin. 
            Không dùng kí tự đặc biệt ngoài dấu +, -. 
            Dấu >=, <=, >, < được biểu diễn bằng lời.
            Không giải thích gì thêm
            {table}
            """
        )
        
        # Use LangChain for converting table to text
        chain = prompt_template | model | StrOutputParser()
        description = chain.invoke({"table": table_text})
        
        # Return in a formatted way with the original table name if present
        if table_text.split('\n')[0].strip().startswith("--- Table "):
            table_name = table_text.split('\n')[0]
            description = f"{table_name}\n\n{description}"
        
        return description
    
    except Exception as e:
        print(f"Error converting table to text: {str(e)}")
        return table_text  # Return the original table on error


def process_file(input_file, output_file=None):
    """
    Process a file containing markdown tables and convert them to descriptive text
    
    Args:
        input_file (str): Path to the input file
        output_file (str, optional): Path to the output file. If None, creates a new file
                                   with "_converted" suffix.
    
    Returns:
        str: Path to the output file
    """
    try:
        # Validate inputs
        if not input_file:
            raise ValueError("Input file path cannot be empty")
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_file}")
        
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
        
        # Create output file path if not provided
        if not output_file:
            output_path = input_path.with_stem(f"{input_path.stem}_converted")
        else:
            output_path = Path(output_file)
        
        print(f"Processing file: {input_file}")
        print(f"Output will be saved to: {output_path}")
        
        # Initialize the DeepSeek model
        model = ChatDeepSeek(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            temperature=0.2,
            max_tokens=4096
        )
        
        # Read the input file
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Clean document content
        content = clean_document_content(content)
        
        # Extract tables from the cleaned content
        html_tables = extract_html_tables(content)
        all_tables = html_tables
        all_tables.sort(key=lambda x: x[0])
        
        print(f"Found {len(all_tables)} tables in the document")
        
        # Process in reverse order to avoid position shifting
        all_tables.sort(reverse=True, key=lambda x: x[0])
        
        for start_pos, end_pos, table_text in all_tables:
            print(f"Converting table with {table_text.count('|') + table_text.count('<td>')} cells...")
            table_description = convert_table_to_text(table_text, model)
            
            # Replace the table with its description
            content = content[:start_pos] + table_description + content[end_pos:]
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print("Conversion completed successfully")
        return str(output_path)
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


if __name__ == "__main__":
    # Check if we need to install langchain-deepseek
    try:
        import langchain_deepseek
    except ImportError:
        print("Installing langchain-deepseek...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-deepseek"])
        print("Installation complete.")
    
    # Get input file path
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "output_2.txt")
    
    # Process the file
    output_file = process_file(input_file)
    
    if output_file:
        print(f"Output saved to: {output_file}")
    else:
        print("Processing failed.")
