"""
PDF text and table extraction with Azure Document Intelligence
and table summarization with Deepseek API
"""

import os
import json
import time
import uuid
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatDeepseek

# Load environment variables
load_dotenv()

# Azure Document Intelligence configuration
AZURE_DOCUMENT_ENDPOINT = os.getenv("AZURE_DOCUMENT_ENDPOINT")
AZURE_DOCUMENT_KEY = os.getenv("AZURE_DOCUMENT_KEY")
AZURE_MODEL_ID = "prebuilt-layout"  # Model for extracting text and tables

# Deepseek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_ENDPOINT = os.getenv("DEEPSEEK_API_ENDPOINT")

# Output directory configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)


class DocumentExtractor:
    """Class for extracting text and tables from PDF documents using Azure Document Intelligence"""
    
    def __init__(self):
        """Initialize the document extractor with Azure configuration"""
        self.endpoint = AZURE_DOCUMENT_ENDPOINT
        self.key = AZURE_DOCUMENT_KEY
        self.model_id = AZURE_MODEL_ID
        
        # Create headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.key
        }
    
    def extract_from_url(self, pdf_url):
        """
        Extract text and tables from a PDF document URL
        
        Args:
            pdf_url (str): URL to the PDF document
            
        Returns:
            dict: Extracted content with text and tables
        """
        # Validate inputs
        if not pdf_url:
            raise ValueError("PDF URL cannot be empty")
            
        if not self.endpoint or not self.key:
            raise ValueError("Azure Document Intelligence configuration missing")
        
        # Prepare the API request URL
        analyze_url = f"{self.endpoint}/documentintelligence/documentModels/{self.model_id}:analyze?_overload=analyzeDocument"
        api_version = "2024-11-30"
        analyze_url = f"{analyze_url}&api-version={api_version}"
        
        # Add parameters for Vietnamese content and table extraction
        analyze_url += "&locale=vi-VN&stringIndexType=textElements&features=keyValuePairs,languages&outputContentFormat=markdown"
        
        # Prepare request body
        request_body = {
            "urlSource": pdf_url
        }
        
        # Submit the document for analysis
        response = requests.post(
            analyze_url,
            headers=self.headers,
            json=request_body
        )
        
        if response.status_code != 202:
            error_msg = f"Azure Document Analysis failed with status code {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        # Get the operation location
        operation_location = response.headers["Operation-Location"]
        
        # Poll for result
        result = self._get_analysis_result(operation_location)
        
        # Process the result to extract text and tables
        return self._process_result(result)
    
    def _get_analysis_result(self, operation_location):
        """
        Poll the Azure Document Intelligence service for analysis results
        
        Args:
            operation_location (str): URL to check operation status
            
        Returns:
            dict: Analysis result
        """
        # Get the operation ID from the operation location URL
        headers = {
            "Ocp-Apim-Subscription-Key": self.key
        }
        
        retry_count = 0
        max_retries = 50
        retry_interval = 3  # seconds
        
        while retry_count < max_retries:
            response = requests.get(operation_location, headers=headers)
            response_json = response.json()
            
            if response.status_code != 200:
                raise Exception(f"Operation failed with status: {response.status_code}, details: {response.text}")
            
            status = response_json.get("status")
            
            if status == "succeeded":
                return response_json
            elif status == "failed":
                error = response_json.get("error", {})
                error_message = error.get("message", "Unknown error")
                raise Exception(f"Analysis operation failed: {error_message}")
            
            # Wait before trying again
            retry_count += 1
            time.sleep(retry_interval)
        
        raise Exception(f"Analysis operation did not complete after {max_retries} retries")
    
    def _process_result(self, result):
        """
        Process the analysis result to extract text and tables
        
        Args:
            result (dict): Analysis result from Azure Document Intelligence
            
        Returns:
            dict: Processed content with text and tables in markdown format
        """
        content = {
            "text": "",
            "tables": []
        }
        
        # Extract overall document text content
        if "content" in result:
            content["text"] = result["content"]
        
        # Extract tables
        if "tables" in result:
            for table in result["tables"]:
                # Convert table to markdown format
                markdown_table = self._table_to_markdown(table)
                content["tables"].append(markdown_table)
        
        return content
    
    def _table_to_markdown(self, table):
        """
        Convert a table object to markdown format
        
        Args:
            table (dict): Table object from the analysis result
            
        Returns:
            str: Markdown representation of the table
        """
        # Extract cell data from the table
        row_count = table.get("rowCount", 0)
        column_count = table.get("columnCount", 0)
        cells = table.get("cells", [])
        
        # Create a 2D array to represent the table
        table_data = [[None for _ in range(column_count)] for _ in range(row_count)]
        
        # Fill the table data
        for cell in cells:
            row_index = cell.get("rowIndex", 0)
            column_index = cell.get("columnIndex", 0)
            content = cell.get("content", "").strip()
            
            if 0 <= row_index < row_count and 0 <= column_index < column_count:
                table_data[row_index][column_index] = content
        
        # Convert to markdown
        markdown_rows = []
        
        # Header row
        if row_count > 0:
            header_row = " | ".join([str(cell or "") for cell in table_data[0]])
            markdown_rows.append(f"| {header_row} |")
            
            # Separator row
            separator_row = " | ".join(["---" for _ in range(column_count)])
            markdown_rows.append(f"| {separator_row} |")
            
            # Data rows
            for i in range(1, row_count):
                data_row = " | ".join([str(cell or "") for cell in table_data[i]])
                markdown_rows.append(f"| {data_row} |")
        
        return "\n".join(markdown_rows)


class TableSummarizer:
    """Class for summarizing tables using Deepseek API"""
    
    def __init__(self):
        """Initialize the table summarizer with Deepseek configuration"""
        self.api_key = DEEPSEEK_API_KEY
        self.api_endpoint = DEEPSEEK_API_ENDPOINT
        
        # Create the summarization prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["table"],
            template="""
            Below is a table in markdown format from a Vietnamese document.
            
            {table}
            
            Please summarize the key information from this table in Vietnamese.
            Make sure to capture all important data points, relationships, and trends.
            The summary should be comprehensive and not miss any critical information from the table.
            """
        )
        
        # Initialize the Deepseek model with LangChain
        self.model = None
        if self.api_key and self.api_endpoint:
            self.model = ChatDeepseek(
                model_name="deepseek-chat",
                deepseek_api_key=self.api_key,
                deepseek_api_base=self.api_endpoint,
                temperature=0.2,
                max_tokens=1024
            )
    
    def summarize_table(self, markdown_table):
        """
        Summarize a markdown table using Deepseek API
        
        Args:
            markdown_table (str): Table in markdown format
            
        Returns:
            str: Summary of the table content
        """
        try:
            if self.model:
                # Use LangChain for summarization
                chain = self.prompt_template | self.model | StrOutputParser()
                summary = chain.invoke({"table": markdown_table})
                return summary
            else:
                # Fallback to direct API call if model initialization failed
                # Prepare the request headers
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                # Prepare the prompt with the table content
                prompt = self.prompt_template.format(table=markdown_table)
                
                # Prepare the API request body
                request_body = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,  # Lower temperature for more factual responses
                    "max_tokens": 1024
                }
                
                # Make the API request
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=request_body
                )
                
                if response.status_code != 200:
                    raise Exception(f"Deepseek API request failed with status code {response.status_code}: {response.text}")
                
                response_data = response.json()
                summary = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                return summary
        
        except Exception as e:
            print(f"Error summarizing table: {str(e)}")
            return "Error generating summary"


def process_pdf(pdf_url):
    """
    Process a PDF document: extract text and tables, summarize tables, and save results
    
    Args:
        pdf_url (str): URL to the PDF document
        
    Returns:
        str: Path to the saved output file
    """
    try:
        # Create document extractor and table summarizer
        document_extractor = DocumentExtractor()
        table_summarizer = TableSummarizer()
        
        # Extract content from the PDF
        print(f"Extracting content from {pdf_url}...")
        content = document_extractor.extract_from_url(pdf_url)
        
        # Generate a unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pdf_extraction_{timestamp}.txt"
        output_path = DATA_DIR / filename
        
        # Write text content to the output file
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("=== EXTRACTED TEXT ===\n\n")
            file.write(content["text"])
            file.write("\n\n=== EXTRACTED TABLES ===\n\n")
            
            # Process each table: write the table and its summary
            for i, table in enumerate(content["tables"]):
                file.write(f"\n--- Table {i+1} ---\n\n")
                file.write(table)
                file.write("\n\n")
                
                # Summarize the table
                print(f"Summarizing table {i+1}...")
                summary = table_summarizer.summarize_table(table)
                
                file.write("--- Table Summary ---\n\n")
                file.write(summary)
                file.write("\n\n")
        
        print(f"Extraction and summarization completed. Output saved to {output_path}")
        return str(output_path)
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_url>")
        sys.exit(1)
    
    pdf_url = sys.argv[1]
    output_path = process_pdf(pdf_url)
    
    if output_path:
        print(f"Output saved to: {output_path}")
    else:
        print("Processing failed.")
