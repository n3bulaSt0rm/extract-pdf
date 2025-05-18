# Enhanced PDF Extractor with Table Support and OCR

This tool provides enhanced PDF text extraction with support for:
- Improved table extraction
- Support for scanned PDFs with OCR
- Direct image-to-text extraction

## Requirements

- Python 3.10+
- Tesseract OCR installed on your system
- GPU support (optional, for faster processing)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

3. For GPU acceleration (recommended for speed):
   - Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
   - Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
   - Run our setup script to install the correct PyTorch version:
     ```bash
     # On Windows
     setup_cuda.bat
     
     # On Linux/macOS
     python setup_cuda.py
     ```

## Usage

### Extract Text from PDFs (including scanned documents)

```python
from src.extractors.docling.main import extract_text_from_pdf

# Extract text from a PDF with tables and OCR support
extract_text_from_pdf(
    pdf_path="path/to/your/document.pdf",
    output_path="path/to/output.txt",
    preserve_tables=True,  # Set to False if you don't need tables
    handle_scanned=True    # Set to False for digital PDFs only
)
```

### Extract Text from Images

```python
from src.extractors.docling.main import extract_text_from_image

# Extract text from an image
extract_text_from_image(
    image_path="path/to/your/image.png",
    output_path="path/to/output.txt"
)
```

## Features

### Table Extraction
- The extractor preserves table structures in Markdown format
- Improved cell matching for better table recognition
- Works for both digital and scanned PDFs

### OCR Support
- Integrated Tesseract OCR for scanned documents
- Full-page OCR option for improved recognition
- GPU acceleration for faster processing when available

### CUDA Support
- GPU acceleration for significantly faster processing
- Automatic detection of CUDA availability
- Enhanced performance for large documents and complex tables

## Command-line Usage

You can run the extractor directly:

```bash
python -m src.extractors.docling.main
```

This will process the default files specified in the script. Edit the `main()` function to customize paths.

## Troubleshooting CUDA Issues

If you encounter issues with CUDA:

1. Run the CUDA verification script:
   ```bash
   python setup_cuda.py
   ```

2. Make sure your NVIDIA drivers are up-to-date

3. Ensure you have the correct PyTorch version for your CUDA version:
   ```bash
   # Test CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ``` 