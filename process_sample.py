#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import difflib
from pathlib import Path
from typing import Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_text

def extract_text_from_pdf(
    input_path: str,
    output_path: Optional[str] = None,
    strategy: str = "hi_res",
    include_page_breaks: bool = True,
) -> str:
    """
    Extract text from a PDF file with high accuracy using unstructured.io.
    
    Args:
        input_path (str): Path to the input PDF file
        output_path (Optional[str], optional): Path to save the extracted text. If None, returns the text.
        strategy (str, optional): The extraction strategy. Defaults to "hi_res".
            Options include:
            - "hi_res": Higher accuracy but slower process
            - "fast": Faster processing but potentially lower accuracy
        include_page_breaks (bool, optional): Whether to include page breaks in the output. Defaults to True.
        
    Returns:
        str: The extracted text if output_path is None
    """
    print(f"Processing PDF: {input_path}")
    start_time = time.time()
    
    # Import main extraction function to ensure we use the latest version
    from main import extract_text_from_pdf as main_extractor
    
    # Extract text using the main extraction function
    result = main_extractor(
        input_path=input_path,
        output_path=output_path,
        strategy=strategy,
        include_page_breaks=include_page_breaks,
        language="vie",  # Set Vietnamese as the language
        use_ocr=True,    # Enable OCR for better accuracy
    )
    
    return result

def calculate_metrics(reference_text: str, extracted_text: str, max_comparison_lines: int = 1000) -> dict:
    """
    Calculate text comparison metrics between reference and extracted text.
    
    Args:
        reference_text (str): The reference text (ground truth)
        extracted_text (str): The text extracted from PDF
        max_comparison_lines (int): Maximum number of lines to compare to prevent performance issues
        
    Returns:
        dict: Dictionary of metrics
    """
    # Prepare lines for comparison
    reference_lines = [line.strip() for line in reference_text.splitlines() if line.strip()]
    extracted_lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
    
    # Limit the number of lines for comparison to prevent performance issues
    if len(reference_lines) > max_comparison_lines:
        reference_lines = reference_lines[:max_comparison_lines]
        print(f"Warning: Reference text contains too many lines. Limited to {max_comparison_lines} lines.")
    
    if len(extracted_lines) > max_comparison_lines:
        extracted_lines = extracted_lines[:max_comparison_lines]
        print(f"Warning: Extracted text contains too many lines. Limited to {max_comparison_lines} lines.")
    
    # Calculate basic metrics
    metrics = {}
    
    # Character-level accuracy using a more efficient approach
    # Only compare the first 100,000 characters to avoid performance issues
    max_chars = 100000
    ref_text_sample = reference_text[:max_chars] if len(reference_text) > max_chars else reference_text
    ext_text_sample = extracted_text[:max_chars] if len(extracted_text) > max_chars else extracted_text
    
    seq_matcher = difflib.SequenceMatcher(None, ref_text_sample, ext_text_sample)
    metrics['character_accuracy'] = seq_matcher.ratio() * 100
    
    # Line-level metrics with performance optimization
    matching_lines = 0
    similar_lines = 0
    
    # Create a hash map of extracted lines for faster lookups
    ext_lines_set = set(extracted_lines)
    
    # For exact matches, use set intersection which is much faster
    for ref_line in reference_lines:
        if ref_line in ext_lines_set:
            matching_lines += 1
            continue
        
        # Only do expensive ratio calculations for lines that didn't have exact matches
        # and limit the number of comparisons
        max_comparisons = 100  # Limit comparison attempts to avoid quadratic complexity
        best_score = 0
        for i, ext_line in enumerate(extracted_lines[:max_comparisons]):
            # Quick length check first - if lengths are very different, similarity will be low
            if abs(len(ref_line) - len(ext_line)) > (len(ref_line) * 0.5):
                continue
                
            score = difflib.SequenceMatcher(None, ref_line, ext_line).ratio()
            if score > best_score:
                best_score = score
                if best_score > 0.8:  # Early exit if we find a good match
                    break
        
        if best_score > 0.8:
            similar_lines += 1
    
    total_ref_lines = len(reference_lines)
    metrics['exact_line_match_percentage'] = (matching_lines / total_ref_lines * 100) if total_ref_lines else 0
    metrics['similar_line_match_percentage'] = ((matching_lines + similar_lines) / total_ref_lines * 100) if total_ref_lines else 0
    
    # Word-level metrics - also limit to improve performance
    max_words = 10000
    ref_words = ' '.join(reference_lines[:500]).split()[:max_words]
    ext_words = ' '.join(extracted_lines[:500]).split()[:max_words]
    
    common_words = set(ref_words).intersection(set(ext_words))
    metrics['word_recall'] = (len(common_words) / len(set(ref_words)) * 100) if ref_words else 0
    metrics['word_precision'] = (len(common_words) / len(set(ext_words)) * 100) if ext_words else 0
    
    if metrics['word_recall'] + metrics['word_precision'] > 0:
        metrics['word_f1'] = (2 * metrics['word_recall'] * metrics['word_precision'] / 
                            (metrics['word_recall'] + metrics['word_precision']))
    else:
        metrics['word_f1'] = 0
    
    # Calculate structural similarity
    metrics['length_ratio'] = min(len(extracted_text) / len(reference_text), 
                                 len(reference_text) / len(extracted_text)) * 100 if reference_text and extracted_text else 0
    
    metrics['line_count_ratio'] = min(len(extracted_lines) / len(reference_lines), 
                                    len(reference_lines) / len(extracted_lines)) * 100 if reference_lines and extracted_lines else 0
    
    return metrics

def evaluate_extraction(sample_size: int = 5000):
    """
    Evaluate the PDF extraction by comparing with reference text.
    
    Args:
        sample_size (int): Maximum number of lines to evaluate to prevent performance issues
    """
    # Paths
    input_pdf = r"d:\DATN_HUST\test\sample.pdf"
    output_txt = r"d:\DATN_HUST\test\output\output.txt"
    reference_txt = r"d:\DATN_HUST\test\sample.txt"
    
    print(f"Starting evaluation...")
    start_time = time.time()
    
    # Extract text from PDF
    print(f"Processing PDF: {input_pdf}")
    extract_text_from_pdf(
        input_path=input_pdf,
        output_path=output_txt,
        strategy="hi_res",
        include_page_breaks=True,
    )
    
    # Read reference text
    if not os.path.exists(reference_txt):
        print(f"ERROR: Reference file not found: {reference_txt}")
        return
    
    with open(reference_txt, 'r', encoding='utf-8') as f:
        reference_text = f.read()
    
    # Read extracted text
    with open(output_txt, 'r', encoding='utf-8') as f:
        extracted_text = f.read()
    
    # For large files, limit the content to evaluate to prevent memory issues
    if len(reference_text) > sample_size*100 or len(extracted_text) > sample_size*100:
        reference_text = reference_text[:sample_size*100]
        extracted_text = extracted_text[:sample_size*100]
        print(f"Warning: Files too large, truncated to {sample_size*100} characters for evaluation")
    
    # Calculate metrics
    metrics = calculate_metrics(reference_text, extracted_text)
    
    # Report results
    print("\n========== EVALUATION RESULTS ==========")
    print(f"Character-level accuracy: {metrics['character_accuracy']:.2f}%")
    print(f"Exact line match: {metrics['exact_line_match_percentage']:.2f}%")
    print(f"Similar line match (>80% similarity): {metrics['similar_line_match_percentage']:.2f}%")
    print(f"Word recall: {metrics['word_recall']:.2f}%")
    print(f"Word precision: {metrics['word_precision']:.2f}%")
    print(f"Word F1 score: {metrics['word_f1']:.2f}%")
    print(f"Content length ratio: {metrics['length_ratio']:.2f}%")
    print(f"Line count ratio: {metrics['line_count_ratio']:.2f}%")
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    
    # Save metrics to file
    with open(r"d:\DATN_HUST\test\output\metrics.txt", 'w', encoding='utf-8') as f:
        f.write("========== PDF EXTRACTION EVALUATION METRICS ==========\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.2f}%\n")
    
    print(f"Metrics saved to: d:\\DATN_HUST\\test\\output\\metrics.txt")

if __name__ == "__main__":
    evaluate_extraction() 