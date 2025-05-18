#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import requests
import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple
import argparse

# Kiểm tra dependencies
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from pyvi import ViTokenizer
except ImportError as e:
    print(f"Error: {e}")
    print("\nTải dependences từ setup.py trước: python setup.py")
    sys.exit(1)

# Thiết lập logging
def setup_debug_logging(log_to_file: bool = True, log_file: str = None):
    """
    Thiết lập logging cho mục đích debug
    
    Args:
        log_to_file: Có ghi log ra file hay không
        log_file: Đường dẫn file log, nếu None sẽ tạo file theo ngày giờ
    """
    logger = logging.getLogger('rag_debug')
    logger.setLevel(logging.DEBUG)
    
    # Format log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler cho console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler cho file nếu cần
    if log_to_file:
        if log_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"rag_debug_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Debug logs được lưu vào file: {log_file}")
    
    return logger

# Tạo logger toàn cục
class QdrantRetriever:
    """
    Lớp lấy thông tin từ Qdrant dựa trên tìm kiếm ngữ nghĩa tiếng Việt
    """
    def __init__(
        self, 
        model_name: str = "vinai/phobert-base",
        host: str = "localhost", 
        port: int = 6333,
        collection_name: str = "hust_documents",
        debug_mode: bool = False
    ):
        """
        Khởi tạo retriever với mô hình và kết nối Qdrant
        
        Args:
            model_name: Tên mô hình embedding
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Tên collection trong Qdrant
            debug_mode: Chế độ debug
        """
        self.debug_mode = debug_mode
        print(f"Khởi tạo retriever với mô hình {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(host=host, port=port, timeout=10.0)
        self.collection_name = collection_name
        
        # Kiểm tra collection
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                raise ValueError(f"Collection '{self.collection_name}' không tồn tại trong Qdrant")
            
            print(f"Đã kết nối thành công đến Qdrant collection: {collection_name}")
            debug_logger.info(f"Kết nối thành công đến collection: {collection_name}")
        except Exception as e:
            print(f"Lỗi khi kiểm tra Qdrant collection: {e}")
            print("Đảm bảo rằng Qdrant server đang chạy và collection đã được tạo")
            debug_logger.error(f"Lỗi kết nối đến Qdrant: {e}")
            raise
    
    def _get_all_article_chunks(self, article_idx: int, exclude_id: int = None) -> List[Dict]:
        """
        Lấy tất cả các chunk thuộc cùng một điều khoản
        
        Args:
            article_idx: Chỉ số của điều khoản
            exclude_id: ID cần loại trừ (không lấy lại chunk đã có)
            
        Returns:
            Danh sách tất cả các chunk trong điều khoản, đã sắp xếp theo sentence_range
        """
        print(f"\n===== DEBUG: LẤY TẤT CẢ CHUNKS TRONG ĐIỀU KHOẢN (article_idx={article_idx}) =====")
        
        try:
            # Tạo filter cho article_idx
            filter_conditions = [
                models.FieldCondition(
                    key="article_idx",
                    match=models.MatchValue(value=article_idx)
                )
            ]
            
            # Tạo filter chính
            main_filter = models.Filter(
                must=filter_conditions
            )
            
            # Thử sử dụng Scroll API trước
            try:
                print("Thử sử dụng Scroll API...")
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=main_filter,
                    limit=100  # Lấy tối đa 100 chunks từ một điều khoản
                )
                
                points = result[0]
                print(f"✓ Tìm thấy {len(points)} chunks bằng Scroll API")
                
                # Lọc thủ công, loại bỏ exclude_id nếu có
                if exclude_id is not None:
                    filtered_points = [point for point in points if point.id != exclude_id]
                    print(f"Sau khi lọc (loại bỏ ID {exclude_id}): còn {len(filtered_points)} chunks")
                    points = filtered_points
                
                # Trích xuất thông tin từ points
                chunks = []
                for i, point in enumerate(points):
                    print(f"  Chunk #{i+1} - ID: {point.id}")
                    
                    # Lấy tất cả các trường quan trọng từ payload
                    # Kiểm tra cả hai trường preview và text
                    preview = None
                    # Thứ tự ưu tiên: text_preview > text > content
                    preview = point.payload.get("text_preview", None)
                    if not preview:
                        preview = point.payload.get("text", None)
                    if not preview:
                        preview = point.payload.get("content", "")
                        
                    # Thêm các thông tin metadata quan trọng 
                    chunk_idx = point.payload.get("chunk_idx", 0)
                    sentence_range = point.payload.get("sentence_range", "")
                    sentences = point.payload.get("sentences", 0)
                    
                    # Phân tích sentence_range để lấy chỉ số bắt đầu/kết thúc
                    try:
                        if sentence_range and "-" in sentence_range:
                            start_idx, end_idx = map(int, sentence_range.split('-'))
                        else:
                            start_idx, end_idx = 0, 0
                    except ValueError:
                        start_idx, end_idx = 0, 0
                    
                    chunks.append({
                        "id": point.id,
                        "preview": preview,
                        "chunk_idx": chunk_idx,
                        "sentence_range": sentence_range,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "sentences": sentences,
                        "full_payload": {k: v for k, v in point.payload.items()}  # Lưu toàn bộ payload
                    })
                
                # Sắp xếp chunks theo start_idx để đảm bảo thứ tự các câu
                chunks.sort(key=lambda x: x.get("start_idx", 0))
                print(f"Đã sắp xếp {len(chunks)} chunks theo thứ tự các câu")
                return chunks
                
            except Exception as scroll_error:
                print(f"✗ Lỗi Scroll API: {scroll_error}")
                print("Chuyển sang sử dụng Search API...")
                
                # Dùng search API nếu scroll không hoạt động
                dummy_vector = np.zeros(768)  # Kích thước mặc định cho PhoBERT
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=dummy_vector.tolist(),
                    query_filter=main_filter,
                    limit=100,  # Lấy tối đa 100 chunks
                    score_threshold=None  # Không dùng ngưỡng điểm vì dùng vector giả
                )
                
                print(f"✓ Tìm thấy {len(search_result)} chunks bằng Search API")
                
                # Lọc thủ công, loại bỏ exclude_id nếu có
                if exclude_id is not None:
                    filtered_hits = [hit for hit in search_result if hit.id != exclude_id]
                    print(f"Sau khi lọc (loại bỏ ID {exclude_id}): còn {len(filtered_hits)} chunks")
                    search_result = filtered_hits
                
                # Trích xuất thông tin
                chunks = []
                for i, hit in enumerate(search_result):
                    print(f"  Hit #{i+1} - ID: {hit.id}")
                    
                    # Kiểm tra cả hai trường preview và text 
                    preview = None
                    # Thứ tự ưu tiên: text_preview > text > content
                    preview = hit.payload.get("text_preview", None)
                    if not preview:
                        preview = hit.payload.get("text", None)
                    if not preview:
                        preview = hit.payload.get("content", "")
                        
                    chunk_idx = hit.payload.get("chunk_idx", 0)
                    sentence_range = hit.payload.get("sentence_range", "")
                    sentences = hit.payload.get("sentences", 0)
                    
                    # Phân tích sentence_range để lấy chỉ số bắt đầu/kết thúc
                    try:
                        if sentence_range and "-" in sentence_range:
                            start_idx, end_idx = map(int, sentence_range.split('-'))
                        else:
                            start_idx, end_idx = 0, 0
                    except ValueError:
                        start_idx, end_idx = 0, 0
                    
                    chunks.append({
                        "id": hit.id,
                        "preview": preview,
                        "chunk_idx": chunk_idx,
                        "sentence_range": sentence_range,
                        "start_idx": start_idx,
                        "end_idx": end_idx, 
                        "sentences": sentences,
                        "full_payload": {k: v for k, v in hit.payload.items()}  # Lưu toàn bộ payload
                    })
                
                # Sắp xếp chunks theo start_idx để đảm bảo thứ tự các câu
                chunks.sort(key=lambda x: x.get("start_idx", 0))
                print(f"Đã sắp xếp {len(chunks)} chunks theo thứ tự các câu")
                return chunks
                
        except Exception as e:
            print(f"❌ Lỗi khi lấy tất cả chunks trong điều khoản: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _expand_search_results(self, search_results: List[Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Mở rộng kết quả bằng cách lấy tất cả các chunk thuộc cùng điều khoản.
        
        Args:
            search_results: Kết quả tìm kiếm ban đầu từ Qdrant
            limit: Giới hạn số lượng điều khoản trả về
            
        Returns:
            Danh sách các điều khoản hoàn chỉnh đã mở rộng
        """
        print("\n===== DEBUG: MỞ RỘNG KẾT QUẢ TÌM KIẾM =====")
        
        # Theo dõi các điều khoản đã xử lý
        processed_articles = set()
        expanded_results = []
        
        # Sắp xếp kết quả tìm kiếm theo điểm số
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        
        for hit in sorted_results:
            article_idx = hit.payload.get("article_idx", None)
            
            # Bỏ qua nếu không thể xác định article_idx hoặc đã xử lý điều khoản này
            if article_idx is None or article_idx in processed_articles:
                continue
            
            # Đánh dấu điều khoản đã xử lý để tránh trùng lặp
            processed_articles.add(article_idx)
            
            # Lấy tất cả các chunk thuộc điều khoản
            article_chunks = self._get_all_article_chunks(article_idx)
            
            if not article_chunks:
                # Nếu không tìm thấy chunks, sử dụng nội dung từ hit hiện tại
                text_preview = hit.payload.get("text_preview", "")
                if not text_preview:
                    text_preview = hit.payload.get("text", "")
                if not text_preview:
                    text_preview = hit.payload.get("content", "")
                    
                article_title = hit.payload.get("article", "")
                
                expanded_results.append({
                    "score": hit.score,
                    "article": article_title,
                    "content": text_preview,
                    "article_idx": article_idx,
                    "chunks_count": 1,
                    "content_length": len(text_preview),
                    "is_expanded": False
                })
            else:
                # Kết hợp nội dung từ tất cả chunks
                article_title = article_chunks[0].get("full_payload", {}).get("article", "")
                if not article_title:  # Fallback nếu không tìm thấy trong full_payload
                    article_title = hit.payload.get("article", "")
                
                # Kết hợp chunks theo thứ tự đã sắp xếp
                context_parts = []
                for chunk in article_chunks:
                    if chunk.get("preview"):
                        context_parts.append(chunk["preview"])
                
                # Thêm tiêu đề cho article để context rõ ràng hơn
                combined_content = f"ĐIỀU KHOẢN: {article_title}\n\n" + "\n".join(context_parts)
                
                expanded_results.append({
                    "score": hit.score,
                    "article": article_title,
                    "content": combined_content,
                    "article_idx": article_idx,
                    "chunks_count": len(article_chunks),
                    "content_length": len(combined_content),
                    "sentence_ranges": [chunk.get("sentence_range") for chunk in article_chunks],
                    "is_expanded": True
                })
            
            # Kiểm tra giới hạn số lượng điều khoản
            if len(expanded_results) >= limit:
                break
        
        print(f"Mở rộng kết quả: {len(search_results)} chunk -> {len(expanded_results)} điều khoản đầy đủ")
        return expanded_results
    
    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Truy xuất thông tin liên quan đến câu hỏi từ Qdrant, lấy toàn bộ các chunks
        thuộc cùng một điều khoản.
        
        Args:
            query: Câu hỏi đầu vào
            limit: Số lượng kết quả trả về tối đa
            
        Returns:
            Danh sách các đoạn văn bản liên quan và metadata
        """
        print("\n===== DEBUG: BẮT ĐẦU TRUY VẤN QDRANT =====")
        print(f"Câu truy vấn: '{query}'")
        print(f"Giới hạn kết quả: {limit}")
        
        # Tokenize và encode câu truy vấn
        tokenized_query = ViTokenizer.tokenize(query)
        print(f"Câu truy vấn sau khi tokenize: '{tokenized_query}'")
        query_vector = self.model.encode(tokenized_query)
        print(f"Kích thước vector truy vấn: {len(query_vector)}")
        
        # Tìm kiếm trong Qdrant
        print("\nGọi API tìm kiếm Qdrant...")
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit * 3,  # Lấy nhiều hơn giới hạn để có đủ điều khoản sau khi loại bỏ trùng lặp
            score_threshold=0.5  # Chỉ lấy kết quả có độ tương đồng > 0.5
        )
        
        print(f"\n===== DEBUG: KẾT QUẢ TÌM KIẾM: {len(search_results)} kết quả =====")
        
        # Lưu raw results để debug
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_results_file = f"qdrant_raw_results_{timestamp}.json"
            
            # Chuyển đổi kết quả thành JSON serializable
            raw_results_data = []
            for hit in search_results:
                hit_data = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                raw_results_data.append(hit_data)
                
            with open(raw_results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": query,
                    "tokenized_query": tokenized_query,
                    "vector_size": len(query_vector),
                    "results": raw_results_data
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Đã lưu raw results vào file: {raw_results_file}")
        except Exception as e:
            print(f"Lỗi khi lưu raw results: {e}")
        
        # Mở rộng kết quả tìm kiếm bằng cách lấy tất cả chunks từ cùng điều khoản
        expanded_results = self._expand_search_results(search_results, limit)
        
        # Lưu processed results để debug
        try:
            processed_results_file = f"qdrant_processed_results_{timestamp}.json"
            with open(processed_results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": query,
                    "processed_results": [{
                        "score": r["score"],
                        "article": r["article"],
                        "content_length": r["content_length"],
                        "content_preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                        "chunks_count": r["chunks_count"]
                    } for r in expanded_results]
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Đã lưu processed results vào file: {processed_results_file}")
        except Exception as e:
            print(f"Lỗi khi lưu processed results: {e}")
        
        print("\n===== DEBUG: KẾT THÚC TRUY VẤN QDRANT =====")
        return expanded_results

class DeepSeekLLM:
    """
    Lớp giao tiếp với DeepSeek API để sinh câu trả lời
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo kết nối DeepSeek API
        
        Args:
            api_key: API key của DeepSeek (lấy từ biến môi trường nếu không cung cấp)
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or "sk-b55e16010a6a43bf86b751617c52c22a"
        if not self.api_key:
            print("Cảnh báo: DEEPSEEK_API_KEY không được cung cấp")
            print("Bạn cần cung cấp API key để sử dụng DeepSeek API")
            print("Thiết lập biến môi trường: export DEEPSEEK_API_KEY=your_api_key")
        else:
            print("Đã cấu hình DeepSeek API với key")
        
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"  # Hoặc mô hình khác từ DeepSeek
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Sinh câu trả lời dựa trên câu hỏi và context
        
        Args:
            query: Câu hỏi
            contexts: Danh sách các context và metadata
            
        Returns:
            Câu trả lời từ DeepSeek
        """
        if not self.api_key:
            return "Không thể kết nối DeepSeek API. Vui lòng cung cấp API key."
        
        # Tạo prompt với thông tin về tổng context
        total_content_length = sum(len(ctx.get('content', '')) for ctx in contexts)
        total_chunks = sum(ctx.get('chunks_count', 1) for ctx in contexts)
        
        # Kiểm tra tổng chiều dài context có phù hợp cho API không
        print(f"\n===== DEBUG: CHUẨN BỊ GỬI CONTEXT ĐẾN DEEPSEEK =====")
        print(f"Tổng cộng: {len(contexts)} điều khoản, {total_chunks} chunks, {total_content_length} ký tự")
        
        # Tạo context text, kết hợp tất cả các đoạn văn liên quan
        context_sections = []
        for i, ctx in enumerate(contexts):
            article = ctx.get('article', 'Unknown Article')
            content = ctx.get('content', '')
            chunks_count = ctx.get('chunks_count', 1)
            
            # Log thông tin
            content_preview = content[:50] + "..." if len(content) > 50 else content
            print(f"Context #{i+1}: '{article}' - {len(content)} ký tự, {chunks_count} chunks")
            print(f"  Preview: {content_preview}")
            
            # Thêm vào danh sách context
            section = f"CONTEXT {i+1} [FROM: {article}]:\n{content}"
            context_sections.append(section)
        
        # Ghép tất cả các phần context lại
        context_text = "\n\n".join(context_sections)
        
        # Tạo prompt với hướng dẫn cụ thể về cách trả lời
        prompt = f"""Bạn là trợ lý AI giúp trả lời câu hỏi về quy chế đào tạo của trường ĐH Bách Khoa Hà Nội.
Hãy trả lời câu hỏi sau dựa trên thông tin từ các đoạn văn bản được cung cấp:

QUESTION: {query}

{context_text}

INSTRUCTIONS:
1. Trả lời dựa HOÀN TOÀN trên thông tin có trong các đoạn CONTEXT
2. Nếu thông tin không có trong đoạn văn bản, hãy nói rõ "Thông tin không có trong dữ liệu"
3. KHÔNG được tự tạo thông tin không có trong CONTEXT, ngay cả khi bạn biết thông tin đó
4. Trả lời một cách súc tích, đầy đủ, dễ hiểu
5. Cấu trúc câu trả lời thành các đoạn rõ ràng nếu cần thiết
6. Có thể trích dẫn nội dung từ CONTEXT khi cần thiết

ANSWER:"""
        
        print(f"Tổng độ dài prompt: {len(prompt)} ký tự")
        print("Gửi yêu cầu đến DeepSeek API...")
        
        # Gọi API DeepSeek
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Bạn là trợ lý trả lời câu hỏi về quy chế đào tạo. Trả lời hoàn toàn dựa trên thông tin được cung cấp bằng tiếng Việt.Hãy suy luận để trả lời thật tốt, thật tự nhiên"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,  # Thấp hơn để tăng tính chính xác
                "max_tokens": 5000
            }
            
            print("Đang gửi yêu cầu...")
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print(f"Đã nhận được câu trả lời từ DeepSeek ({len(answer)} ký tự)")
            return answer
            
        except Exception as e:
            print(f"❌ Lỗi khi gọi DeepSeek API: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback nếu API lỗi - tạo câu trả lời từ context
            answer = f"Không thể kết nối DeepSeek API. Dưới đây là thông tin liên quan:\n\n"
            for i, ctx in enumerate(contexts):
                article = ctx.get('article', 'Unknown Article')
                content = ctx.get('content', '')
                answer += f"- {article}:\n{content[:300]}...\n\n"
            return answer

class RAG:
    """
    Hệ thống RAG (Retrieval-Augmented Generation) kết hợp tìm kiếm và sinh trả lời
    """
    def __init__(
        self,
        retriever: Optional[QdrantRetriever] = None,
        generator: Optional[DeepSeekLLM] = None,
        num_results: int = 3,
        debug_mode: bool = False
    ):
        """
        Khởi tạo hệ thống RAG
        
        Args:
            retriever: Thành phần tìm kiếm
            generator: Thành phần sinh câu trả lời
            num_results: Số lượng kết quả lấy từ DB
            debug_mode: Chế độ debug
        """
        self.retriever = retriever or QdrantRetriever(debug_mode=debug_mode)
        self.generator = generator or DeepSeekLLM()
        self.num_results = num_results
        self.debug_mode = debug_mode
        
        debug_logger.info(f"Khởi tạo RAG với num_results={num_results}, debug_mode={debug_mode}")
    
    def answer_question(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Trả lời câu hỏi bằng RAG
        
        Args:
            query: Câu hỏi đầu vào
            
        Returns:
            Tuple gồm câu trả lời và danh sách context đã sử dụng
        """
        print(f"\nĐang xử lý câu hỏi: {query}")
        
        # Bước 1: Lấy thông tin từ Qdrant
        print("Đang tìm kiếm thông tin liên quan...")
        contexts = self.retriever.retrieve(query, limit=self.num_results)
        
        if not contexts:
            return "Không tìm thấy thông tin liên quan đến câu hỏi của bạn.", []
        
        num_contexts = len(contexts)
        total_chars = sum(len(ctx.get('content', '')) for ctx in contexts)
        print(f"Đã tìm thấy {num_contexts} đoạn văn bản liên quan (tổng cộng {total_chars} ký tự)")
        
        # Kiểm tra kích thước context - nếu quá lớn có thể vượt quá token limit của API
        MAX_CONTEXT_CHARS = 20000  # Khoảng 5000 tokens cho tiếng Việt
        
        if total_chars > MAX_CONTEXT_CHARS:
            print(f"⚠️ Cảnh báo: Tổng kích thước context ({total_chars} ký tự) vượt quá ngưỡng an toàn ({MAX_CONTEXT_CHARS})")
            print("Đang cắt giảm số lượng context...")
            
            # Sắp xếp contexts theo điểm số giảm dần để giữ lại các kết quả tốt nhất
            contexts = sorted(contexts, key=lambda x: x.get('score', 0), reverse=True)
            
            # Cắt giảm contexts cho đến khi vừa với giới hạn
            pruned_contexts = []
            current_total = 0
            
            for ctx in contexts:
                ctx_size = len(ctx.get('content', ''))
                if current_total + ctx_size <= MAX_CONTEXT_CHARS:
                    pruned_contexts.append(ctx)
                    current_total += ctx_size
                else:
                    print(f"Bỏ qua context '{ctx.get('article', '')}' ({ctx_size} ký tự) do vượt quá giới hạn")
            
            contexts = pruned_contexts
            print(f"Sau khi cắt giảm: {len(contexts)} contexts, {current_total} ký tự")
        
        # Bước 2: Sinh câu trả lời bằng DeepSeek
        print("Đang sinh câu trả lời...")
        try:
            answer = self.generator.generate_answer(query, contexts)
            return answer, contexts
        except Exception as e:
            error_msg = f"Lỗi khi sinh câu trả lời: {str(e)}"
            print(f"❌ {error_msg}")
            debug_logger.exception("Lỗi trong quá trình sinh câu trả lời")
            
            # Tạo câu trả lời đơn giản dựa trên contexts
            fallback_answer = f"Xin lỗi, có lỗi khi tạo câu trả lời. Dưới đây là các thông tin liên quan:\n\n"
            for i, ctx in enumerate(contexts):
                article = ctx.get('article', '')
                preview = ctx.get('content', '')[:200] + "..." if len(ctx.get('content', '')) > 200 else ctx.get('content', '')
                fallback_answer += f"{i+1}. {article}: {preview}\n\n"
            
            return fallback_answer, contexts

def display_results(answer: str, contexts: List[Dict], show_contexts: bool = False):
    """
    Hiển thị kết quả RAG
    
    Args:
        answer: Câu trả lời
        contexts: Context được sử dụng
        show_contexts: Có hiển thị context không
    """
    print("\n" + "=" * 80)
    print("CÂU TRẢ LỜI:")
    print("-" * 80)
    print(answer)
    print("=" * 80)
    
    if show_contexts and contexts:
        print("\nCÁC NGUỒN THAM KHẢO:")
        total_chars = sum(len(ctx.get('content', '')) for ctx in contexts)
        total_chunks = sum(ctx.get('chunks_count', 1) for ctx in contexts)
        print(f"Tổng cộng: {len(contexts)} điều khoản, {total_chunks} chunks, {total_chars} ký tự")
        
        for i, ctx in enumerate(contexts):
            chunks_count = ctx.get('chunks_count', 1)
            content_length = len(ctx.get('content', ''))
            
            print(f"\n[{i+1}] {ctx.get('article', 'Unknown')} (độ liên quan: {ctx.get('score', 0):.4f})")
            print(f"    - Số chunks: {chunks_count}")
            print(f"    - Độ dài: {content_length} ký tự")
            
            content_preview = ctx.get('content', '')[:200] + "..." if len(ctx.get('content', '')) > 200 else ctx.get('content', '')
            print(f"    - Đoạn trích: {content_preview}")
            
        print("\nLưu ý: Câu trả lời được DeepSeek tạo ra dựa trên các nguồn tham khảo trên.")

def main():
    # Xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser(description="RAG system sử dụng Qdrant và DeepSeek")
    parser.add_argument("--api-key", help="DeepSeek API key")
    parser.add_argument("--show-contexts", action="store_true", help="Hiển thị context đã sử dụng")
    parser.add_argument("--results", type=int, default=3, help="Số lượng kết quả tối đa")
    parser.add_argument("--debug", action="store_true", help="In chi tiết thông tin debug")
    parser.add_argument("--save-results", action="store_true", help="Lưu kết quả tìm kiếm vào file JSON")
    parser.add_argument("--log-file", help="Đường dẫn file log")
    args = parser.parse_args()
    
    # Thiết lập logger với file tùy chỉnh nếu có
    
    # Khởi tạo DeepSeek với API key cố định hoặc từ tham số dòng lệnh
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or "sk-b55e16010a6a43bf86b751617c52c22a"
    
    # Hiển thị thông tin
    print("=== Hệ thống Hỏi Đáp Quy Chế Đào Tạo ĐHBK Hà Nội ===")
    print("Sử dụng RAG (Retrieval-Augmented Generation)")
    print("Tính năng mới: Lấy toàn bộ nội dung Điều khoản để tăng chất lượng trả lời")
    debug_logger.info("Khởi động hệ thống RAG với tính năng lấy toàn bộ điều khoản")
    
    # Một số câu hỏi mẫu
    sample_questions = [
        "Chương trình đào tạo tiến sĩ kéo dài bao lâu?",
        "Thế nào là một tín chỉ học tập?",
        "Điều kiện để được xét tốt nghiệp?",
        "Sinh viên có thể đăng ký tối đa bao nhiêu tín chỉ một học kỳ?",
        "Quy định về học lại và học cải thiện điểm?",
    ]
    
    # Hiển thị câu hỏi mẫu
    print("\nCác câu hỏi mẫu:")
    for i, q in enumerate(sample_questions):
        print(f"{i+1}. {q}")
    
    try:
        # Khởi tạo RAG
        rag_system = RAG(
            generator=DeepSeekLLM(api_key),
            num_results=args.results,
            debug_mode=args.debug
        )
        
        while True:
            # Nhận câu hỏi từ người dùng
            print("\nNhập số để chọn câu hỏi mẫu hoặc nhập câu hỏi của bạn (gõ 'q' để thoát):")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
                
            # Xử lý câu hỏi mẫu
            if user_input.isdigit() and 1 <= int(user_input) <= len(sample_questions):
                query = sample_questions[int(user_input) - 1]
            else:
                query = user_input
            
            # Trả lời câu hỏi
            answer, contexts = rag_system.answer_question(query)
            
            # Lưu kết quả nếu được yêu cầu
            if args.save_results:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"rag_results_{timestamp}.json"
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "query": query,
                        "answer": answer,
                        "contexts": contexts
                    }, f, ensure_ascii=False, indent=2)
                
                print(f"Đã lưu kết quả vào file: {results_file}")
            
            # Hiển thị kết quả
            display_results(answer, contexts, args.show_contexts)
            
    except KeyboardInterrupt:
        print("\nĐã thoát chương trình")
    except Exception as e:
        print(f"\nLỗi không mong muốn: {e}")
        debug_logger.exception("Lỗi không mong muốn")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 