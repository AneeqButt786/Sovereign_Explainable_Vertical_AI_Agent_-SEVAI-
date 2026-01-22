"""
Vector Store - Pinecone Integration
Optimized for free tier (100k vectors, 1 index)
"""

from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import hashlib
import json
from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger("vector_store")


class VectorStore:
    """
    Manages vector storage and retrieval using Pinecone.
    Optimized for free tier constraints and medical domain use cases.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize Vector Store
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            embedding_model: Sentence transformer model name
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.pinecone_api_key
        self.environment = environment or settings.pinecone_environment
        self.index_name = index_name or settings.pinecone_index_name
        self.embedding_model_name = embedding_model or settings.embedding_model
        
        # Initialize embedding model first (needed for dimension)
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Pinecone client (new API)
        self.pc = Pinecone(api_key=self.api_key)
        
        # Get or create index
        self.index = self._get_or_create_index()
        
        # Track vector count (important for free tier limit)
        self.max_vectors = settings.max_vectors
        self._warn_threshold = int(self.max_vectors * 0.9)  # Warn at 90%
        
        logger.info(
            f"Vector Store initialized: {self.index_name}, "
            f"dimension: {self.embedding_dim}"
        )
    
    def _get_or_create_index(self):
        """Get existing index or create new one"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Free tier region
                )
            )
        else:
            logger.info(f"Using existing index: {self.index_name}")
        
        return self.pc.Index(self.index_name)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batched)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=len(texts) > 10
        )
        return [emb.tolist() for emb in embeddings]
    
    def chunk_document(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        medical_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk document into smaller pieces for embedding.
        Prioritizes medical concepts if provided.
        
        Args:
            text: Document text
            chunk_size: Target size in characters (approximate)
            chunk_overlap: Overlap between chunks
            medical_keywords: Medical terms to preserve in chunking
            
        Returns:
            List of chunk dicts with text and metadata
        """
        # Simple sentence-based chunking
        # TODO: Improve with medical concept-aware chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "char_count": len(chunk_text)
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "char_count": len(chunk_text)
            })
        
        logger.debug(f"Document chunked into {len(chunks)} chunks")
        return chunks
    
    def generate_doc_id(self, text: str, source: str = "") -> str:
        """Generate unique ID for document"""
        content = f"{source}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = "default",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Ingest documents into vector store
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
            namespace: Pinecone namespace for organization
            batch_size: Batch size for upserts
            
        Returns:
            Dict with ingestion stats
        """
        total_docs = len(documents)
        total_chunks = 0
        
        # Check vector count
        current_stats = self.index.describe_index_stats()
        current_vectors = current_stats.get('total_vector_count', 0)
        
        if current_vectors >= self._warn_threshold:
            logger.warning(
                f"Approaching free tier limit: {current_vectors}/{self.max_vectors} vectors"
            )
        
        all_vectors = []
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            
            # Chunk document
            chunks = self.chunk_document(text)
            total_chunks += len(chunks)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            # Prepare vectors for upsert
            for chunk, embedding in zip(chunks, embeddings):
                vector_id = self.generate_doc_id(chunk["text"], source)
                
                chunk_metadata = {
                    **metadata,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_text": chunk["text"],
                    "char_count": chunk["char_count"]
                }
                
                all_vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": chunk_metadata
                })
        
        # Batch upsert
        logger.info(f"Upserting {len(all_vectors)} vectors in batches of {batch_size}")
        
        for i in range(0, len(all_vectors), batch_size):
            batch = all_vectors[i:i+batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
        
        logger.info(
            f"Ingestion complete: {total_docs} documents, "
            f"{total_chunks} chunks, {len(all_vectors)} vectors"
        )
        
        return {
            "documents": total_docs,
            "chunks": total_chunks,
            "vectors": len(all_vectors),
            "namespace": namespace
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using RAG
        
        Args:
            query: Query text
            top_k: Number of results to return
            namespace: Pinecone namespace to query
            filter_dict: Metadata filters
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of results with similarity scores and metadata
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata
        )
        
        # Format results
        formatted_results = []
        for match in results.get("matches", []):
            result = {
                "id": match.get("id"),
                "score": match.get("score"),
            }
            
            if include_metadata and "metadata" in match:
                result["metadata"] = match["metadata"]
                result["text"] = match["metadata"].get("chunk_text", "")
            
            formatted_results.append(result)
        
        logger.debug(f"Retrieved {len(formatted_results)} results for query")
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = self.index.describe_index_stats()
        
        total_vectors = stats.get('total_vector_count', 0)
        utilization = (total_vectors / self.max_vectors) * 100
        
        return {
            "total_vectors": total_vectors,
            "max_vectors": self.max_vectors,
            "utilization_percent": utilization,
            "namespaces": stats.get('namespaces', {}),
            "dimension": self.embedding_dim,
            "index_name": self.index_name
        }
    
    def delete_all(self, namespace: str = "default"):
        """Delete all vectors in a namespace (use with caution)"""
        logger.warning(f"Deleting all vectors in namespace: {namespace}")
        self.index.delete(delete_all=True, namespace=namespace)


# Singleton instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create singleton Vector Store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
