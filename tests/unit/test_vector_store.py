"""
Tests for Vector Store
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from storage.vector_store import VectorStore

class TestVectorStore:
    @pytest.fixture
    def mock_pinecone(self):
        with patch('storage.vector_store.Pinecone') as mock_pc:
            yield mock_pc

    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch('storage.vector_store.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            
            def encode_side_effect(text_or_list, **kwargs):
                if isinstance(text_or_list, list):
                    return [MagicMock(tolist=lambda: [0.1] * 384) for _ in text_or_list]
                return MagicMock(tolist=lambda: [0.1] * 384)
                
            mock_model.encode.side_effect = encode_side_effect
            mock_st.return_value = mock_model
            yield mock_model

    def test_initialization(self, mock_pinecone, mock_sentence_transformer):
        """Test initialization logic"""
        vs = VectorStore(api_key="test", environment="test-env")
        assert mock_pinecone.called
        assert mock_sentence_transformer.encode.called is False # Just init

    def test_upsert_documents_chunking(self, mock_pinecone, mock_sentence_transformer):
        """Test document chunking and upsert"""
        vs = VectorStore(api_key="test")
        vs.index = MagicMock()
        vs.index.describe_index_stats.return_value = {'total_vector_count': 0}
        
        # Long text that needs chunking
        long_text = "word " * 1000 
        docs = [{"text": long_text, "metadata": {"source": "test"}}]
        
        stats = vs.upsert_documents(docs, batch_size=10)
        
        assert vs.index.upsert.called
        assert stats["vectors"] > 0
        
    def test_retrieve(self, mock_pinecone, mock_sentence_transformer):
        """Test semantic search (retrieve)"""
        vs = VectorStore(api_key="test")
        vs.index = MagicMock()
        vs.index.query.return_value = {
            "matches": [
                {"id": "1", "score": 0.9, "metadata": {"chunk_text": "result"}}
            ]
        }
        
        results = vs.retrieve("query string", top_k=1)
        
        assert len(results) == 1
        assert results[0]["score"] == 0.9
        assert mock_sentence_transformer.encode.called
