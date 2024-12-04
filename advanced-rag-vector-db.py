import numpy as np
import faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedRAGSystem:
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 qa_model: str = 'deepset/roberta-base-squad2'):
        """
        Advanced RAG System with Vector Database
        
        Args:
            embedding_model (str): Sentence transformer for embeddings
            qa_model (str): Question answering model
        """
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Vector database using FAISS
        self.vector_db = {
            'embeddings': [],
            'documents': [],
            'metadata': []
        }
        
        # FAISS index for efficient similarity search
        self.index = None
        
        # Question Answering model
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model)
        
        # Advanced retrieval configurations
        self.retrieval_config = {
            'top_k': 5,
            'similarity_threshold': 0.7,
            'reranking_strategy': 'ensemble',
            'diversity_penalty': 0.2
        }

    def add_documents(self, 
                      documents: List[str], 
                      metadata: List[Dict[str, Any]] = None):
        """
        Add documents to the vector database
        
        Args:
            documents (List[str]): Text documents to index
            metadata (List[Dict]): Optional metadata for each document
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Add to vector database
        self.vector_db['documents'].extend(documents)
        self.vector_db['embeddings'].extend(embeddings)
        self.vector_db['metadata'].extend(metadata or 
            [{}] * len(documents))
        
        # Rebuild FAISS index
        self._build_index()

    def _build_index(self):
        """
        Build FAISS index for efficient retrieval
        """
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.vector_db['embeddings']).astype('float32')
        
        # Create index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

    def advanced_retrieval(self, 
                            query: str, 
                            retrieval_config: Dict[str, Any] = None) -> List[Dict]:
        """
        Advanced retrieval with multi-stage filtering and reranking
        
        Args:
            query (str): Search query
            retrieval_config (Dict): Custom retrieval configuration
        
        Returns:
            List of retrieved documents with metadata
        """
        # Merge configuration
        config = {**self.retrieval_config, **(retrieval_config or {})}
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Similarity search
        distances, indices = self.index.search(query_embedding, config['top_k'])
        
        # Filter and rerank results
        retrieved_docs = []
        for idx, distance in zip(indices[0], distances[0]):
            # Cosine similarity filtering
            similarity = 1 - (distance / np.linalg.norm(query_embedding))
            
            if similarity >= config['similarity_threshold']:
                doc = {
                    'text': self.vector_db['documents'][idx],
                    'metadata': self.vector_db['metadata'][idx],
                    'similarity': similarity
                }
                retrieved_docs.append(doc)
        
        # Diversify results
        retrieved_docs = self._diversify_results(retrieved_docs, config['diversity_penalty'])
        
        return retrieved_docs

    def _diversify_results(self, 
                            documents: List[Dict], 
                            diversity_penalty: float) -> List[Dict]:
        """
        Diversify retrieved documents
        
        Args:
            documents (List[Dict]): Retrieved documents
            diversity_penalty (float): Penalty for similar documents
        
        Returns:
            Diversified document list
        """
        if not documents:
            return []
        
        # Sort by similarity
        documents.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Diversity filtering
        diverse_docs = [documents[0]]
        for doc in documents[1:]:
            is_diverse = True
            for selected_doc in diverse_docs:
                # Compute text similarity
                text_similarity = cosine_similarity(
                    self.embedding_model.encode([doc['text']]),
                    self.embedding_model.encode([selected_doc['text']])
                )[0][0]
                
                # Apply diversity penalty
                if text_similarity > diversity_penalty:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_docs.append(doc)
        
        return diverse_docs

    def question_answering(self, 
                            query: str, 
                            context: str) -> Dict[str, Any]:
        """
        Perform question answering with retrieved context
        
        Args:
            query (str): Question to answer
            context (str): Context for answering
        
        Returns:
            Answer dictionary
        """
        # Tokenize inputs
        inputs = self.qa_tokenizer(
            query, 
            context, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True
        )
        
        # Generate answer
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
        
        # Extract answer
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Find best answer span
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        
        # Decode answer
        answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
        answer = self.qa_tokenizer.decode(answer_tokens)
        
        return {
            'answer': answer,
            'start_confidence': torch.max(start_scores).item(),
            'end_confidence': torch.max(end_scores).item()
        }

    def comprehensive_rag_pipeline(self, query: str) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline
        
        Args:
            query (str): User query
        
        Returns:
            Comprehensive RAG response
        """
        # Retrieve relevant documents
        retrieved_docs = self.advanced_retrieval(query)
        
        # Combine contexts
        combined_context = " ".join([doc['text'] for doc in retrieved_docs])
        
        # Perform question answering
        answer = self.question_answering(query, combined_context)
        
        return {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'answer': answer,
            'num_sources': len(retrieved_docs)
        }

# Example usage
def example_usage():
    # Initialize RAG system
    rag_system = AdvancedRAGSystem()
    
    # Add sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "RAG combines retrieval and generation techniques in NLP."
    ]
    
    metadata = [
        {'domain': 'AI', 'category': 'Definition'},
        {'domain': 'Machine Learning', 'category': 'Technique'},
        {'domain': 'NLP', 'category': 'Advanced Method'}
    ]
    
    rag_system.add_documents(documents, metadata)
    
    # Perform RAG
    query = "What is machine learning?"
    rag_result = rag_system.comprehensive_rag_pipeline(query)
    
    print("RAG Result:", rag_result)

if __name__ == '__main__':
    example_usage()
