import os
import logging
import warnings

# Root logging level adjustment to mask telemetry components early
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
warnings.simplefilter("ignore")

# Framework Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings 

class PolicyVectorStoreManager:
    """Ingests text-based regulatory frameworks, computes embeddings on the GPU, and handles disk persistence."""
    
    def __init__(self, db_dir: str = "./data/chroma_db"):
        self.db_dir = db_dir
        self.device = "cuda"
        print(f"Initializing Embedding Engine. Routing calculations to: {self.device.upper()}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )

    def build_or_load_vector_store(self, document_path: str) -> Chroma:
        """Parses target documentation, builds dense vectors, and saves them locally."""
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Target document not found at: {document_path}")
            
        print(f"Reading compliance source text: {document_path}")
        loader = TextLoader(document_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        print(f"Generated {len(splits)} semantic chunks from policy documents.")
        
        print("Computing embeddings on RTX 3080 and writing to persistent ChromaDB database...")
        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_dir,
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        print(f"Vector directory successfully established at: {self.db_dir}")
        return vector_db

    def get_relevant_rules(self, query: str, k: int = 2) -> str:
        """Queries the local vector store and returns the most semantically relevant text blocks."""
        vector_db = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings,
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        results = vector_db.similarity_search(query, k=k)
        return "\n---\n".join([doc.page_content for doc in results])

if __name__ == "__main__":
    policy_file = "./data/company_policy.txt"
    manager = PolicyVectorStoreManager()
    manager.build_or_load_vector_store(policy_file)
