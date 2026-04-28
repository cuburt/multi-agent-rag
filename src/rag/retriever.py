from sqlmodel import Session, select
from src.db.session import engine
from src.db.models import Document
from src.rag.embeddings import get_embedding

def retrieve_documents(query: str, tenant_id: str, doc_type: str = None, top_k: int = 3):
    """
    Hybrid retrieval: vector similarity + metadata filtering (hard tenant_id filter).
    """
    query_embedding = get_embedding(query)
    
    with Session(engine) as session:
        # Filter by tenant_id (Mandatory Security Control)
        stmt = select(Document).where(Document.tenant_id == tenant_id)
        
        # Optional metadata filter
        if doc_type:
            stmt = stmt.where(Document.doc_type == doc_type)
            
        # Order by vector L2 distance (nearest neighbor)
        # We assume embedding column exists and uses pgvector's vector type
        stmt = stmt.order_by(Document.embedding.l2_distance(query_embedding)).limit(top_k)
        
        results = session.exec(stmt).all()
        
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "doc_type": doc.doc_type
            }
            for doc in results
        ]
