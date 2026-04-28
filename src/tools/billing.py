from sqlmodel import Session, select
from src.db.session import engine
from src.db.models import Claim

def check_claim_status(tenant_id: str, patient_id: str) -> str:
    """Check the status of insurance claims for a patient."""
    with Session(engine) as session:
        stmt = select(Claim).where(
            Claim.tenant_id == tenant_id,
            Claim.patient_id == patient_id
        ).order_by(Claim.service_date.desc())
        
        claims = session.exec(stmt).all()
        
        if not claims:
            return "No claims found for this patient."
            
        res = []
        for claim in claims:
            res.append(f"- Claim {claim.id}: {claim.service_date} | Amount: ${claim.amount} | Status: {claim.status} | Details: {claim.details}")
        return "\n".join(res)
