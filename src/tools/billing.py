from collections import Counter, defaultdict
from sqlmodel import Session, select
from src.db.session import engine
from src.db.models import Claim

def check_claim_status(tenant_id: str, patient_id: str) -> str:
    """Returns this patient's claims as a Summary block followed by line items.

    The Summary covers the questions patients actually ask — total outstanding,
    total paid, denied count — pre-computed so the summarizer LLM doesn't have
    to derive arithmetic from line items (where it tends to hallucinate).
    "Outstanding" here means submitted + denied, i.e. anything not yet paid.
    """
    with Session(engine) as session:
        stmt = select(Claim).where(
            Claim.tenant_id == tenant_id,
            Claim.patient_id == patient_id,
        ).order_by(Claim.service_date.desc())

        claims = session.exec(stmt).all()

        if not claims:
            return "No claims found for this patient."

        status_counts: Counter[str] = Counter()
        status_totals: dict[str, float] = defaultdict(float)
        for c in claims:
            status_counts[c.status] += 1
            status_totals[c.status] += c.amount

        outstanding = status_totals.get("submitted", 0.0) + status_totals.get("denied", 0.0)
        paid = status_totals.get("paid", 0.0)
        total_billed = sum(status_totals.values())

        summary_lines = [
            "Summary:",
            f"  Total claims: {len(claims)} (${total_billed:.2f} billed)",
            f"  Outstanding balance: ${outstanding:.2f} "
            f"(submitted: {status_counts.get('submitted', 0)}, "
            f"denied: {status_counts.get('denied', 0)})",
            f"  Paid: ${paid:.2f} ({status_counts.get('paid', 0)} claim(s))",
        ]

        line_items = [
            f"- Claim {c.id}: {c.service_date} | Amount: ${c.amount} | "
            f"Status: {c.status} | Details: {c.details}"
            for c in claims
        ]
        return "\n".join(summary_lines + ["", "Line items:"] + line_items)
