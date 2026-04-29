"""The state dict that flows between every graph node.

`messages` is the only field that accumulates — `operator.add` tells LangGraph
to append rather than overwrite, so multi-turn history builds up correctly.
Everything else gets replaced by whichever node returns it last, which is
why /agent_routes resets `scratchpad` and `citations` at the start of each turn.
"""

import operator
from typing import TypedDict, Annotated, Sequence, List, Optional

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tenant_id: str
    patient_id: str
    user_role: str
    # Server-resolved User row dropped into the summarizer prompt. None when
    # the session is owned by an "unknown" placeholder.
    user_profile: Optional[dict]
    session_id: Optional[str]
    citations: List[str]
    scratchpad: str
    next_step: Optional[str]
