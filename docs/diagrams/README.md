# Diagrams

draw.io source files plus their rendered PNGs for every diagram referenced
from the docs. Edit the `.drawio` files in <https://app.diagrams.net> (File
→ Open) or with the draw.io VS Code / IntelliJ extension, then re-run the
export step below to refresh the PNGs.

| Source | Rendered | What it shows | Referenced from |
|---|---|---|---|
| [architecture.drawio](architecture.drawio) | [architecture.png](architecture.png) | Full system: client, FastAPI, the LangGraph nodes, and the stores behind them. | [readout.md §1](../readout.md) · [design.md](../design.md) |
| [graphs.drawio](graphs.drawio) | [graphs.png](graphs.png) | The two compiled graphs side by side — `agent_app` (mutating) and `ask_app` (read-only). | [design.md](../design.md) · [readout.md](../readout.md) |
| [retrieval.drawio](retrieval.drawio) | [retrieval.png](retrieval.png) | Hybrid RAG flow: vector + lexical + RRF, with tenant + RBAC filters applied before ranking. | [design.md §Retrieval Strategy](../design.md) |
| [model-tiers.drawio](model-tiers.drawio) | [model-tiers.png](model-tiers.png) | ROUTER / AGENTIC / SYNTHESIS tiers with their hardcoded fallback chains and paid Vercel anchor. | [design.md §Model Routing Strategy](../design.md) |
| [phi-redaction.drawio](phi-redaction.drawio) | [phi-redaction.png](phi-redaction.png) | The three PHI-redaction layers — input, log records, output. | [design.md](../design.md) · [prompts.md](../prompts.md) |
| [observability.drawio](observability.drawio) | [observability.png](observability.png) | How a request becomes a Langfuse trace and how `/metrics` rolls runtime + offline eval baselines into one JSON. | [design.md](../design.md) · [readout.md](../readout.md) |

## Regenerating the PNGs

A docker one-liner from the repo root re-exports every `.drawio` in this
folder. Output filenames need a quick rename so they match the markdown
references; the helper script below handles that.

```bash
./scripts/render-diagrams.sh
```

What it runs (verbatim, if you'd rather invoke directly):

```bash
docker run --rm -v "$(pwd)/docs/diagrams:/data" \
  rlespinasse/drawio-export -f png --output .
```

The image is `rlespinasse/drawio-export` — a maintained wrapper around the
official drawio Electron app running headless. First run pulls ~1 GB.
