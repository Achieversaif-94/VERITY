from pydantic import BaseModel
from verity import llm
from verity.trace import Tracer

class Pick(BaseModel):
    best_tool: str
    reason: str

tracer = Tracer("traces/smoke.jsonl", run_id="smoke")
out = llm.complete_json(
    "You pick the single best tool for a task. Available tools: read_file, write_file, run_python, grep, csv_query, http_fetch.",
    "Task: sum the third column of sales.csv and verify the result.",
    Pick,
    tracer=tracer,
    node_id="smoke",
)
print("best_tool:", out.best_tool)
print("reason:", out.reason)
from verity.trace import verify_chain
print("trace:", verify_chain(tracer.path))