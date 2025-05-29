TEMPLATE = """You are a medical expert in clinical reasoning, diagnostics, and treatment planning. Carefully analyze the question below and provide a clear, logical, and well-reasoned answer.

### Query:
{}

### Answer:
<think>{}
"""

def infer_prompt_template(question: str) -> str:
    return TEMPLATE.format(question, "")