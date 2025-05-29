import torch
import unsloth
from transformers import AutoTokenizer
from huggingface_hub import login
from unsloth import FastLanguageModel
from config.settings import HF_TOKEN
from src.prompts.infer_prompt_template import infer_prompt_template

login(HF_TOKEN)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
max_sequence_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_sequence_length,
    dtype=torch.float16,
    load_in_4bit=True,
    token=HF_TOKEN
)

sample_inputs = [
    "Patient reports chest pain and shortness of breath. What could be the possible causes?",
    "What are the recommended treatments for type 2 diabetes?"
]

for idx, prompt in enumerate(sample_inputs, start=1):
    formatted = infer_prompt_template(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    input_length = inputs["input_ids"].shape[1]
    max_new_tokens = max(128, max_sequence_length - input_length)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = decoded.find("<think>")
    response = decoded[response_start:].strip() if response_start != -1 else decoded.strip()

    print("=" * 80)
    print(f"[{idx}] Query:\n{prompt}\n")
    print("Answer:")
    print(response)
    print("=" * 80 + "\n")
