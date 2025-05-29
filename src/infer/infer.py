import torch
import unsloth
from transformers import AutoTokenizer
from huggingface_hub import login
from unsloth import FastLanguageModel
from config.settings import HF_TOKEN
from src.prompts.infer_prompt_template import infer_prompt_template

login(HF_TOKEN)

model_name = "devrahulbanjara/DeepSeek-R1-Distill-Llama-8B-medical-finetune"
max_sequence_length = 800

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_sequence_length,
    dtype=torch.float16,
    load_in_4bit=True,
    token=HF_TOKEN
)

sample_inputs = [
    "How do i cure a viral fever happened during season change? ",
    "What are the recommended treatments for type 2 diabetes? Explain me to the point."
]

for i, prompt in enumerate(sample_inputs, start=1):
    formatted = infer_prompt_template(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        top_k=50,
        temperature=0.7
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("### Answer:")[1].strip() if "### Answer:" in decoded else decoded

    print("=" * 80)
    print(f"[{i}] Query:")
    print(prompt)
    print("\nAnswer:")
    print(response)
    print("=" * 80 + "\n")