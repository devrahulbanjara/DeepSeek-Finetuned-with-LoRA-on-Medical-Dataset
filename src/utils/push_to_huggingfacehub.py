import torch
from unsloth import FastLanguageModel
from huggingface_hub import login
from transformers import AutoTokenizer
from config.settings import HF_TOKEN

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CHECKPOINT_DIR = "outputs/checkpoint-1686"
HF_REPO_NAME = "devrahulbanjara/DeepSeek-R1-Distill-Llama-8B-medical-finetune"
MAX_SEQ_LENGTH = 800
LOAD_IN_4BIT = True

def print_section(title):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("\n" + "=" * 70)
            print(f"{title}")
            print("=" * 70 + "\n")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@print_section("Logging in to Hugging Face Hub")
def login_hf():
    login(token=HF_TOKEN)

@print_section("Loading base model and tokenizer with Unsloth")
def load_model():
    return FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN
    )

@print_section(f"Loading LoRA adapter from checkpoint: {CHECKPOINT_DIR}")
def load_adapter(model):
    model.load_adapter(CHECKPOINT_DIR)

@print_section(f"Pushing model and tokenizer to Hugging Face Hub:\n{HF_REPO_NAME}")
def push_model(model, tokenizer):
    model.push_to_hub(HF_REPO_NAME, use_auth_token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_NAME, use_auth_token=HF_TOKEN)

@print_section("Model and tokenizer successfully pushed to Hugging Face Hub!")
def success():
    pass

# === EXECUTION ===
if __name__ == "__main__":
    login_hf()
    model, tokenizer = load_model()
    load_adapter(model)
    push_model(model, tokenizer)
    success()
