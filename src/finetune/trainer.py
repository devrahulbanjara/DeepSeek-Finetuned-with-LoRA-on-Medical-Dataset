import os
import torch
import wandb
import transformers
from huggingface_hub import login
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from src.prompts.infer_prompt_template import infer_prompt_template
from src.prompts.train_prompt_template import train_prompt_template
from src.utils.early_stopping import SaveBestModelCallback
from config.settings import HF_TOKEN, WANDB_API_KEY
from src.data.loader import medical_dataset, preprocess_input_data
from transformers import pipeline
import numpy as np

print("\n" + "=" * 60)
print("Initializing environment...")
print("=" * 60 + "\n")

transformers.utils.logging.set_verbosity_info()
login(HF_TOKEN)
wandb.login(key=WANDB_API_KEY)

print(f"CUDA Available       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device           : {torch.cuda.get_device_name(0)}")
else:
    print("GPU Device           : No GPU Detected")
print()

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
max_sequence_length = 800
load_in_4bit = True

print("=" * 60)
print("Loading Model and Tokenizer...")
print("=" * 60 + "\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_sequence_length,
    dtype=torch.float16,
    load_in_4bit=load_in_4bit,
    token=HF_TOKEN
)

print("Model and tokenizer loaded!\n")

EOS_TOKEN = tokenizer.eos_token
print("=" * 60)
print("Preprocessing Dataset...")
print("=" * 60 + "\n")

processed_dataset = medical_dataset.map(
    lambda examples: preprocess_input_data(examples, EOS_TOKEN),
    batched=True,
    desc="Preprocessing dataset"
)

print("Splitting dataset into train and validation sets...")
train_test = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]
print(f"✔ Train samples       : {len(train_dataset)}")
print(f"✔ Validation samples  : {len(eval_dataset)}\n")

print("=" * 60)
print("Setting Up LoRA Adapters...")
print("=" * 60 + "\n")

model_lora = FastLanguageModel.get_peft_model(
    model=model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3047,
    use_rslora=False,
    loftq_config=None
)

model.gradient_checkpointing_enable()

if hasattr(model, '_unwrapped_old_generate'):
    del model._unwrapped_old_generate

print("=" * 60)
print("Configuring Trainer...")
print("=" * 60 + "\n")

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_steps=10,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="wandb",
    evaluation_strategy="epoch",
    eval_steps=200,
    save_strategy="epoch",
    disable_tqdm=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model_lora,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="texts",
    max_seq_length=max_sequence_length,
    dataset_num_proc=1,
    args=training_args,
    callbacks=[SaveBestModelCallback(early_stopping_patience=2)]
)

print("=" * 60)
print("Starting Training...")
print("=" * 60 + "\n")

run = wandb.init(
    project='DeepSeek-R1-Finetune-Medical-Consultant',
    job_type="training",
    anonymous="allow"
)

trainer_stats = trainer.train()
print("\n✔ Training completed!\n")

print("=" * 60)
print("Evaluating Model...")
print("=" * 60 + "\n")

def compute_perplexity(eval_preds):
    losses = eval_preds[0]
    return {"perplexity": np.exp(np.mean(losses))}

eval_result = trainer.evaluate()
perplexity = compute_perplexity((eval_result["eval_loss"],))
print(f"✔ Perplexity: {perplexity['perplexity']:.2f}\n")
wandb.log(perplexity)

print("=" * 60)
print("Generating Sample Predictions...")
print("=" * 60 + "\n")

pipe = pipeline("text-generation", model=model_lora, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

sample_inputs = [
    "Patient reports chest pain and shortness of breath. What could be the possible causes?",
    "What are the recommended treatments for type 2 diabetes?"
]

for prompt in sample_inputs:
    formatted = infer_prompt_template(prompt)
    outputs = pipe(formatted, max_new_tokens=100, do_sample=True, top_k=50, temperature=0.7)
    prediction = outputs[0]["generated_text"]
    print(f"Prompt   : {prompt}")
    print(f"Response : {prediction.strip()}\n")
    wandb.log({f"sample_output_{prompt[:10]}": prediction})

print("=" * 60)
print("Pushing Model to Hugging Face Hub...")
print("=" * 60 + "\n")

model_lora.push_to_hub("devrahulbanjara/deepseek-medical-lora-2000samples", use_auth_token=HF_TOKEN)
tokenizer.push_to_hub("devrahulbanjara/deepseek-medical-lora-2000samples", use_auth_token=HF_TOKEN)

wandb.finish()
print("\n All done!\n")