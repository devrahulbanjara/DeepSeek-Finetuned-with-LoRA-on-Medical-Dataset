from datasets import load_dataset
from src.prompts.train_prompt_template import train_prompt_template
from tqdm.auto import tqdm

print("Loading medical dataset...")
medical_dataset = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT", 
    "en", 
    split = "train[:5000]", 
    trust_remote_code = True
)
print(f"Medical dataset loaded successfully! Total samples: {len(medical_dataset)}")

def preprocess_input_data(examples, eos_token):
  inputs = examples["Question"]
  cots = examples["Complex_CoT"]
  outputs = examples["Response"]

  texts = []

  for input, cot, output in zip(inputs, cots, outputs):
    text = train_prompt_template.format(input, cot, output) + eos_token
    texts.append(text)

  return {
      "texts" : texts,
  }