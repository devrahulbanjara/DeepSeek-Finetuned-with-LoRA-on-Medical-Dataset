# Medical SFT Dataset for HuatuoGPT-o1 Fine-tuning

## Overview

This dataset is designed for supervised fine-tuning (SFT) of medical large language models (LLMs) specialized in advanced medical reasoning. It contains high-quality medical question-answer pairs generated and verified with the help of GPT-4o and a medical verifier to ensure accuracy and reliability.

The data is sourced from the [Medical-R1 dataset](https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data) and contains verifiable medical problems aimed at improving the reasoning ability of medical LLMs.

---

## Usage in Fine-tuning

This dataset has been used to fine-tune **DeepSeek-R1-Distill-Llama-8B**, leveraging its distilled medical reasoning capabilities and combining it with domain-specific supervision to enhance performance on medical question-answering and complex reasoning tasks.

---

## Dataset Structure

The dataset is organized into language-specific configurations:

- **en**: English medical SFT data (19.7k examples)
- **zh**: Chinese medical SFT data
- **en_mix**: Mixed English medical and general instruction data
- **zh_mix**: Mixed Chinese medical and general instruction data

Each configuration contains a JSON file with rows of data entries structured as follows:

| Field         | Description                                                |
| ------------- | ---------------------------------------------------------- |
| `Question`    | Medical question or clinical scenario requiring reasoning. |
| `Complex_CoT` | Chain-of-thought style reasoning supporting the answer.    |
| `Response`    | Model's answer or explanation to the question.             |

---

## Example Entry

**Question:**  
_Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?_

**Complex_CoT:**  
_Okay, let's see what's going on here. We've got sudden weakness in the person's left arm and leg - and that screams something neuro-related, maybe a stroke? But wait, there's more. The right lower leg is swollen and tender, which is like waving a big flag for deep vein thrombosis, especially after a long flight or sitting around a lot. So, now I'm thinking, how could a clot in the leg end up causing issues like weakness or stroke symptoms? Oh, right! There's this thing called a paradoxical embolism. It can happen if there's some kind of short circuit in the heart - like a hole that shouldn't be there. Let's put this together: if a blood clot from the leg somehow travels to the left side of the heart, it could shoot off to the brain and cause that sudden weakness by blocking blood flow there. Hmm, but how would the clot get from the right side of the heart to the left without going through the lungs and getting filtered out? Here's where our cardiac anomaly comes in: a patent foramen ovale or PFO. That's like a sneaky little shortcut in the heart between the right and left atria. And it's actually pretty common, found in about a quarter of adults, which definitely makes it the top suspect here. So with all these pieces - long travel, leg clot, sudden weakness - a PFO fits the bill perfectly, letting a clot cross over and cause all this. Everything fits together pretty neatly, so I'd bet PFO is the heart issue waiting to be discovered. Yeah, that really clicks into place!_

**Response:**  
_The specific cardiac abnormality most likely to be found in this scenario is a patent foramen ovale (PFO). This condition could allow a blood clot from the venous system, such as one from a deep vein thrombosis in the leg, to bypass the lungs and pass directly into the arterial circulation. This can occur when the clot moves from the right atrium to the left atrium through the PFO. Once in the arterial system, the clot can travel to the brain, potentially causing an embolic stroke, which would explain the sudden weakness in the left arm and leg. The connection between the recent travel, which increases the risk of deep vein thrombosis, and the neurological symptoms suggests the presence of a PFO facilitating a paradoxical embolism._

---

## Citation

```bibtex
@misc{chen2024huatuogpto1medicalcomplexreasoning,
  title={HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs},
  author={Junying Chen and Zhenyang Cai and Ke Ji and Xidong Wang and Wanlong Liu and Rongsheng Wang and Jianye Hou and Benyou Wang},
  year={2024},
  eprint={2412.18925},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2412.18925},
}
```
