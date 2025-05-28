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
_Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation?_

**Complex_CoT:**  
_The symptoms suggest a paradoxical embolism caused by a clot crossing from the venous to arterial system through a patent foramen ovale (PFO), a common cardiac anomaly._

**Response:**  
_The specific cardiac abnormality most likely to be found is a patent foramen ovale (PFO), allowing a clot from deep vein thrombosis to bypass the lungs and cause embolic stroke._

---

## Citation

If you use this dataset or model, please cite:

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
