# 🚀 Fine-tuning Gemma-3-4B-it with LoRA for Financial QA

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a comprehensive Jupyter Notebook demonstrating how to fine-tune the **google/gemma-3-4b-it** model for financial question-answering tasks. The implementation leverages **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning (PEFT) and the **Supervised Fine-tuning (SFT) Trainer** from the TRL library.

## 🎯 Objective

The primary goal of this project is to adapt a powerful, pre-trained language model (**Gemma-3**) to better understand and respond to financial questions that require **multi-step reasoning**. This is achieved by fine-tuning it on the `TheFinAI/Fino1_Reasoning_Path_FinQA` dataset, which provides financial questions along with their **chain-of-thought (CoT)** reasoning paths and final answers.

## 🛠️ Key Components

| Component | Description |
|-----------|-------------|
| **🤖 Model** | `google/gemma-3-4b-it` - Instruction-tuned version of Google's Gemma-3 model with 4B parameters |
| **📊 Dataset** | `TheFinAI/Fino1_Reasoning_Path_FinQA` - Financial QA dataset with chain-of-thought reasoning |
| **⚡ Fine-tuning** | **LoRA** (Low-Rank Adaptation) via the `peft` library |
| **🏋️ Trainer** | **SFTTrainer** from the `trl` library for supervised fine-tuning |

## 🚀 Methodology

The notebook follows these key steps:

### 1. 🔧 Environment Setup
- Installs necessary libraries: `transformers`, `trl`, `peft`, and `datasets`
- Configures CUDA environment for GPU training

### 2. 📥 Model Loading
- Loads the pre-trained `Gemma3ForConditionalGeneration` model
- Initializes the corresponding tokenizer from Hugging Face Hub

### 3. 📋 Dataset Preprocessing
- **Dataset Loading**: Loads the financial QA dataset
- **Custom Prompt Template**: Structures data for supervised fine-tuning with placeholders for:
  - Context
  - Question  
  - Chain-of-thought reasoning (`<think>...</think>`)
  - Final response
- **Formatting**: Applies template and appends EOS tokens

### 4. ⚙️ LoRA Configuration
- Creates `LoraConfig` specifying target layers (`q_proj`, `k_proj`, `v_proj`, etc.)
- Enables parameter-efficient fine-tuning with significantly reduced trainable parameters

### 5. 🏋️ SFT Trainer Setup
- Configures `SFTConfig` with hyperparameters (learning rate, batch size, logging)
- Initializes `SFTTrainer` with model, dataset, tokenizer, and LoRA configuration

### 6. 🎯 Fine-tuning
- Executes `trainer.train()` to begin the fine-tuning process

## 📋 Prerequisites

- 🐍 **Python 3.x**
- 🖥️ **GPU Required** (NVIDIA T4, A100, or similar CUDA-enabled device)
- 🤗 **Hugging Face Account** with Gemma-3 model access and valid token

## ⚙️ How to Run

### 1. 📂 Clone Repository
```bash
git clone https://github.com/AhemdMahmoud/lora-sft-gemma-3-4b-.git
cd lora-sft-gemma-3-4b-.git
```

### 2. 📦 Install Dependencies
```bash
pip install trl "peft>=0.11.1" "transformers>=4.41.2" "datasets>=2.19.2" torch
```

### 3. 🔐 Hugging Face Authentication
Run this in your notebook to authenticate:
```python
from huggingface_hub import notebook_login
notebook_login()
```
*Paste your access token when prompted*

### 4. 🚀 Execute Notebook
Run all cells in `Gamma__3_Finetuing_lora_SFT.ipynb` sequentially. The notebook handles:
- Model/data loading
- Preprocessing 
- Training loop execution

## 🔧 Configuration Details

### ⚡ LoRA Configuration
```python
peft_config = LoraConfig(
    r=6,                    # Rank of adaptation
    lora_alpha=8,          # LoRA scaling parameter
    lora_dropout=0.05,     # Dropout probability
    bias="none",           # Bias type
    target_modules=[       # Target transformer modules
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)
```

### 🏋️ SFT Trainer Configuration
```python
args = SFTConfig(
    output_dir="./sft_output",           # Output directory
    max_steps=1000,                      # Maximum training steps
    per_device_train_batch_size=1,       # Training batch size
    per_device_eval_batch_size=1,        # Evaluation batch size
    learning_rate=5e-5,                  # Learning rate
    logging_steps=10,                    # Logging frequency
    save_steps=100,                      # Checkpoint saving frequency
)
```

## 📊 Expected Results

After fine-tuning, the model should demonstrate improved performance on:
- 💰 **Financial reasoning tasks**
- 🧠 **Multi-step problem solving**
- 🔗 **Chain-of-thought generation**
- 📈 **Domain-specific financial knowledge**

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- 🐛 Report bugs
- 💡 Suggest enhancements
- 🔧 Submit pull requests
- 📖 Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🤗 **Hugging Face** for the transformers and datasets libraries
- 🧠 **Google** for the Gemma-3 model
- 📊 **TheFinAI** for the Fino1 reasoning dataset
- ⚡ **Microsoft** for the LoRA technique

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

</div>
