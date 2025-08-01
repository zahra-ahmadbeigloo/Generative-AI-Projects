## DPO-GPT2-Preference-FineTuning Built with:

| Tool / Library | Purpose                           |
| -------------- | --------------------------------- |
| `transformers` | Model & tokenizer setup           |
| `trl`          | DPO fine-tuning (`DPOTrainer`)    |
| `peft`         | LoRA (Low-Rank Adaptation)        |
| `datasets`     | Dataset loading and preprocessing |
| `torch`        | Deep learning backend             |
| `matplotlib`   | Visualization (optional)          |

---

# DPO-GPT2-Preference-FineTuning

This project implements **Direct Preference Optimization (DPO)** using Hugging Face's `trl` library to fine-tune GPT-2 for better alignment with human preferences. The notebook explores preference-based fine-tuning on a small synthetic dataset and a larger human-rated dataset, demonstrating both training and qualitative generation.

---

## Key Insights

- Fine-tuned GPT-2 using DPO and LoRA on a binary preference dataset.
- Demonstrated qualitative improvements in output relevance compared to base GPT-2.
- Used both local CPU-limited training and pretrained checkpoint loading for analysis.
- Applied generation configuration with temperature, top-k sampling, and max token limits.

---

## Dependencies

Install the required packages:

```bash
pip install torch transformers datasets trl peft matplotlib
````

---

## Project Workflow

### 1. Load and Preprocess Datasets

* Load `Dahoas/synthetic-instruct-gptj-pairwise` and `argilla/ultrafeedback-binarized-preferences-cleaned`
* Select subset samples (due to resource limits)
* Flatten `chosen` and `rejected` responses
* Clean and format into DPO-compatible structure

### 2. Model and Tokenizer Setup

* Base model: `GPT-2`
* Tokenizer: `GPT2Tokenizer`
* Configuration: padding, cache, LoRA support

### 3. LoRA + DPO Configuration

* LoRA on attention layers (`c_attn`, `c_proj`)
* DPOConfig: epochs, batch size, learning rate, prompt length
* Max token length set to 512

### 4. DPOTrainer Initialization

* Initialize trainer with model, tokenizer, dataset, and configurations
* Run `.train()` (commented due to resource limitations)

### 5. Evaluation

* Generate outputs from DPO model and GPT-2
* Compare qualitative differences in generation

### 6. Fallback Option

* Download and load pretrained DPO model from cloud storage
* Reproduce generation results on new prompts


---

## License

This project is licensed under the MIT License.
