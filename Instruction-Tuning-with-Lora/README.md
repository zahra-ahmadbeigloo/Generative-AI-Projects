## This project is built with:

- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [TRL (Training LLMs)](https://github.com/huggingface/trl)
- [Evaluate](https://github.com/huggingface/evaluate)
- PyTorch & Hugging Face Pipelines
- Matplotlib, Pandas, TQDM, SacreBLEU

---

# Instruction Tuning with LLMs using LoRA

This project demonstrates how to instruction-tune Large Language Models (LLMs) using the CodeAlpaca-20k dataset. It includes preprocessing, base model evaluation, LoRA fine-tuning, and comparative analysis of generation quality with SacreBLEU scoring.

---

## Project Overview

The objective of this project is to adapt pretrained language models to follow task-specific instructions more effectively. We experiment with:
- **OPT-350M**: Tested and evaluated as the base model.
- **GPT-Neo-125M**: Used with an alternative response formatting style.
- **LoRA (Low-Rank Adaptation)**: A parameter-efficient fine-tuning method applied to instruction-tune models.

---

## Key Insights

- The base OPT-350M model struggles to generate coherent responses with low BLEU scores.
- Instruction-tuned models using **LoRA** show measurable improvement in generation quality.
- Using a QA-style prompt format ("### Question:", "### Answer:") can significantly affect output style and coherence.
- The CodeAlpaca-20k dataset is filtered to focus on instruction-only samples for cleaner supervision.

---

## Dataset

**CodeAlpaca-20k**: An instruction-style dataset consisting of ~20k examples.

Filtering step:
- Only examples with empty input (`"input": ""`) are used.

Split:
- 80% training / 20% test

---

## Usage Instructions

To replicate or extend this project:

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
  ```bash
  pip install -U datasets transformers peft trl evaluate torch sacrebleu
  ```

3. Download the dataset:

   ```bash
   wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/WzOT_CwDALWedTtXjwH7bA/CodeAlpaca-20k.json
   ```

4. Run the notebook step-by-step to:

   * Load and preprocess data
   * Evaluate base model
   * Fine-tune with LoRA
   * Compare BLEU scores
   * Visualize training loss

---

## Evaluation

* **Metric**: SacreBLEU
* **Results**:

  * Base Model: BLEU ≈ 0.0
  * LoRA-tuned Model: BLEU ≈ 1.2

*(Note: Results may vary slightly depending on runtime environment and batch size.)*

---

## Outputs

* `instruction-tuning-final-model-lora`: LoRA fine-tuned model directory (optional save)
* `training_loss.png`: Training curve plot
* `generated_outputs_base.pkl` & `generated_outputs_lora.pkl`: Generated texts from base and LoRA models
* `log_history_lora.json`: Training logs for LoRA model

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

* Hugging Face for the open-source libraries
* IBM for the dataset and cloud resources
