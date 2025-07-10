### BERT Multiclass Classification & LLM Instruction Fine-Tuning built with:  
- **Python** and **PyTorch** for training and evaluation  
- **Hugging Face Transformers** for BERT (`AutoModelForSequenceClassification`) and OPT (`AutoModelForCausalLM`)  
- **Hugging Face Datasets** for loading `yelp_review_full` and OpenAssistant datasets  
- **SFTTrainer & SFTConfig** for instruction tuning (supervised fine-tuning)  
- **Custom training loop** with `AdamW`, `LambdaLR`, and PyTorch `DataLoader`  
- **Matplotlib** for training loss visualization  
- **Hugging Face Pipeline** for conversational text generation  
- **GPU support** and model saving/loading with `torch.save()` and `.load_state_dict()`

---

# BERT-Based Classification and Instruction-Tuned OPT for Conversational Generation

This project combines two core tasks in modern NLP workflows:

1. **Multiclass text classification** using **`bert-base-cased`**, fine-tuned on the `yelp_review_full` dataset
2. **Instruction tuning** of a **causal language model (OPT-350M)** on the OpenAssistant-Guanaco dataset using **`SFTTrainer`**

The project showcases both **discriminative** and **generative** NLP techniques, from classification to conversational fine-tuning.

---

## Workflow Summary

### Part 1: Multiclass Sentiment Classification (BERT)
- Dataset: `yelp_review_full` (650k training, 50k test)
- Tokenized using `BertTokenizerFast` with truncation/padding
- BERT model loaded via `AutoModelForSequenceClassification` (5 output labels)
- Training:
  - Optimizer: `AdamW`
  - Scheduler: `LambdaLR`
  - Loss & backpropagation done manually
- Evaluation: Used `torchmetrics.Accuracy` for 5-class predictions

### Part 2: Conversational Instruction Tuning (OPT-350M)
- Dataset: `timdettmers/openassistant-guanaco`
- Model: `facebook/opt-350m` for text generation (`AutoModelForCausalLM`)
- DataCollator: `DataCollatorForCompletionOnlyLM` with instruction-response structure
- Fine-tuning:
  - Trainer: `SFTTrainer`
  - Args: `SFTConfig` with `fp16=True`, `max_seq_length=1024`
  - Evaluation and generation using `pipeline("text-generation")`
- Model loading/saving demonstrated with `.pt` checkpoint files

---

## Key Insights

- Achieved efficient **fine-tuning of BERT** on a large multiclass dataset using PyTorch-native training
- Used **OPT and instruction tuning** to align a conversational model with question-answer prompts
- Demonstrated generation from instruction-tuned LLM using Hugging Face `pipeline`
- Used both **classification accuracy** and **text generation quality** to evaluate performance

---

## Dependencies

```bash
pip install torch transformers datasets matplotlib accelerate peft
````

Optional (for `SFTTrainer` usage):

```bash
pip install trl
```

---

## Technologies Used

| Task                      | Tools                                        |
| ------------------------- | -------------------------------------------- |
| Multiclass Classification | BERT, Yelp, PyTorch, AdamW                   |
| Conversational Generation | OPT-350M, SFTTrainer, OpenAssistant          |
| Data Handling             | Hugging Face Datasets                        |
| Tokenization              | AutoTokenizer (BERT, OPT)                    |
| Evaluation                | torchmetrics.Accuracy, Hugging Face pipeline |
| Instruction Format        | `### Human:` and `### Assistant:` prompts    |
| Model Save/Load           | `torch.save`, `load_state_dict`              |

---

## License

This project is licensed under the MIT License.
