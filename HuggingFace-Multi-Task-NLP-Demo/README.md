# HuggingFace Multi-Task NLP Demo

This project showcases the usage of various pre-trained Transformer models from Hugging Face for a variety of common NLP tasks, including sentiment classification, text generation, translation, language detection, and masked word prediction. The implementation leverages both manual model loading and the high-level `pipeline()` API.

---

## Tasks and Models Used

### 1. Sentiment Classification (DistilBERT)
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Task**: Binary sentiment classification (positive/negative).
- **Workflow**:
  - Load tokenizer and model manually.
  - Preprocess input text.
  - Perform inference and apply softmax.
  - Extract and display predicted label.

### 2. Text Generation (GPT-2)
- **Model**: `gpt2`
- **Task**: Auto-regressive text generation.
- **Workflow**:
  - Tokenize prompt (`"Once upon a time"`).
  - Generate sequences using `.generate()`.
  - Decode and display generated text.

### 3. Text Classification using `pipeline()`
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Workflow**:
  - Use `pipeline("text-classification")`.
  - Classify a sample spam-style message.

### 4. Language Detection (XLM-RoBERTa)
- **Model**: `papluca/xlm-roberta-base-language-detection`
- **Workflow**:
  - Use `pipeline("text-classification")`.
  - Detect language from multilingual input text (e.g., French).

### 5. Text Generation with `pipeline()` (GPT-2)
- **Model**: `gpt2`
- **Workflow**:
  - Use `pipeline("text-generation")`.
  - Generate a single output for a given prompt.

### 6. Translation with T5
- **Model**: `t5-small`
- **Task**: Text-to-text generation for translation.
- **Workflow**:
  - Prompt: `"translate English to French: How are you?"`
  - Use `pipeline("text2text-generation")` to translate.

### 7. Masked Word Prediction (BERT)
- **Model**: `bert-base-uncased`
- **Task**: Fill-in-the-blank using masked language modeling.
- **Workflow**:
  - Input: `"Rabbit eats delicious [MASK]."`
  - Use `pipeline("fill-mask")` to infer the most likely completions.

---

## Tools & Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Pretrained models:
  - `distilbert-base-uncased-finetuned-sst-2-english`
  - `gpt2`
  - `t5-small`
  - `papluca/xlm-roberta-base-language-detection`
  - `bert-base-uncased`

---

## Key Insights

- Demonstrates a variety of NLP tasks using the Hugging Face `transformers` library.
- Explores both low-level and high-level interfaces.
- Highlights multilingual and multimodal text capabilities.
- Serves as a practical introduction to rapid prototyping in NLP.

---

## Installation

Install dependencies via pip:

```bash
pip install torch transformers
```

---

## License

This project is for educational and demonstration purposes under the MIT License.
