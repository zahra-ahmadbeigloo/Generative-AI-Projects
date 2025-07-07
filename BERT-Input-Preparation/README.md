### BERT Input Preprocessing built with:  
- **Python** for text preprocessing and sequence handling  
- **Hugging Face Transformers** for tokenization using pretrained BERT tokenizer  
- **TensorFlow/Keras** for model-compatible input preparation  
- **IMDB Review Dataset** for binary sentiment input examples  
- **NumPy** for tensor formatting and attention mask logic  

> This notebook prepares data for downstream **generative tasks** using BERT, such as masked language modeling or question answering.

---

# BERT Input Preparation for Text Classification (IMDB)


This project demonstrates how to **load and process raw text datasets** for input into a **BERT-based Transformer model**. Using Hugging Face’s `BertTokenizer`, the IMDB movie review dataset is preprocessed to produce BERT-compatible inputs — including token IDs, attention masks, and segment embeddings — suitable for text classification or fine-tuning.

---

### **Key Insights:**  
- Loaded IMDB review data and prepared it for input to pretrained BERT models.  
- Applied **WordPiece tokenization** and **special tokens** ([CLS], [SEP]) with `BertTokenizer`.  
- Generated **attention masks** and **token type IDs** to support BERT architecture.  
- Converted string inputs into padded sequences and tensors.  
- Provided reusable code for preprocessing any text classification dataset using BERT.

---

## Dependencies

```bash
pip install transformers tensorflow numpy
```

---

## License

This project is licensed under the MIT License.
