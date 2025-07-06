### BERT-Style Masked Language Model built with:  
- **Python** and **TensorFlow/Keras** for model architecture and training  
- **Multi-Head Attention**, **Layer Normalization**, and **Positional Encoding**  
- **TextVectorization** for tokenization and input embedding  
- **Custom masking logic** for Masked Language Modeling (MLM)  
- **NumPy** for data processing and loss shaping  

---

# BERT-Style Masked Language Model (MLM Pretraining)

This project implements a simplified **BERT-like Transformer encoder** from scratch using TensorFlow/Keras. The model is trained using the **Masked Language Modeling (MLM)** objective, where some input tokens are masked and the model learns to predict them. This approach captures **bidirectional context** and reflects the core of modern large language model pretraining.

---

### **Key Insights:**  
- Designed a **BERT-style architecture** using encoder blocks with residual connections and attention layers.  
- Implemented **masked token prediction** logic to simulate the MLM objective used in original BERT.  
- Used a simple text corpus tokenized with `TextVectorization`.  
- Validated learning by observing improved prediction accuracy over masked positions.  
- Reinforces foundational understanding of pretraining strategies in language models.

---

## Dependencies

```bash
pip install tensorflow numpy
```
---

## License
This project is licensed under the MIT License.

