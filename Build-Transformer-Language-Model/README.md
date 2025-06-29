### Transformer Language Modeling built with:  
- **Python** and **TensorFlow/Keras** for deep learning model construction  
- **TextVectorization** layer for tokenizing raw text  
- **Custom Transformer blocks** for encoder-decoder architecture  
- **Shakespeare corpus** as the training dataset  
- **Sequential autoregressive decoding** for text generation  

---

# Build a Transformer-based Language Model from Scratch

This project implements a **Transformer-based language model** using **TensorFlow/Keras**, trained on the **Shakespeare dataset**. The model is designed to predict and generate natural language sequences in an autoregressive manner. All key components—attention, positional encoding, masked self-attention, and feedforward layers—are implemented manually to provide a deeper understanding of how Transformer architectures function.

---

## Key Insights

- Built a full **Transformer model architecture from scratch**, not using Hugging Face or pretrained weights.
- Used **TextVectorization** to preprocess and tokenize sequences for training.
- Incorporated **masked multi-head attention** for decoder logic, mimicking causal prediction.
- Generated novel text sequences that resemble Shakespearean style.
- Demonstrated how to train and sample from a Transformer language model manually.

---

## Sample Output

> **Input:** “to be or not to be”  
> **Generated:**  
> “to be or not to be the man that shall live in time’s grace and hold his tongue.”

---

## Dependencies

```bash
pip install tensorflow numpy matplotlib
````

---

## Dataset

* [Shakespeare text](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt)

---

##License

This project is licensed under the MIT License.
