### Custom GPT built with:  
- **Python** and **TensorFlow/Keras** for Transformer architecture and training  
- **Custom Positional Encoding** for token order awareness  
- **Masked Multi-Head Attention** for autoregressive decoding  
- **NumPy** and **TextVectorization** for data preparation  
- **Custom token generation loop** for step-wise sequence output  

---

# Custom GPT from Scratch (Decoder-Only Transformer)

This project implements a simplified **GPT-style language model** from scratch using a **decoder-only Transformer architecture**. The model is trained to generate coherent text sequences by learning next-token prediction on a toy text corpus. It demonstrates the principles behind large language models like GPT-2, including **causal attention**, **masked decoding**, and **autoregressive generation**.

---

### **Key Insights:**  
- Designed and trained a **GPT-like model** using custom-built Keras layers.  
- Implemented **masked attention**, **positional encoding**, and **causal self-attention**.  
- Used a simple token-based dataset to train the model for next-word prediction.  
- Generated text samples using a manual decoding loop, showing token-by-token prediction.  
- Reinforced understanding of **decoder-only Transformer pipelines** for language modeling.

---

## Dependencies

```bash
pip install tensorflow numpy
```
---

## License

This project is licensed under the MIT License.
