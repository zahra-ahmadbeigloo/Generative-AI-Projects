### BERT-Based Text Classification with Adapters & LoRA built with:  
- **Python** and **PyTorch** for model training and evaluation  
- **Hugging Face Transformers** for pretrained BERT and adapter integration  
- **Hugging Face Adapters library** for lightweight fine-tuning using **adapter modules**  
- **LoRA (Low-Rank Adaptation)** for parameter-efficient training of transformer layers  
- **Hugging Face Datasets** for loading and preprocessing the IMDB dataset  
- **BertTokenizerFast** for tokenization with truncation, padding, and attention masks  
- **AdamW optimizer** with a linear learning rate scheduler  
- **Matplotlib** for visualizing loss and accuracy curves  

---

# Adapter- and LoRA-Based Fine-Tuning of BERT for IMDB Sentiment Classification

This project fine-tunes a **pretrained BERT model** on the **IMDB movie review dataset** using **parameter-efficient fine-tuning techniques** — namely **Adapters** and **LoRA (Low-Rank Adaptation)**. The goal is to reduce training time and memory usage while maintaining strong classification performance.

By leveraging Hugging Face’s `transformers.adapters` framework, the model is trained using **only a small number of additional parameters** (adapter layers), making this approach ideal for resource-constrained environments or multi-task learning setups.

---

### **Key Insights:**  
- Loaded the `bert-base-uncased` model and dynamically added **adapter modules** for the classification task.  
- Applied **LoRA configuration** inside adapter layers to reduce trainable parameter count via rank decomposition.  
- Tokenized text using `BertTokenizerFast`, generating token IDs, attention masks, and segment IDs.  
- Created custom PyTorch `Dataset` and `DataLoader` objects from tokenized inputs.  
- Trained the model using **AdamW** and a **linear scheduler** from Hugging Face's optimization utilities.  
- Visualized the **training loss and validation accuracy** over multiple epochs using `matplotlib`.  
- Achieved effective fine-tuning with minimal parameter updates, thanks to adapters and LoRA.

---

## Fine-Tuning Strategy

| Method | Description |
|--------|-------------|
| **Adapters** | Inserted small bottleneck MLPs into transformer layers. Only adapter weights are updated during training. |
| **LoRA** | Low-rank matrices added to attention projections, further reducing the number of trainable parameters. |
| **Combined** | Used Hugging Face’s `ModelWithHeads` and adapter config system to apply both methods concurrently. |

---

## Dependencies

```bash
pip install transformers datasets torch matplotlib
````

If using adapters with LoRA:

```bash
pip install adapter-transformers
```

---

## Workflow Overview

1. Load `bert-base-uncased` model using Hugging Face Transformers
2. Add task-specific adapter with LoRA configuration
3. Load and tokenize IMDB dataset
4. Prepare PyTorch DataLoaders for training and validation
5. Train using AdamW and evaluate using classification accuracy
6. Visualize results and save the adapter setup

---

## Technologies Used

* **PyTorch**
* **Hugging Face Transformers & Adapters**
* **BERT**
* **LoRA (Low-Rank Adaptation)**
* **IMDB sentiment dataset**
* **GPU-accelerated training**

---

## License

This project is licensed under the MIT License.
