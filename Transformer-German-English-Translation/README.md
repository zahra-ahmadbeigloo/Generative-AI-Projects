### Transformer German-English Translation built with:

* **Python**, **PyTorch**, and **TorchText** for data processing and model building
* **nn.Transformer** for custom sequence-to-sequence translation
* **Multi30k dataset** for German-to-English language translation
* **Custom Transformer architecture** including positional encoding and token embeddings
* **BLEU Score** for evaluation using NLTK
* **Matplotlib** for loss visualization
* **FPDF & pdfplumber** for translating text in PDF files
* **Model saving and loading** with `torch.save()` and `.load_state_dict()`

---

# Custom Transformer for German-to-English Translation

This project implements a full pipeline for **training, evaluating, and deploying** a transformer-based sequence-to-sequence model for German-to-English translation using PyTorch’s native `nn.Transformer`.

The model is trained from scratch using the **Multi30k dataset**, and supports both **interactive translation** and **document translation (PDF)** using a greedy decoding approach.

---

## Workflow Summary

### Part 1: Data Preparation

* Dataset: `torchtext.datasets.Multi30k`
* Tokenization: Custom vocabulary transforms with `<bos>`, `<eos>`, `<pad>`, `<unk>` tokens
* Batch generation with custom `collate_fn` and Torch `DataLoader`

### Part 2: Custom Transformer Model

* Architecture:

  * `TokenEmbedding` for source/target embeddings
  * `PositionalEncoding` to inject positional information
  * `nn.Transformer` for encoder-decoder structure
* Output: Linear projection to target vocabulary
* Masking:

  * Padding masks and causal masks applied in training/inference

### Part 3: Training and Inference

* Training:

  * Loss: `CrossEntropyLoss(ignore_index=PAD_IDX)`
  * Optimizer: `Adam`
  * Epochs: 20
* Inference:

  * Greedy decoding implemented for translation generation
* BLEU Score: Evaluated using NLTK's `sentence_bleu`

### Part 4: PDF Translation

* Translates entire **German-language PDF documents** into English
* Uses `pdfplumber` for reading and `FPDF` for saving translated output

---

## Key Insights

* Successfully trained a custom Transformer model without using pretrained weights
* Demonstrated end-to-end translation from raw text and PDF documents
* Used **greedy decoding** to generate translations with a BLEU score of \~0.38
* Showcased how to deploy a sequence-to-sequence model for **real-world document translation**

---

## Dependencies

```bash
pip install torch torchtext nltk matplotlib fpdf pdfplumber
```

---

## Technologies Used

| Task                 | Tools                              |
| -------------------- | ---------------------------------- |
| Data Loading         | TorchText (Multi30k), custom vocab |
| Model Architecture   | nn.Transformer, PositionalEncoding |
| Training             | PyTorch, CrossEntropyLoss, Adam    |
| Inference & Decoding | Greedy decode, custom loop         |
| Evaluation           | NLTK BLEU score                    |
| Document Translation | pdfplumber, FPDF                   |
| Visualization        | Matplotlib                         |
| Model Save/Load      | torch.save(), load\_state\_dict()  |

---

## License

This project is licensed under the MIT License.
