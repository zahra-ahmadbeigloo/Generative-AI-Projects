## This Project Was Built With:

| Tool/Library    | Version      | Purpose                                 |
|------------------|--------------|-----------------------------------------|
| Python           | 3.10+        | Programming language                    |
| Hugging Face Transformers | 4.36.2 | Model loading and tokenization         |
| PyTorch          | 2.2.2        | Deep learning backend                   |
| FAISS            | -            | Semantic similarity search (Indexing)  |
| scikit-learn     | -            | t-SNE dimensionality reduction         |
| matplotlib       | -            | Embedding visualization                |
| Google Colab     | -            | Development environment                 |

---
# Enhance LLMs with RAG and Hugging Face

This project demonstrates how to improve the quality of responses from a Large Language Model by integrating a Retriever using FAISS and DPR, followed by contextual generation using GPT-2. It compares answer generation with and without context and includes visualization of embedding spaces using t-SNE.

---

## Key Insights

- **RAG improves response accuracy**: When DPR-retrieved contexts are included, GPT-2 generates more grounded and informative responses.
- **t-SNE visualization**: 3D t-SNE helps show how document embeddings are distributed in semantic space.
- **DPR vs no-context**: Without retrieved context, GPT-2 may hallucinate or repeat, while with RAG it gives richer, policy-specific answers.
- **Parameter tuning**: Adjusting `num_beams`, `max_new_tokens`, and `length_penalty` affects the fluency and focus of generated answers.

---

## ⚙Setup & Dependencies

```bash
pip install numpy==1.25.2
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.2 datasets==2.18.0 faiss-cpu wget trl==0.7.9
````

---

## Project Structure & Workflow

### Step 1: Data Loading & Preprocessing

* Downloads `companyPolicies.txt` from a remote URL.
* Splits the document into cleaned paragraphs.

### Step 2: Context Encoding with DPR

* Uses `facebook/dpr-ctx_encoder-single-nq-base` to convert paragraphs to embeddings.
* Visualizes paragraph embeddings using 3D t-SNE.

### Step 3: Indexing with FAISS

* Constructs a flat L2 FAISS index for fast nearest-neighbor search.

### Step 4: Question Embedding

* Embeds user question using `facebook/dpr-question_encoder-single-nq-base`.

### Step 5: Context Retrieval

* Searches FAISS index for top-k most similar paragraphs to the query.

### Step 6: Generation with GPT-2

* Uses `openai-community/gpt2` to generate responses:

  * (A) Without context
  * (B) With retrieved context
* Compares both approaches side by side.

### Step 7: Generation Parameter Tuning

* Explores the impact of `beam width`, `max/min length`, and `length penalty` on final outputs.

---

## License

This project is licensed under the [MIT License](LICENSE).
