# Generative AI Projects

This repository showcases advanced **Generative AI** projects that combine the power of **transformer architectures**, **diffusion models**, and **LLM pipelines** to generate images, text, and intelligent responses. Each project reflects state-of-the-art methods in modern AI, as explored in the **IBM AI Engineering Professional Certificate** coursework and independent research.

---

## Projects Overview

| Project | Description | Tools & Models |
|--------|-------------|----------------|
| [Aircraft Damage Captioning](./Aircraft-Damage-Captioning) | Classifies aircraft damage and generates natural language captions from images. | VGG16, BLIP, Keras, Hugging Face |
| [Diffusion Models](./Diffusion-Models) | Implements image generation using denoising diffusion probabilistic models. | NumPy, Matplotlib, Forward/Reverse Sampling |
| [Text Generation with Transformers](./Text-Generation) | Generates creative and coherent text using Transformer-based language models. | GPT2, Hugging Face Transformers |
| [Advanced Transformers](./Advanced-Transformers) | Custom Transformer architecture from scratch with positional encoding and masked attention. | TensorFlow/Keras |

---

## Tools & Libraries Used

- **Transformers** (Hugging Face, BLIP, GPT2)
- **TensorFlow/Keras**
- **LangChain** (for RAG)
- **Matplotlib, NumPy, PIL**
- **Diffusers**, **Datasets**, **Scikit-learn**

---

## Key Highlights

- **Image Captioning with BLIP** using pretrained transformer encoders + decoders
- **Diffusion model training loop** from scratch with noise schedules
- **Text generation** with GPT2 (sampling, top-k, and temperature control)
- **Retrieval-Augmented Generation (RAG)** architecture with external context and vector databases *(coming soon)*

---

## Setup

```bash
pip install transformers tensorflow datasets diffusers langchain pillow matplotlib numpy
````

---

## License

This project is licensed under the MIT License.

