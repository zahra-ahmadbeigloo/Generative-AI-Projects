### Image Captioning built with:  
- **Python** for data handling and modeling  
- **TensorFlow/Keras** for transfer learning with **VGG16** and model training  
- **Transformers (BLIP)** for image captioning and summarization  
- **Matplotlib** for visualizations  
- **PIL** & **HuggingFace Transformers** for image decoding and text generation  

---

# Aircraft Damage Detection and Description (VGG16 + BLIP)

This project classifies aircraft images into **damaged or not** using a **transfer learning model (VGG16)**, and then generates **natural language captions and summaries** using a **Transformer model (BLIP)**. It demonstrates a hybrid use of deep learning for both visual recognition and image-to-text generation.

### **Key Insights:**  
- VGG16-based CNN achieved **85% training accuracy** and **69% test accuracy** on the damage classification task.  
- Transfer learning significantly reduced training time while retaining strong feature extraction power.  
- The BLIP model was used to generate high-quality **image captions** and **summaries** such as:  
  > *"this is a detailed photo showing the engine of a Boeing 747."*  
- The BLIP layer was integrated into a custom Keras layer for modular use inside TensorFlow models.  
- Visualization tools confirmed consistent improvement across epochs for both loss and accuracy.

---

## Dependencies

```bash
pip install tensorflow keras transformers matplotlib pillow numpy
```
---

## License
This project is licensed under the MIT License.

