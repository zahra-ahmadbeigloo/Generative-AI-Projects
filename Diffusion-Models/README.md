### Diffusion Model Implementation with Keras built with:

* **Python** for neural network training and logic
* **TensorFlow** & **Keras** for model creation and training
* **NumPy** for numerical operations
* **Matplotlib** for visualizing model loss and performance
* **EarlyStopping** & **ModelCheckpoint** for efficient training
* **tf.data Pipeline** for caching, batching, and prefetching

---

# Diffusion Model for Image Denoising

This project implements a simplified **Diffusion Model** using a convolutional autoencoder architecture. The model learns to reconstruct clean images from noisy inputs, simulating the denoising objective common in diffusion-based generation models.

### **Key Insights:**

* Custom model with `Conv2D`, `Conv2DTranspose`, and `Flatten` layers was trained to remove Gaussian noise.
* Training performance improved using `EarlyStopping` and dynamic layer unfreezing strategy.
* Noise injection and normalization were performed using NumPy and TensorFlow preprocessing.
* Final model was fine-tuned using partial layer unfreezing and adaptive loss monitoring.
* Caching and prefetching optimized data flow during training.

---

## Dependencies

```bash
pip install tensorflow numpy matplotlib
```

---

## License

This project is licensed under the MIT License.
