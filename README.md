# ASD Data Synthesis using GANs: A Personal Project

Hey there! 👋 This project represents my exploration into generating synthetic fMRI data for Autism Spectrum Disorder (ASD) research using Generative Adversarial Networks (GANs).  I've always been fascinated by the potential of AI to aid in medical research, and this project combines that fascination with my interest in ASD.

This isn't just another GAN project; it's a journey of learning and discovery. I wanted to push myself beyond simple tutorials and build something that could potentially contribute to a more nuanced understanding of ASD.

## Project Goals

The primary goal is to create a GAN capable of generating realistic fMRI data that resembles the characteristics observed in individuals with ASD.  This synthetic data can then be used to:

* **Augment limited real-world datasets:**  Often, obtaining large, high-quality fMRI datasets for ASD research is challenging.  Synthetic data can help overcome this limitation.
* **Test and validate algorithms:**  Generated data can be used to evaluate the robustness and performance of machine learning models designed for ASD diagnosis or biomarker discovery.
* **Explore data distributions:**  Analyzing the generated data can provide insights into the underlying patterns and characteristics of brain activity associated with ASD.

## Technologies Used

* **Python:** The core language for this project.
* **PyTorch:** A powerful deep learning framework for building and training the GAN.
* **NumPy:** For numerical computations and data manipulation.
* **Matplotlib:** For visualizing the results.


## Getting Started

1. **Install dependencies**

Make sure you have PyTorch and the other necessary libraries installed. You can use pip:

```bash
pip install torch numpy matplotlib
```

2. **Prepare your data**

Replace the placeholder fmri_data.npy in the data/ directory with your preprocessed fMRI data. The data should be a NumPy array where each row represents a feature vector of an fMRI image. Ensure your data is properly preprocessed (e.g., normalized).

3. **Train the GAN**

Run the train.py script. You can adjust hyperparameters in the script as needed.

```bash
python train.py
```

## My Future Development

Future work will include:

- Implementing more advanced GAN architectures (e.g., Wasserstein GAN, StyleGAN).
- Developing more robust evaluation metrics.
- Exploring different preprocessing techniques.
- Investigating the use of conditional GANs to control specific aspects of the generated data.

## Contributions
Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to open an issue or submit a pull request.

## Contact
Feel free to reach out if you have any questions or want to discuss this project further.

You can learn more about my coding projects here: [https://sites.google.com/view/wong-kin-on-christopher/computer-science](https://sites.google.com/view/wong-kin-on-christopher/computer-science)
