# Artistic-Vision-Neural-Style-Transfer-for-Image-Synthesis

This project implements a Controllable Neural Style Transfer (NST) system that extends the original Gatys et al. method. It introduces two novel features — Style Interpolation and Color Preservation — allowing users to blend multiple artistic styles while maintaining the original content colors.

---

## Overview

Traditional Neural Style Transfer applies a single style image to a content image, often overwriting the original color palette and offering no user control.  
This project addresses these limitations through:

- **Style Interpolation** — Blend between two distinct styles using an interpolation weight `ω` (e.g., 30% Style A + 70% Style B).  
- **Color Preservation** — Retain the original content colors by applying a color preservation loss in the YUV color space.  

The result is a controllable, high-quality stylization system ideal for digital art generation and creative design.

---

## Features

- Blend two artistic styles using a single parameter `ω`
- Preserve the original color palette of the content image
- TensorFlow + Keras implementation using VGG-19
- Modular, customizable loss functions
- Colab-friendly — works out-of-the-box

---

## How to Use This Repository

- **`style_transfer.ipynb`** – Main Google Colab notebook implementing the controllable artistic style transfer algorithm.  
- **`method_img.png`** – Diagram explaining the model architecture and loss flow.  
- **`interp_*.png`** – Output images showing interpolation results for different values of ω.  
- **`color_preserved_blend.png`** – Example output demonstrating the effect of color preservation.  
- **`report.tex`** – Full LaTeX report describing the method, experiments, and conclusions.  

To reproduce the results, open the notebook in Google Colab and run all cells (as described in the Usage section below).


