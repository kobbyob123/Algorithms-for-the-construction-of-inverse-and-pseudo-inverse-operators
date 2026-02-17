# Iterative Matrix Inversion for Audio Signal Reconstruction

This repository contains the implementation and experimental analysis for my Master's Thesis: **"Algorithms for the construction of inverse and pseudo-inverse operators with application to the reconstruction of corrupted audio signals."**

The project focuses on replacing traditional, computationally expensive matrix inversion with iterative algorithms to speed up audio inpainting tasks, specifically within the framework of Non-negative Matrix Factorization (NMF).

## 📌 Project Overview
The primary goal is to address the "bottleneck" of large matrix inversions in signal processing. By utilizing iterative constructions of the inverse and pseudo-inverse of linear operators, we aim to:
* Reduce computational complexity.
* Maintain a controlled loss of accuracy.
* Improve the reconstruction speed of corrupted or missing audio sections.

## 🛠 Features
* **Iterative Algorithms:** Implementation of selected iterative methods for (pseudo)inverse construction.
* **Performance Benchmarking:** Comparative analysis of convergence speed and computational time vs. matrix size.
* **Audio Inpainting:** A practical application using NMF-based optimization to fill missing sections in audio signals.
* **Analysis Tools:** Scripts to evaluate the quality of reconstructed signals (SNR, MSE, etc.).

## 📂 Repository Structure
* `/src`: Core implementation of the iterative algorithms (MATLAB/Python).
* `/experiments`: Scripts for benchmarking convergence and complexity.
* `/audio_inpainting`: Application-specific code for signal reconstruction.
* `/data`: Sample corrupted audio files and test matrices (if applicable).
* `/results`: Plots and logs of the experimental findings.

## 🚀 Getting Started
### Prerequisites
* MATLAB (Version R2023b or later recommended) or Python 3.x
* Signal Processing Toolbox

### Running the Benchmarks
...still under construction
