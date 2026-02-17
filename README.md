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
<!-- * `/src`: Core implementation of the iterative algorithms (MATLAB/Python).
* `/experiments`: Scripts for benchmarking convergence and complexity.
* `/audio_inpainting`: Application-specific code for signal reconstruction.
* `/data`: Sample corrupted audio files and test matrices (if applicable).... 
-->
* `/algorithms`: Core implementation of the iterative algorithms (MATLAB)

## 🚀 Getting Started
### Prerequisites
* MATLAB
* Signal Processing Toolbox

### Running the Benchmarks
... under construction
## 📚 References

1. **Veselý, V., Rajmic, P., & Mokrý, O.** (2024). *Moderní funkcionální analýza s aplikacemi ve zpracování signálů*. VUTIUM. [Link](https://www.luxor.cz/v/2120913/)
<!--2. **Mokrý, O., Magron, P., Oberlin, T., & Févotte, C.** (2022). "Algorithms for audio inpainting based on probabilistic nonnegative matrix factorization." *Signal Processing*, 206, 1-10. [doi:10.1016/j.sigpro.2022.108713](https://doi.org/10.1016/j.sigpro.2022.108713)
3. **Teolis, A.** (1998). *Computational Signal Processing with Wavelets*. Birkhäuser.
-->
