# Transformer Research: Last Token Prediction Analysis

This project explores the internal mechanics of a simplified Transformer model, specifically focusing on the manual calculation of gradients and phase-wise training dynamics for the Key-Query (`Wqk`) and Output-Value (`Wov`) matrices.

The goal is to understand how a Transformer performs linear regression-like tasks on a specific "ABC" alternating pattern dataset by mathematically analyzing the gradients of the attention and output layers without relying on automatic differentiation.

## Project Overview

The experiments utilize a custom `SimpleTransformerScratch` model to solve a sequence prediction task. We simplified the problem to:
*   **1 Attention Head**
*   **26-dimensional embeddings** (Identity initialization)
*   **Predicting the last token** in a sequence.

### Key Components

1.  **Dataset (`ABCAlternatingPatternDataset`)**:
    Generates sequences based on a recurring "ABCDE" pattern (e.g., `ABCDEABCDE...`). The model learns to predict the next character in this deterministic sequence.

2.  **Model (`SimpleTransformerScratch`)**:
    A barebones Transformer implementation designed for transparency:
    *   No Multi-Layer Perceptron (MLP).
    *   No LayerNorm (by default).
    *   Single Head Attention.
    *   Identity Embeddings/Unembeddings to map directly to characters.

3.  **Manual Gradients**:
    We derived the exact mathematical formulation for the gradients of the Key-Query matrix ($\nabla W_{qk}$) and the Output-Value matrix ($\nabla W_{ov}$). These are implemented in Python and verified against PyTorch's `autograd`.

## Repository Structure

*   `run_experiment.py`: **Main Verification Script**.
    *   Implements the `SimpleTransformerScratch` model.
    *   Contains `compute_manual_gradients` to calculate gradients from scratch.
    *   Runs a training loop that compares manual gradients vs. PyTorch `autograd` at every step to ensure mathematical correctness.

*   `run_phase_experiment.py`: **Phase-wise Training Experiment**.
    *   Investigates training dynamics by optimizing `Wqk` and `Wov` in alternating phases.
    *   **Phase 1**: Freeze `Wqk`, Train `Wov`.
    *   **Phase 2**: Freeze `Wov`, Train `Wqk`.
    *   Generates a plot `phase_training_accuracy.png` to visualize convergence.

*   `10. simlyfying_for_last_token.ipynb`: **Research Notebook**.
    *   Original exploratory code, mathematical derivations, and prototyping.

## Installation

Ensure you have Python 3.9+ and the following dependencies:

```bash
pip install torch matplotlib numpy
```

## Usage

### 1. Verification Experiment
Run this script to confirm that the manual gradient calculations are correct.

```bash
python run_experiment.py
```
**Expected Output:**
accuracy logs and gradient difference checks.
```text
Batch 0: Wqk Diff: 0.000000, Wov Diff: 0.000000
...
Training complete.
```
*A difference of `0.000000` indicates perfect alignment with PyTorch's automatic differentiation.*

### 2. Phase-wise Experiment
Run this script to observe the model's behavior when training layers in isolation.

```bash
python run_phase_experiment.py
```
**Expected Output:**
Training logs for each phase and a generated plot.
```text
--- Iteration 1 Phase 1 (Train Wov) ---
  OV step 0: Acc=0.0400
...
Plot saved to phase_training_accuracy.png
```

## Results & Insights

*   **Gradient Verification**: The manual derivation for single-head attention gradients is correct, effectively replicating backpropagation.
*   **Phase-wise Training**: Detailed plots show how the model stepwise improves its accuracy, revealing the distinct contributions of the Attention mechanism (finding *where* to look) and the Output mechanism (deciding *what* to write).
