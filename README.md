# Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring

**Anonymous Submission** | Under Review as a **Poster at ICLR 2026**  
**Code Release**: Post-Acceptance

---

## 📋 Overview

This repository contains the supporting materials for the paper:  
**"Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring"**.

The full codebase will be released upon paper acceptance, in accordance with conference submission guidelines.

---

## ⚠️ Current Release Status

This is an **interim release**. The following components are included:

*   **Environment Implementations**:
    *   `isaac/` – Modified Isaac Gym robotics environments.
    *   `bi_dexterity/` – Bi-DexHands bimanual manipulation environments.
*   **LLM Interaction Resources**:
    *   Prompt templates for reward design and state mapping.
*   **Code Samples**:
    *   Example reward and mapping functions.
    *   Model architecture prototypes.
*   **Configuration Files**:
    *   Baseline training configurations.

---

## 🛠️ Preliminary Setup

### Dependencies
*(Finalized list will be provided with the full release.)*
*   Python 3.9+
*   PyTorch 2.0+
*   NVIDIA GPU with CUDA 11.7+

### Installation
1.  **For Isaac Gym environments:**
    ```bash
    cd isaac && pip install -e .
    ```

2.  **For Bi-Dexterity environments:**
    ```bash
    cd bi_dexterity && pip install -e .
    ```

---

## 📁 Repository Structure

```
├── prompts/       # LLM prompt templates for reward design and state mapping
├── examples/      # Sample reward functions and affordance space mappings
└── configs/       # Baseline training configurations
```

---

## 📄 Citation
Citation details will be updated upon paper acceptance.

---
*This README was last updated in February 2026.*
