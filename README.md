Here's a professional README file for your anonymous repository:

PoRSE: Policy-grounded Synergy of Reward Shaping and Exploration

Anonymous Submission [Paper Under Review]
 Code Release: Post-Acceptance

This repository contains supporting materials for the paper:  
"Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring"

⚠️ Current Repository Status
This interim release contains:
Environment implementations:

isaac/: Modified Isaac Gym robotics environments

bi_dexterity/: Bi-DexHands bimanual manipulation environments

Prompt examples for LLM interactions

Function code samples (reward/mapping functions)

Model architecture prototypes

Full codebase will be released upon paper acceptance per conference submission guidelines.

🛠️ Preliminary Setup
Dependencies (will be finalized in full release):

Python 3.9+

PyTorch 2.0+

NVIDIA GPU with CUDA 11.7
Environment Installation:

For Isaac Gym environments

cd isaac && pip install -e .

For Bi-Dexterity environments

cd bi_dexterity && pip install -e .

📋 Included Resources
Directory Contents

prompts/ LLM prompt templates for reward design and state mapping
examples/ Sample reward functions and affordance space mappings
configs/ Baseline training configurations

🔒 Usage Notes
All materials are provided solely for verification of claims in the submitted paper

The repository will be updated with:

Complete training pipelines

Policy optimization code

LLM interaction modules

Full experiment replicability scripts

📜 Citation
@article{anonymous2024porse,
  title={Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring},
  author={Anonymous},
  journal={Under Review},
  year={2024}

📧 Contact
For inquiries during the review period, please contact the conference submission system.

This repository will be maintained for at least 3 years post-publication.
