<div align="center">
  <img align="left" width="100" height="100" src="assets/logo.png" alt="">

# SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation

📄 **[arXiv Preprint (2025)]()**  
[Claudia Cuttano](https://scholar.google.it/citations?user=W7lNKNsAAAAJ&hl=en), [Gabriele Trivigno](https://scholar.google.com/citations?user=JXf_iToAAAAJ&hl=en), [Giuseppe Averta](https://scholar.google.it/citations?user=i4rm0tYAAAAJ&hl=en), [Carlo Masone](https://scholar.google.it/citations?user=cM3Iz_4AAAAJ&hl=en)
</div>

Welcome to the official repository for **SANSA**, our paper:  
*"SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation."*

#### 🚀 Code and Trained Models Coming Soon! 🚀


---

## 🌟 Why SANSA?

Beneath SAM2 tracking architecture lies a surprisingly **rich semantic feature space**.
**SANSA unveils this hidden structure** and repurposes SAM2 into a powerful few-shot segmenter.

🎯 First solution to fully leverage SAM2 — no external feature matchers, no prompt engineering, no multi-stage pipelines.
🖱️ Prompt anything — points, boxes, scribbles, or masks.
⚡ 3–5× faster, 4–5× smaller than prior methods.
🏆 State-of-the-art generalization to novel classes in few-shot segmentation benchmarks.




---

## 🎬 SANSA in Action

**Use SANSA to annotate your images — with any prompt, any reference, and no extra manual setup.**



Whether it's **points**, **boxes**, **scribbles**, or **masks**, SANSA supports them all.  
Just provide one or more reference images, and SANSA will segment **semantically similar objects** in your target.  
All of this runs in a single, unified pipeline: no prompt-specific tweaks, no external models, no extra effort.

> 🛠️ Coming soon: try SANSA on your own data!
---
❄️ **Segmentation is coming. With SANSA, no prompt can hide.** 🐺

## ❓ Why Does It Work?
We extract **SAM2 features** from object instances across diverse images and visualize their distribution using the first three **principal components from PCA**.  
While **zero-shot features from SAM2 lack clear semantic structure**, after adapting features with **SANSA**, we observe the emergence of **well-defined semantic clusters**: **semantically similar instances** group together, forming **coherent clusters** despite strong **intra-class variation in visual appearance**.

<p align="center">
  <img src="assets/pca_3D.png" alt="PCA Semantic Clusters" width="70%">
</p>

⚠️ **These are *not* training classes: SANSA learns from base categories but generalizes to unseen ones by reorganizing the feature space for semantic alignment.**


---
## 🔜 Code & Models

Code and pretrained models will be released soon.  
Stay tuned for updates!

---


