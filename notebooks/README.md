Notebooks: Lens-MAE 
A self-supervised learning pipeline for gravitational lens detection, built as a proof-of-concept for my DeepLense 2026 proposal.

Notebooks
PHASE_1.ipynb — MAE Pre-training
Vision Transformer trained using Masked Autoencoders on CIFAR-100 with an extreme 90% masking ratio. The goal is to force the encoder to learn global structure rather than local pixel patterns, making it robust enough to adapt to sparse astronomical data in the next stage.

PHASE_2.ipynb — Supervised Fine-tuning
The pretrained encoder is loaded and fine-tuned on the DeepLense binary lens-finding dataset using a two-stage approach  frozen encoder first, then full end-to-end training. Handles severe class imbalance (25:1) using class weights. Achieves 91.4% validation accuracy with AUC of 0.97.

PHASE_3.ipynb — Physics-Informed Learning
Extends the model with a multi-task head that simultaneously classifies lens/no-lens and predicts the Einstein radius θ_E. A custom PhysicsPriorLoss penalizes predictions outside the physically realistic range of 0.5–3.0 arcseconds, ensuring the model produces scientifically valid outputs alongside high classification accuracy.

Stack
Python · TensorFlow · Keras · NumPy · Scikit-learn · Kaggle T4 GPU
Author
Sabiha Patel · github.com/psiba15 · Deeplense 2026
