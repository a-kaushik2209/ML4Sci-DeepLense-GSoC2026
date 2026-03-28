# DEEPLENSE8: Physics-Informed Diffusion Models for Gravitational Lensing Simulation

**GSoC 2026 | ML4Sci**

**Aryan Kaushik** · B.Tech IT, MAIT Delhi (GGSIPU) · CGPA 8.5
[Email](mailto:a.kaushik0908@gmail.com) · [LinkedIn](https://www.linkedin.com/in/aryan-2209-kaushik/) · [GitHub](https://github.com/a-kaushik2209)

---

## What This Repository Is

These are my evaluation test submissions for the DEEPLENSE8 project under ML4Sci GSoC 2026.

The three tests build on each other. Test I gives me a strong classifier. Test VII shows that bolting physics onto that classifier at the output level does not work. Test VIII builds a working DDPM baseline. Together, they motivate the architecture I propose: injecting convergence ($\kappa$) information into the diffusion model's internal representations via cross-attention, not constraining its outputs after the fact.

---

## The Problem

Alexander et al. (2021) showed that a ResNet-18 trained on fixed-redshift lensing simulations scores AUC **0.996**. The same model, tested on variable-redshift images with realistic SNR, drops to **0.880**. Domain adaptation helps (ADDA gets you to 0.955, equivariant networks to 0.980), but these are patches. The underlying issue is that the data pipeline generating training images does not encode the lensing equations. A generator that encodes the lens equation $\beta = \theta - \alpha(\theta)$ and the Poisson equation $\nabla^2\psi = 2\kappa$ would produce data that is robust to distribution shifts by construction, not by post-hoc adaptation.

That is what this project aims to build.

---

## Results at a Glance

| Test | Task | Model | Key Metric | Result |
|---|---|---|---|---|
| **Test I** | Multi-class Classification | EfficientNet-B0 | Macro AUC | **0.9948** |
| **Test VII** | Physics-Guided ML (PINN) | EfficientNet-B0 + κ-head | Macro AUC | **0.9947** |
| **Test VIII** | Diffusion Models (DDPM) | U-Net (21.8M params) | FID / Morphological Consistency | **191.85 / 200 of 200** |

> **A note on the Test VIII numbers.** The FID is inflated because InceptionV3 was trained on natural RGB images, not 64×64 single-channel astrophysical data. The 200/200 morphological consistency means the Test I classifier labels all generated images as valid lensing observations, but the class distribution was {no\_sub: 199, sphere: 1}. That tells me the model learned to generate the dominant training class, not that it understands lensing physics. An unconditional DDPM has no mechanism to control class distribution. Fixing that is the whole point of the DEEPLENSE8 proposal.

---

## Repository Structure

```
ML4Sci-DeepLense-GSoC2026/
│
├── README.md
│
├── test-1/                                        
│   ├── test1-multiclass-classification-ff.ipynb    
│
├── test-7/                                         
│   ├── test7-pinn.ipynb                           
│
└── test-8/                                         
    ├── test8-ddpm.ipynb                            
```

---

## Test I: Multi-Class Classification

**Task:** Classify strong gravitational lensing images into 3 classes: no substructure (`no_sub`), subhalo substructure (`sphere`), and vortex substructure (`vort`).

### Why EfficientNet-B0

I picked EfficientNet-B0 over ResNet-18 because its squeeze-and-excitation blocks give channel-wise feature recalibration. That matters here: the morphological differences between a smooth Einstein ring (no\_sub), a ring perturbed by localized mass clumps (sphere), and a ring with vortex-induced asymmetry (vort) are subtle. EfficientNet-B0 also has roughly 5.3M fewer parameters, which helps on a 37,500-image dataset where overfitting is a real concern.

### Training

- **Optimizer:** AdamW
- **Scheduler:** CosineAnnealingLR (T\_max = 15 epochs)
- **Mixed precision:** `torch.cuda.amp` throughout
- **Early stopping:** patience = 4 epochs on validation AUC
- **Split:** 90/10 stratified (SEED=42, fixed across all tests for fair comparison)
- **Augmentation:** flips + 180° rotation, all physically valid since lensing images have no preferred orientation
- **Hardware:** Kaggle T4 GPU (16GB VRAM)

### Results

| Class | AUC | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| no\_sub | 0.9952 | 0.9157 | 0.9992 | 0.9556 | 1250 |
| sphere | 0.9922 | 0.9876 | 0.8952 | 0.9392 | 1250 |
| vortex | 0.9969 | 0.9625 | 0.9648 | 0.9636 | 1250 |
| **Macro** | **0.9948** | 0.9553 | 0.9531 | 0.9528 | 3750 |

**Test accuracy: 95.31%.**

This classifier does double duty. It is the primary evaluation for Test I, and it serves as the independent morphological consistency evaluator for Test VIII: I use it to check whether generated images look like valid gravitational lenses.

---

## Test VII: Physics-Guided ML (PINN)

**Task:** Extend the Test I classifier with a physics-informed architecture that integrates the gravitational lensing equation as a structural constraint.

### Physics Background

The lensing potential $\psi$ satisfies the Poisson equation:

$$\nabla^2\psi = 2\kappa$$

where $\kappa$ is the convergence (the dimensionless projected mass density). Each substructure class *should* have a distinct $\kappa$ signature:

- **no\_sub:** smooth, low-variance $\kappa$ map
- **sphere:** localized high-$\kappa$ peaks from NFW-profile mass concentrations
- **vortex:** ring-like $\kappa$ gradients from the linear vortex density profile

### Architecture

```
Input Image
     │
EfficientNet-B0 Backbone (shared features)
     ├──→ Classification Head → 3-class logits
     └──→ κ-Head → [κ_mean, κ_std, κ_max, κ_skewness]
```

The κ-head is supervised by Laplacian-derived convergence statistics ($\nabla^2 I \approx \kappa$ proxy). Total loss:

```
L_total = CE + 0.5 × PhysicsLoss

PhysicsLoss = MSE(κ_pred, κ_target) + class_consistency_penalty
```

CE and physics loss tracked separately throughout training, because if you don't track them independently you have no idea whether the physics term is actually doing anything.

### Results

| Metric | Test I (Baseline) | Test VII (PINN) | Delta |
|---|---|---|---|
| no\_sub AUC | 0.9952 | 0.9956 | **+0.0004** |
| sphere AUC | 0.9922 | 0.9916 | -0.0006 |
| vortex AUC | 0.9969 | 0.9969 | +0.0000 |
| **Macro AUC** | **0.9948** | **0.9947** | -0.0001 |
| Accuracy | 95.31% | **96.05%** | **+0.74%** |
| Sphere Recall | 89.52% | **90.64%** | **+1.12%** |

**Test accuracy: 96.05%**, a modest improvement over baseline.

### The Finding That Matters

The accuracy bump is not the point. This is:

| Class | κ mean | κ std | κ max | κ skewness |
|---|---|---|---|---|
| no\_sub | -0.0003 | 0.0494 | 0.3538 | -0.5537 |
| sphere | -0.0003 | 0.0501 | 0.3550 | -0.5596 |
| vortex | -0.0003 | 0.0436 | 0.3546 | -0.5636 |

The κ-head collapsed to the dataset mean. It predicted nearly identical statistics for all three classes. no\_sub should have low variance. sphere should have high max values from localized mass peaks. vortex should have distinctive skewness. None of that emerged.

**This is not a hyperparameter problem. It is an architectural one.** By the time a loss function acts on the output, the backbone has already formed features without physical awareness. The gradient signal from the physics loss is too weak to reshape representations that were optimized for classification. The physics must enter the forward pass, not the loss function.

This finding directly determines the architecture I propose for DEEPLENSE8.

---

## Test VIII: DDPM from Scratch

**Task:** Train a DDPM to generate realistic strong gravitational lensing images. Every component implemented from first principles. No diffusion library used.

### Architecture

**U-Net noise predictor $\epsilon_\theta(x_t, t)$:**

```
Noisy image x_t  +  Timestep t
         │                │
    [1, 64, 64]    Sinusoidal PE → MLP → embedding
         │                │
    ┌────▼────────────────▼────────────────┐
    │  Encoder                              │
    │  64ch  → ResBlock × 2                │
    │  ↓ stride conv                        │
    │  128ch → ResBlock × 2                │
    │  ↓ stride conv                        │
    │  256ch → ResBlock × 2 + Attention    │
    │  ↓ stride conv                        │
    │  256ch → ResBlock × 2 + Attention    │  ← 8×8, global structure
    │                                       │
    │  Bottleneck: ResBlock + Attention     │
    │                                       │
    │  Decoder (mirror + skip connections)  │
    └──────────────────────────────────────┘
         │
    Predicted noise ε  [1, 64, 64]
```

### Design Choices

- **Cosine β-schedule** (Nichol & Dhariwal, 2021): preserves meaningful signal up to t ≈ 900, compared to linear schedule which crashes to near-zero SNR by t ≈ 600, wasting 40% of timesteps
- **Min-SNR-γ loss weighting** (γ=5): prevents high-SNR timesteps from dominating gradients. Directly improves fine structural detail in generated lensing arcs
- **EMA** (decay=0.9999): shadow model averaged over training, loaded for all inference
- **DDIM sampler** (50 steps, η=0): deterministic, 20x faster than full T=1000 reverse process

**Training:** 100 epochs, batch size 64, AdamW lr=2e-4, weight\_decay=1e-4, cosine LR schedule, T=1000 diffusion steps, mixed precision on Kaggle T4.

### Results

| Metric | Value | Notes |
|---|---|---|
| FID Score | 191.85 | Inflated by InceptionV3 domain mismatch (see below) |
| Best Training Loss | 0.002331 | Min-SNR weighted MSE |
| SSIM (avg best-match) | 0.3295 | Over 100 real/generated pairs (see below) |
| Morphological Consistency | 200/200 | All classified as valid lensing by Test I model |
| U-Net Parameters | 21.8M | |
| Inference (DDIM 50 steps) | ~2s/batch | On Kaggle T4 |

### Training Loss Curve

```
Epoch   1/100 | Loss: 0.1570
Epoch  10/100 | Loss: 0.0156
Epoch  20/100 | Loss: 0.0105
Epoch  30/100 | Loss: 0.0071
Epoch  50/100 | Loss: 0.0042
Epoch  70/100 | Loss: 0.0029
Epoch 100/100 | Loss: 0.0023  ← best
```

### Honest Interpretation of the Metrics

**FID (191.85):** FID compares InceptionV3 feature distributions between real and generated images. InceptionV3 was pretrained on ImageNet (full-color natural images at 224x224). My images are single-channel, 64x64, astrophysical. The domain mismatch inflates FID regardless of actual generation quality. [Adam et al. (2022)](https://arxiv.org/abs/2211.03812) explicitly avoid FID when evaluating score-based lensing models for exactly this reason.

**SSIM (0.3295):** This is low in absolute terms, but that is expected. I am comparing each generated image to its closest match in a pool of 200 real images. These are different images, not reconstructions. A generative model that produces novel lensing morphologies rather than memorizing the training set should have moderate SSIM against real data. For context, unconditional GANs on astrophysical data typically report best-match SSIM in the 0.25 to 0.45 range.

**Morphological Consistency (200/200):** I passed 200 DDIM-generated images through the Test I classifier. All 200 were classified as valid lensing observations. The class distribution was {no\_sub: 199, sphere: 1}.

I want to be clear about what this shows and what it does not show. It shows the model learned to produce images that look like gravitational lenses: correct ring structure, plausible intensity profiles, reasonable arc geometry. It does *not* show the model understands lensing physics. The training set was unlabeled and predominantly smooth lensing morphology, so the model correctly learned to generate that dominant mode. An unconditional DDPM has no mechanism to produce a controlled distribution across substructure classes. That is exactly the problem the DEEPLENSE8 project aims to solve.

---

## How The Three Tests Connect

**Test I** gives me a high-accuracy classifier (AUC 0.9948) and an independent oracle for evaluating generated images.

**Test VII** shows that output-level physics constraints do not work. The κ-head collapsed to dataset means. The backbone forms its features without physical awareness, and a loss penalty applied after the fact is too weak to change that. The physics has to enter the representations, not constrain the outputs.

**Test VIII** shows the baseline DDPM already generates morphologically valid lensing images, but has no mechanism for class control. It reproduces the dominant training mode because that is all an unconditional model can do.

**The proposed DEEPLENSE8 architecture** fixes this by encoding $\kappa$ information via cross-attention at the U-Net bottleneck during the denoising process. During training, $\kappa$ comes from the Laplacian of the real image. During generation, $\kappa$ comes from one of two strategies:

1. **Class-conditional κ lookup** (simple baseline): precompute mean κ statistics per class from the 37,500 labeled Test I images, look up the desired class at generation time
2. **Learned κ prior** (more flexible): a small auxiliary MLP, supervised by real Laplacian statistics during training, that produces diverse κ embeddings given just a class label

I will start with approach 1 because it has no moving parts. If within-class κ variance matters (likely for the subhalo class), I switch to approach 2.

**Fallback plan:** if κ cross-attention does not produce class-discriminative distributions after 30 epochs, I switch to spatial κ map conditioning (concatenate the full κ map as an input channel). If that also fails, I fall back to classifier-free guidance with class labels, which still produces controllable generation even without explicit physics encoding.

---

## Environment

```
Hardware:  Kaggle T4 GPU (16GB VRAM)
Framework: PyTorch 2.x
Python:    3.10+
```

**Dependencies:**

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
clean-fid          # FID computation (Test VIII)
```

All notebooks are self-contained and run top-to-bottom on Kaggle with datasets attached as input.

---

## Datasets

| Test | Dataset | Images | Classes | Source |
|---|---|---|---|---|
| Test I & VII | Strong lensing classification | 37,500 | no\_sub, sphere, vortex | ML4Sci DeepLense |
| Test VIII | Strong lensing generation | 10,000 | unlabeled | [Google Drive](https://drive.google.com/file/d/1cJyPQzVOzsCZQctNBuHCqxHnOY7v7UiA/view) |

Images are single-channel `.npy` files, 150×150 pixels, min-max normalized to [0, 1].

---

## References

1. Alexander, Gleyzer, McDonough, Toomey, Usai. *Deep Learning the Morphology of Dark Matter Substructure.* ApJ 893, 15 (2020). [arXiv:1909.07346](https://arxiv.org/abs/1909.07346)

2. Alexander, Gleyzer, Parul, Reddy, Toomey, Usai, Von Klar. *Decoding Dark Matter Substructure without Supervision.* (2021). [arXiv:2008.12731](https://arxiv.org/abs/2008.12731)

3. Alexander, Gleyzer, Reddy, Tidball, Toomey. *Domain Adaptation for Simulation-Based Dark Matter Searches.* (2021). [arXiv:2112.12121](https://arxiv.org/abs/2112.12121)

4. Adam, Coogan, Malkin, Legin, Perreault-Levasseur, Hezaveh, Bengio. *Posterior Samples of Source Galaxies in Strong Gravitational Lenses with Score-Based Priors.* NeurIPS Workshop (2022). [arXiv:2211.03812](https://arxiv.org/abs/2211.03812)

5. Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS (2020).

6. Nichol, Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML (2021).

7. Song, Meng, Ermon. *Denoising Diffusion Implicit Models.* ICLR (2021).

---

*Submitted as part of GSoC 2026 application to ML4Sci. Project: DEEPLENSE8, Physics-Informed Diffusion Models for Gravitational Lensing Simulation.*
