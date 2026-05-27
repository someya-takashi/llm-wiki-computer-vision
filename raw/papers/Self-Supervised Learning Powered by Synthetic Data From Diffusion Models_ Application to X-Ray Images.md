---
title: "Self-Supervised Learning Powered by Synthetic Data From Diffusion Models: Application to X-Ray Images"
source: "https://ieeexplore.ieee.org/abstract/document/10945534"
author:
  - "[[Abdullah Hosseini]]"
  - "[[Ahmed Serag]]"
published:
created: 2026-05-28
description: "Synthetic data offers a compelling solution to the challenges associated with acquiring high-quality medical data, which is often constrained by privacy concern"
tags:
  - "clippings"
---
Overview of the workflow: (A) Training a diffusion model to generate synthetic medical images. (B) Pretraining two self-supervised learning models, one using synthetic da...

## Abstract:

Synthetic data offers a compelling solution to the challenges associated with acquiring high-quality medical data, which is often constrained by privacy concerns and limi...

---

CCBY - IEEE is not the copyright holder of this material. Please follow the instructions via [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/) to obtain full-text articles and stipulations in the API documentation.

Advancments in Artificial Intelligence (AI) and deep learning have revolutionized medical imaging, empowering clinicians with tools for rapid examinations and precise diagnoses. These breakthroughs have amplified the significance of imaging biomarkers, the distinctive patterns in medical images that reveal underlying biological or pathological states. They serve as crucial tools for diagnosing diseases, tracking treatment progress, and deciphering disease mechanisms \[1\], \[2\], \[3\]. These patterns are also the foundation for deep learning models to excel in tasks like disease detection, anatomical segmentation, and condition classification.

While traditional imaging techniques effectively capture these biomarkers, they often face significant limitations, including high costs \[4\], restricted accessibility \[5\], and stringent data privacy regulations that impede dataset sharing \[6\], \[7\], \[8\]. Moreover, for rare or low-incidence diseases, assembling sufficiently large datasets for robust AI model training presents an additional challenge \[9\], \[10\], \[11\], further underscoring the need for innovative solutions.

Various strategies have been proposed to address these challenges. Federated learning \[12\], \[13\], for example, allows decentralized model training across multiple institutions without aggregating sensitive data, preserving patient privacy. Despite its promise, federated learning is not immune to privacy risks, as malicious participants can exploit the training process to infer private information \[14\]. Similarly, data augmentation is a widely adopted method to expand datasets in medical imaging, generating variations of existing samples. However, conventional augmentation techniques often struggle to create semantically meaningful transformations in highly specialized medical contexts \[15\], \[16\].

Synthetic data generation has emerged as a promising alternative, offering the potential to replicate real medical images while preserving critical biomarkers \[17\], \[18\], \[19\], \[20\], \[21\]. By simulating diverse imaging conditions, synthetic datasets can augment training pipelines without risking patient confidentiality. However, generating clinically relevant synthetic images is a complex task. The intricate biological structures and biomarkers present in medical images must be faithfully represented to ensure the resulting data supports accurate AI model development. Without accurate biomarker preservation, synthetic data risks losing clinical relevance, limiting its utility for downstream tasks.

Beyond data-related challenges, clinical workflows demand a holistic interpretation of patient data to ensure effective decision-making \[22\]. While humans excel at contextualizing and synthesizing complex information, most existing machine learning models fall short. These models are typically designed to address narrowly defined tasks, often neglecting the broader clinical context necessary for nuanced medical decision-making. This limitation is particularly pronounced in medical imaging, where the scarcity of annotated datasets makes the development of multiple task-specific models impractical.

To bridge these gaps, Self-supervised learning (SSL) has gained traction in medical AI as an approach that leverages large volumes of unlabeled data to learn robust representations. By capturing latent features across diverse data, SSL models offer the flexibility to generalize across tasks, making them particularly suited for medical imaging scenarios with limited annotated datasets. Synthetic data complements SSL frameworks by providing diverse, biomarker-preserving datasets while eliminating privacy concerns. However, critical questions remain about the generalizability and interpretability of SSL models trained on synthetic data, especially in real-world clinical applications \[23\], \[24\], \[25\].

This study investigates the feasibility of using synthetic data to pretrain SSL models for downstream medical imaging applications. Specifically, we focus on the ability of synthetic data to replicate real-world visual characteristics and preserve essential biomarkers critical for effective AI training. We evaluate these capabilities through classification and segmentation tasks, comparing the performance of models pretrained on synthetic data with those trained on real datasets.

The contributions of this study are threefold: (1) developing a diffusion-based model to generate synthetic radiological images, (2) training high-performance SSL models using synthetic data and benchmarking their performance against real-data-trained models, and (3) experimentally validating that synthetic images retain key biomarkers by demonstrating competitive performance on unseen real-world datasets. The implementation code for our proposed method, *I* - *SynMed*, along with the synthetic dataset used in this study, is publicly available at [https://github.com/serag-ai/I-SynMed](https://github.com/serag-ai/I-SynMed) to support reproducibility and foster further research in medical imaging.

### A. Datasets

Five publicly available datasets were used in this study: two for training and three for evaluation in downstream tasks. The details of these datasets, including their purpose and sample sizes, are summarized in Table 1.

**TABLE 1** Summary of Datasets Used in the Study

[![Table 1- Summary of Datasets Used in the Study](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t1-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t1-3555619-large.gif)

The training datasets comprised the COVIDx CXR-4 \[26\] and NIH Chest X-ray \[27\] datasets, collectively providing approximately 200,000 samples. The COVIDx CXR-4 dataset included 85,000 chest X-ray images, specifically curated for binary classification of COVID-19 cases. The NIH dataset contributed 112,120 labeled images spanning 15 disease categories, with annotations extracted from radiology reports using natural language processing techniques. These datasets were instrumental in training both the diffusion model and the SSL model on real data.

Using the trained diffusion model, a synthetic dataset was generated specifically for SSL pretraining, as in Fig. 1A. This dataset consisted of 100,000 chest X-ray images. The diffusion-based image generation model was meticulously designed to produce high-quality, clinically relevant images that closely replicate the characteristics of real-world medical imaging. This synthetic dataset, intended to be made publicly available, offers a scalable and privacy-preserving solution for advancing research and development in medical AI.

[![FIGURE 1. - Overview of the workflow: (A) Training a diffusion model to generate synthetic medical images. (B) Pretraining two self-supervised learning models, one using synthetic data and the other using real data, with a Vision Transformer (ViT) backbone. (C) Fine-tuning the pretrained models on downstream classification and segmentation tasks, while keeping the backbone frozen.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag1abc-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag1abc-3555619-large.gif)

**FIGURE 1.**

Overview of the workflow: (A) Training a diffusion model to generate synthetic medical images. (B) Pretraining two self-supervised learning models, one using synthetic data and the other using real data, with a Vision Transformer (ViT) backbone. (C) Fine-tuning the pretrained models on downstream classification and segmentation tasks, while keeping the backbone frozen.

For evaluation, the study utilized three additional datasets. The Pneumonia Dataset \[28\] was employed for classification tasks, while the SIIM-ACR Pneumothorax Dataset \[29\] and the Pulmonary Datasets \[30\], \[31\] were used for segmentation tasks. Together, these datasets provided a diverse testbed to assess the generalizability of models trained on real and synthetic data.

### B. Framework Overview: Pretraining and Downstream Task Configuration

Our methodological approach consists of two distinct phases. First, we perform a rigorous pretraining phase for the backbone in a self-supervised manner. Next, the pretrained weights are utilized for downstream tasks, with the backbone weights kept frozen throughout. During pretraining, we employ the DINO \[32\] (Self-Distillation with No Labels) algorithm with a Vision Transformer (ViT) backbone architecture, as illustrated in Fig. 1B. Preprocessing involves normalizing images to ImageNet distribution values and center-cropping each image to a size of $224\times 224$ pixels. To evaluate the efficacy of pretraining, we construct two models: one trained exclusively on real images and the other solely on synthetic images, with both subjected to identical downstream tasks.

For downstream applications, the backbone remains frozen, and additional layers are fine-tuned for classification and segmentation tasks, as shown in Fig. 1C. Classification tasks utilize a simple network consisting of four linear layers, while segmentation tasks are performed using four transposed convolution layers, producing output masks with dimensions (1, 256, 256).

### C. Denoising Diffusion Probabilistic Model

The Denoising Diffusion Probabilistic Model (DDPM) \[33\] stands out as a versatile tool capable of training on datasets to generate novel, high-quality variations of original training images \[34\], \[35\], \[36\]. DDPM operates by transforming a two-dimensional Gaussian noise sample, $X_{t} \sim \mathcal {N}(0,1)$, into a synthetic image, *x*, through a diffusion process (see Fig. 2). This diffusion process entails overlaying a small amount of noise, $\epsilon$, onto the image *x* over *t* timesteps, effectively converting *x* into a pure Gaussian noise sample *T*; this is commonly referred to as forward diffusion. Subsequently, the noise sample *T* can be reverted to its noise-free equivalent, *x*, by removing the overlaid noise, $\epsilon$; the procedure is known as reverse diffusion.

[![FIGURE 2. - Overview of DDPM model used for generating synthetic data. The forward diffusion process gradually introduces noise to an original x-ray image $ X_{0} $
, culminating in a highly noisy image ($ X_{T} $
). The reverse diffusion process iteratively denoises the noisy image, ultimately reconstructing a new, synthetic x-ray image.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag2-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag2-3555619-large.gif)

**FIGURE 2.**

Overview of DDPM model used for generating synthetic data. The forward diffusion process gradually introduces noise to an original x-ray image $X_{0}$, culminating in a highly noisy image ($X_{T}$ ). The reverse diffusion process iteratively denoises the noisy image, ultimately reconstructing a new, synthetic x-ray image.

#### 1) Forward Diffusion

The forward diffusion process gradually overlays small amounts of Gaussian noise, *T*, over successive timesteps to a given source image, $x_{0}$, generating a sequence of noisy images $\{ X_{0}, X_{1}, {\dots }, X_{T} \}$. This process is based on the concept of a Markov process, *q*, wherein the noisy image at timestep *t* depends solely on the noisy image at timestep $t-1$. The distribution can thus be expressed as $q(x_{t} | x_{t-1})$, with $x_{t}$ as a latent variable. As such, the formulation of the forward diffusion process can be expressed as:

$$
\begin{equation*} q(x_{t} \mid x_{t-1}) = \mathcal {N} \left ({{ x_{t}; \mu _{t} = \sqrt {1 - \beta _{t}} \, x_{t-1}, \sigma _{t} = \beta _{t} \, \mathbf {I} }}\right) \tag {1}\end{equation*}
$$
 View Source

Here, $\mathcal {N}$ represents a normal distribution, while $\beta _{t}$ denotes the variances at timestep *t*. The transition from the initial sample $x_{0}$ to the final sample $x_{T}$ in closed form appears as follows:

$$
\begin{equation*} q(x_{1:T} \mid x_{0}) = \prod _{t=1}^{T} q(x_{t} \mid x_{t-1}) \tag {2}\end{equation*}
$$
 View Source

Thanks to the re-parameterization trick introduced in \[33\], computing $x_{t}$ no longer requires the computation of all the samples from previous steps. This is achieved through:

$$
\begin{align*} \alpha _{t} & = 1 - \beta _{t} \tag {3}\\ \bar {\alpha }_{t} & = \prod _{s=0}^{t} \alpha _{s} \tag {4}\\ x_{t} \sim q(x_{t} \mid x_{t-1}, y)& = \mathcal {N} \left ({{ x_{t}; \sqrt {\bar {\alpha }_{t}} \, x_{0}, (1 - \bar {\alpha }_{t}) \, \mathbf {I} }}\right) \tag {5}\end{align*}
$$
 View Source

#### 2) Reverse Diffusion

The reverse diffusion process aims to iteratively denoise $X_{T}$ until it yields a meaningful representation $X_{0}$, achieved by computing the conditional probability $q(X_{t-1} \mid X_{t})$. Given that $q(X_{t-1} \mid X_{t})$ is typically unknown, we approximate it using a parameterized function $\epsilon _{\theta }(x_{t}, t)$, which can be conceptualized as a series of denoising auto-encoders. Employing a Gaussian parameterization and manually removing the predicted Gaussian noise simplifies the process for sampling $X_{t-1}$ as follows:

$$
\begin{equation*} p_{\theta }(x_{t-1} \mid x_{t}, t, y) = \mathcal {N}(x_{t-1}; \mu _{\theta }(x_{t}, t), \sigma _{\theta }(x_{t}, t)) \tag {6}\end{equation*}
$$
 View Source

Subsequently, the reverse process is applied to all time steps:

$$
\begin{equation*} p_{\theta }(x_{0:T}, 0:T) = p_{\theta }(x_{T}, T) \prod _{t=1}^{T} p_{\theta }(x_{t-1} \mid x_{t}, t) \tag {7}\end{equation*}
$$
 View Source

By encapsulating $\epsilon _{\theta }(x_{t}, t)$ as a denoising function trained to predict a denoised version of its input, i.e., $x_{0}$ from $x_{t}$, the objective function can be streamlined, representing the difference between the actual noise introduced during the forward process and the predicted noise, expressed as:

$$
\begin{equation*} L = \mathbb {E}_{x, \epsilon \sim \mathcal {N}(0,1), t} \left [{{ \lVert \epsilon - \epsilon _{\theta }(x_{t}, t) \rVert }}\right ] \tag {8}\end{equation*}
$$
 View Source

For the function approximator, a UNet architecture was employed. The diffusion process was configured with 1,000 timesteps, and the batch size was set to 32.

### D. Self-Supervised Pretraining

Our approach leverages the DINO framework to self-supervise the training of a ViT. This method utilizes a teacher-student paradigm, where both the teacher and student networks are ViT models. The student network is trained by aligning its output distributions with those of the teacher network, using multiple augmented views of the same image. This process enables the student to learn robust and transferable visual representations without requiring labeled data. To ensure stability and prevent feature collapse, a centering mechanism is employed during training. The overall process is illustrated in Fig. 3.

[![FIGURE 3. - Overview of the DINO algorithm, which employs a teacher-student paradigm for pretraining the backbone network in a self-supervised manner, eliminating the need for labeled data. This approach facilitates learning robust feature representations by aligning the outputs of the teacher and student network.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag3-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag3-3555619-large.gif)

**FIGURE 3.**

Overview of the DINO algorithm, which employs a teacher-student paradigm for pretraining the backbone network in a self-supervised manner, eliminating the need for labeled data. This approach facilitates learning robust feature representations by aligning the outputs of the teacher and student network.

Given an input image *x*, we generate a set of *K* augmented views $\{ x_{i} \}_{i=1}^{K}$ using stochastic transformations $T_{i}$:

$$
\begin{equation*} x_{i} = T_{i}(x) \tag {9}\end{equation*}
$$
 View Source

These views consist of $V_{g}$ global crops with high resolution and $V_{l}$ local crops with lower resolution, promoting scale-invariant feature learning and robustness to transformations. Both the student and teacher networks, denoted by $f_{s}$ and $f_{t}$ respectively, map an input image $x_{i}$ to a feature representation. A projection head *h*, implemented as a multi-layer perceptron (MLP), transforms these representations into probability distributions over a feature space:

$$
\begin{align*} z_{i}^{s} & = f_{s}(x_{i}; \theta _{s}), \quad z_{i}^{t} = f_{t}(x_{i}; \theta _{t}) \tag {10}\\ p_{i}^{s} & = h(z_{i}^{s}), \quad p_{i}^{t} = h(z_{i}^{t}) \tag {11}\end{align*}
$$
 View Source

where $\theta _{s}$ and $\theta _{t}$ are the parameters of the student and teacher networks. To avoid trivial solutions where the network outputs collapse to a constant vector, DINO applies centering and sharpening to the teacher’s output. The centered teacher output is:

$$
\begin{equation*} \tilde {p}_{i}^{t} = p_{i}^{t} - c, \quad c \leftarrow m c + (1 - m) \cdot \frac {1}{B} \sum _{i=1}^{B} p_{i}^{t}, \tag {12}\end{equation*}
$$
 View Source

where *c* is a mean vector updated during training with momentum *m* and batch size *B*. Sharpening is also applied by adjusting the temperature $T_{t}$ in the softmax function for the teacher:

$$
\begin{equation*} y_{i}^{t} = \text {softmax}\left ({{ \frac {\tilde {p}_{i}^{t}}{T_{t}} }}\right), \tag {13}\end{equation*}
$$
 View Source

where $T_{t} \lt T_{s}$, and $T_{s}$ is the student’s temperature.

The learning objective is to minimize the cross-entropy loss between the student’s and teacher’s probability distributions over different views:

$$
\begin{equation*} \mathcal {L} = \sum _{i \in {\mathcal {I}}_{s}} \sum _{j \in {\mathcal {I}}_{t}} - y_{j}^{t} \cdot \log y_{i}^{s}, \tag {14}\end{equation*}
$$
 View Source

where ${\mathcal {I}}_{s}$ indexes student views and ${\mathcal {I}}_{t}$ indexes teacher’s global views. The teacher network parameters $\theta _{t}$ are updated using an exponential moving average of the student’s parameters.

To pretrain DINO on synthetic data, we utilized a single A100 GPU, and the main training for the SSL backbone was conducted over 500 epochs. The code for pretraining DINO is heavily based on the open-source project LightlySSL.1

### E. Evaluation Metrics

To assess the quality of the generated images, we employed the Inception Score (IS) \[37\] and the Fréchet Inception Distance (FID) \[38\], which evaluate the extent to which synthetic images align with the data distribution of real images. A higher IS indicates improved image quality and greater diversity among the generated images, while a lower FID reflects superior overall fidelity. Additionally, the Structural Similarity Index Measure (SSIM) was calculated for each synthetic image relative to all other synthetic images to identify the most similar pair. The SSIM values for these closest pairs, referred to as nearest SSIMs, were reported, following the approach outlined in Pan et al. \[20\].

The t-distributed Stochastic Neighbor Embedding (t-SNE) algorithm \[39\] was used to embed images into a three-dimensional feature space, providing compressed representations for real and synthetic images, and to visualize the pretrained model’s ability to learn representations through self-supervised learning.

Classifier performance was meticulously evaluated using multiple metrics, including Area Under the Curve (AUC), precision, recall, accuracy, and F1-score, with cross-entropy loss applied during the training phase. For semantic segmentation, we employed Average Precision (AP), AP50, AP75, and mean Average Precision (mAP) as metrics, and utilized Dice loss to refine the model’s accuracy. This multi-metric evaluation ensures a comprehensive assessment of model performance across various predictive and segmentation capacities.

### F. Statistical Analysis

To test for differences between the results of both models in downstream tasks, t-tests were used for normally distributed data, while the non-parametric Mann-Whitney U test was applied to non-normal distributions (normality was assessed using the Shapiro-Wilk test). p < 0.05 were considered significant after controlling for error using false discovery rate (FDR).

The performance of the synthetic images generated by the DDPM model was evaluated using two complementary approaches. First, a visual analysis was conducted to assess the resemblance of synthetic images to real medical images, providing insights into their visual realism. Second, two SSL models were pre-trained independently on synthetic and real datasets, and their performance was evaluated on unseen real-world data in downstream tasks. This systematic evaluation aimed to determine whether the synthetic medical images effectively preserved critical medical imaging biomarkers required for clinical applications.

### A. Visual and Feature-Based Evaluation of Synthetic and Real Images

Fig. 4A provides a visual comparison between synthetic and real images, highlighting their qualitative similarities. The quantitative evaluation of the synthetic dataset generated using the DDPM model reveals its performance across multiple metrics. The Inception Score was 4.33, indicating the diversity of the generated images. The Mean-ABS was 3.74, reflecting the accuracy in capturing image characteristics. The FID was 12.2, suggesting the degree of similarity between the synthetic and real datasets. Additionally, the nearest-SSIM score of 0.64 highlights a moderate level of structural similarity between the generated and reference images.

[![FIGURE 4. - Quantitative evaluation of synthetic images generated by the DDPM model for SSL pretraining: (A) Visual comparison of synthetic and real chest X-ray images. (B) t-SNE feature space visualization comparing features of real images (blue) and DDPM-generated images (red) for chest X-rays; greater overlap indicates higher similarity in feature representations. (C) Mean pixel values of the synthetic dataset compared to the real dataset. (D) SSIM scores representing the similarity between each synthetic image and its closest real counterpart.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag4abcd-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag4abcd-3555619-large.gif)

**FIGURE 4.**

Quantitative evaluation of synthetic images generated by the DDPM model for SSL pretraining: (A) Visual comparison of synthetic and real chest X-ray images. (B) t-SNE feature space visualization comparing features of real images (blue) and DDPM-generated images (red) for chest X-rays; greater overlap indicates higher similarity in feature representations. (C) Mean pixel values of the synthetic dataset compared to the real dataset. (D) SSIM scores representing the similarity between each synthetic image and its closest real counterpart.

To assess feature distribution similarity, Fig. 4B presents a t-SNE visualization of the feature space, where a greater degree of overlap between synthetic and real samples indicates closer alignment of their feature representations. The box plots in Fig. 4C compare the mean absolute pixel values of real and synthetic datasets, providing a detailed analysis of their pixel level characteristics. Additionally, the diversity of the synthetic dataset, measured using the nearest SSIM, is shown in Fig. 4D.

### B. Feature Space Analysis of Pretrained SSL Models

The pretrained SSL models were evaluated by analyzing their feature spaces on unseen data. Fig. 5 presents the t-SNE visualization of the feature distribution, illustrating the separation of different classes.

[![FIGURE 5. - T-SNE visualization of feature extraction on the pneumonia dataset, including a subset of COVID-19 samples. Features were extracted using pretrained models trained on real and synthetic data. Light green points represent healthy samples, purple points denote COVID-19 samples, and blue points indicate pneumonia samples.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag5-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag5-3555619-large.gif)

**FIGURE 5.**

T-SNE visualization of feature extraction on the pneumonia dataset, including a subset of COVID-19 samples. Features were extracted using pretrained models trained on real and synthetic data. Light green points represent healthy samples, purple points denote COVID-19 samples, and blue points indicate pneumonia samples.

Table 2 summarizes the numerical results for the top-k nearest neighbors on the pneumonia dataset for the binary classification task. The results indicate no statistically significant difference between the models pretrained on real and synthetic data $(p = 0.999)$.

**TABLE 2** Results of Top-1, Top-5, and Top-10 Nearest Neighbors (KNN) for Models Pretrained on Real and Synthetic Data in Binary Classification Tasks for the Pneumonia Dataset. Values are Reported as Percentages

[![Table 2- Results of Top-1, Top-5, and Top-10 Nearest Neighbors (KNN) for Models Pretrained on Real and Synthetic Data in Binary Classification Tasks for the Pneumonia Dataset. Values are Reported as Percentages](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t2-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t2-3555619-large.gif)

Furthermore, we examined the noise resilience of the SSL pretrained model by evaluating its performance in three distinct noisy settings: Gaussian noise, salt-and-pepper noise, and blur, as illustrated in Fig 6.

[![FIGURE 6. - Visualization of the performance of SSL pretrained models on fully real and fully synthetic data for pneumonia classification using top-KNN. From left to right: images with added Gaussian noise, salt-and-pepper noise, and blurred images.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag6ab-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag6ab-3555619-large.gif)

**FIGURE 6.**

Visualization of the performance of SSL pretrained models on fully real and fully synthetic data for pneumonia classification using top-KNN. From left to right: images with added Gaussian noise, salt-and-pepper noise, and blurred images.

### C. Evaluation of Performance in Classification Tasks

A comparison of the the confusion matrices and Area Under the Receiver Operating Characteristic curve (AUROC) values for the two SSL models—one pretrained on real data and the other on synthetic data—is presented in Fig. 7. Specifically, Fig. 7A corresponds to the pneumonia classification task, while Fig. 7B represents the pneumothorax classification task. The comparison of our pneumonia classification results with selected baselines is presented in Table 3. Additionally, comprehensive numerical results for both models across the two tasks are detailed in Table 4.

**TABLE 3** Performance Comparison of the Synthetic Model With Established Baselines for Pneumonia Classification

[![Table 3- Performance Comparison of the Synthetic Model With Established Baselines for Pneumonia Classification](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t3-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t3-3555619-large.gif)

**TABLE 4** Numerical Evaluation in Classification Tasks

[![Table 4- Numerical Evaluation in Classification Tasks](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t4-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t4-3555619-large.gif)

[![FIGURE 7. - Classification results for pneumonia (A) and pneumothorax (B) tasks using backbones pretrained on synthetic and real data. Confusion matrices (top row) and ROC curves (bottom row) are shown for both tasks.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag7ab-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag7ab-3555619-large.gif)

**FIGURE 7.**

Classification results for pneumonia (A) and pneumothorax (B) tasks using backbones pretrained on synthetic and real data. Confusion matrices (top row) and ROC curves (bottom row) are shown for both tasks.

In the pneumonia classification task, the model pretrained on synthetic data slightly outperformed the one pretrained on real data, demonstrating a 0.5% improvement in accuracy, AUC, and F1 score. Conversely, in the pneumothorax classification task, the model pretrained on synthetic data showed marginally better accuracy and F1 scores compared to the real-data-pretrained model. However, a t-test analysis indicated no statistically significant difference, p-value=0.562.

### D. Evaluation of Performance in Segmentation Tasks

Quantitative results for lung segmentation are provided in Table 5, showing no statistically significant difference between the models (p=0.107).

**TABLE 5** Numerical Results of Both Models of Lung Segmentation Task

[![Table 5- Numerical Results of Both Models of Lung Segmentation Task](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t5-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t5-3555619-large.gif)

The performance of the two models on Pneumothorax localization was evaluated using IoU and Dice similarity coefficients. The real model achieved an IoU of 0.13 and a Dice score of 0.25. In comparison, the synthetic model demonstrated slightly better performance, with an IoU of 0.14 and a Dice score of 0.26.

While the model pretrained on synthetic data achieved slightly better results than the model the one pretrained on real data, statistical analysis revealed no significant difference (p=0.127). Fig. 8A presents the quality of predictions for the pneumothorax segmentation task, comparing the models pretrained on real and synthetic data with ground-truth segmentations. Similarly, Fig. 8B illustrates the segmentation results for lung segmentation in chest imaging.

[![FIGURE 8. - Visualization of results for segmentation tasks: (A) Pneumothorax segmentation outputs, displayed from left to right: the original image, ground truth mask, attention classifier map (thresholded at 95%) with confidence score of classifier and segmentation from the synthetic-data-pretrained backbone, and attention classifier map (thresholded at 95%) and segmentation from the real-data-pretrained backbone. (B) Lung segmentation outputs, displayed from left to right: the original image, ground truth annotation, and segmentations from synthetic- and real-data-pretrained backbones.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag8a-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag8a-3555619-large.gif)

**FIGURE 8.**

Visualization of results for segmentation tasks: (A) Pneumothorax segmentation outputs, displayed from left to right: the original image, ground truth mask, attention classifier map (thresholded at 95%) with confidence score of classifier and segmentation from the synthetic-data-pretrained backbone, and attention classifier map (thresholded at 95%) and segmentation from the real-data-pretrained backbone. (B) Lung segmentation outputs, displayed from left to right: the original image, ground truth annotation, and segmentations from synthetic- and real-data-pretrained backbones.

### E. Ablation Study

To optimize performance, we conducted a systematic series of ablation studies, examining various SSL algorithms and dataset configurations. In the first part of the study, we compared multiple SSL methods, including MoCo, and DINO. Among these, DINO consistently demonstrated superior performance in classification tasks. In the second part of the study, we evaluated the effectiveness of synthetic data by training the SSL model on three variations of the dataset. These variations combined synthetic and real data in ratios of 3:1, 1:1, and 1:3, with each scenario utilizing 100,000 image samples.

Regarding SSL algorithms, DINO combined with ViT16 significantly outperformed MoCo paired with ResNet18, achieving faster convergence and higher classification accuracy. As detailed in Table 6, DINO with ViT16 emerged as the best-performing configuration for classification tasks. To identify the optimal SSL algorithm, we employed linear probing on a pneumonia classification task and finalized model selection using pulmonary segmentation results. Table 7 highlights the best-performing configurations for MoCo and DINO in the segmentation task.

**TABLE 6** Pneumonia Classification Performance Metrics for Different Models and Data Types

[![Table 6- Pneumonia Classification Performance Metrics for Different Models and Data Types](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t6-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t6-3555619-large.gif)

**TABLE 7** Pulmonary Segmentation Performance Metrics for Different Models and Data Types

[![Table 7- Pulmonary Segmentation Performance Metrics for Different Models and Data Types](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t7-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag.t7-3555619-large.gif)

To evaluate the utility of synthetic data and demonstrate the generalizability of the models, we trained three SSL models on datasets containing varying proportions of real and synthetic data. Fig 9 illustrates the performance of these models relative to others, revealing no statistically significant differences(*p\_value = 0.977*; after FDR correction). Due to the computational expense of training, all models were trained for 100 epochs, and the results reported are based on this training duration.

[![FIGURE 9. - Performance analysis of DINO ViT across varying dataset compositions. (A) KNN classification performance on the pneumonia detection task. (B) Model performance metrics (F1 score, AUC, and accuracy) following linear probing with frozen backbone parameters for pneumonia classification. (C) Quantitative assessment of chest-lung segmentation efficacy, characterized by IoU and Dice scores.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag9abc-3555619-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/6287639/10820123/10945534/serag9abc-3555619-large.gif)

**FIGURE 9.**

Performance analysis of DINO ViT across varying dataset compositions. (A) KNN classification performance on the pneumonia detection task. (B) Model performance metrics (F1 score, AUC, and accuracy) following linear probing with frozen backbone parameters for pneumonia classification. (C) Quantitative assessment of chest-lung segmentation efficacy, characterized by IoU and Dice scores.

Collecting high-quality, labeled medical data poses significant challenges. Privacy concerns often limit access to sensitive information, while the process of annotating medical imaging data is time-intensive and requires expert knowledge. These issues are further compounded by the need for machine learning models to interpret complex, context-rich medical information to support clinical decision-making. However, most existing models struggle to achieve this level of contextual understanding, limiting their effectiveness in real-world scenarios.

To address these challenges, our study investigated whether synthetic data, generated using state-of-the-art diffusion models, can effectively serve as an alternative to real data for pretraining machine learning models. Specifically, we evaluate the ability of synthetic data to preserve critical medical biomarkers and support SSL models for downstream classification and segmentation tasks. By comparing models pretrained on synthetic data with those pretrained on real data, we assess the feasibility and robustness of this approach.

To evaluate the SSL model’s ability to identify medical biomarkers in chest X-rays, we began with k-nearest neighbors (KNN) classification. Remarkably, the model effectively distinguished between classes such as COVID-19, pneumonia, and healthy chest X-ray, see Fig 5. To further validate our results, we tested the resilience of the SSL-pretrained features under various noise conditions, including Gaussian noise, salt-and-pepper noise, and blur. As shown in Fig 6, performance dropped significantly with Gaussian and salt-and-pepper noise, which disrupt fine details and pixel-level consistency crucial for feature extraction. However, the model exhibited robust tolerance to blur, maintaining a KNN accuracy of around 0.9 even with a kernel size of 25. We believe this robustness stems from the fact that blur was included as one of the augmentation techniques employed by DINO during training.

For classification tasks, our results showed that models pretrained on synthetic data generalize effectively, identifying patterns in unseen real-world cases and achieving classification accuracies exceeding 90%. Notably, there was no statistically significant difference in performance between models pretrained on real versus synthetic data. In addition, we observed that models pretrained on synthetic data successfully converged for pneumonia and pneumothorax classification tasks. Using shallow linear layers and fine-tuning for fewer than ten epochs, these models achieved robust performance.

The synthetic model notably outperformed established baselines in pneumonia classification, achieving an AUC of 99.1 and an accuracy of 95.0%, surpassing Google AutoML Vision \[40\] (AUC 99.1, accuracy 94.6%) and UMAC \[41\] (AUC 96.6, accuracy 90.3%). These results highlight the potential of synthetic data for training robust machine learning models in medical applications.

In segmentation tasks, the synthetic model performed comparably to the real-data-pretrained model for lung segmentation, with no significant difference. The synthetic model achieved higher IoU (0.74 vs. 0.72), Dice (0.85 vs. 0.83), and mAP (0.55 vs. 0.50), while maintaining identical AP50 (0.99) and AP75 (0.48) values. For pneumothorax localization, the synthetic model had a slight edge over the real model in IoU (0.14 vs. 0.13) and mean Dice score (0.26 vs. 0.25), further showcasing its potential in medical imaging tasks.

We hypothesize that the difference in performance between classification and segmentation tasks when using synthetic medical data arises from the distinct nature of these tasks. Classification primarily relies on learning global patterns and high-level representations, which synthetic data variability may enhance. The presence of outliers in synthetic data might augment dataset diversity, enabling SSL models to learn more generalized and robust representations. This robustness could be advantageous for real-world variability in downstream classification tasks. However, the same variability may hinder segmentation tasks, as pixel-level precision and contextual understanding are critical. Outliers or noise in synthetic data could misrepresent local structures, leading to less reliable feature maps and decreased segmentation accuracy.

To optimize performance, we evaluated various SSL algorithms and dataset variations. Among the SSL approaches tested, DINO combined with ViT16s consistently achieved superior results (see Table 6). Furthermore, longer pretraining epochs significantly improved the model’s capacity to learn robust image representations, thereby minimizing the need for extensive fine-tuning during downstream tasks. Regarding dataset variations, the results of models trained for 100 epochs on different combinations of synthetic and real data showed minimal differences. Notably, in the 50/50 split scenario, the model achieved slightly better performance (see Fig 9).

Despite the promising results, our study has certain limitations. The high computational demands associated with synthetic data generation present significant challenges for implementation in resource-constrained environments. Specifically, the DDPM framework requires substantial computational resources, which may not be readily available in many healthcare settings. Moreover, ensuring that synthetic data fully obscures sensitive patient information remains a critical area for evaluation.

Recent advancements, such as the Stabl framework \[43\], have introduced novel methodologies for identifying sparse and reliable biomarkers in omic datasets. These developments highlight the importance of exploring diverse approaches, including imaging and omics, to address the challenges inherent in medical biomarker discovery. While this study primarily focuses on applications within X-ray imaging, the adoption of similar strategies for synthetic data generation and validation has the potential to enhance biomarker discovery across various imaging modalities.

Future research should also investigate the extent to which the DDPM model retains or memorizes original image information during data generation. This is essential to ensure privacy protection and compliance with ethical standards, further reinforcing the utility and reliability of synthetic data in medical applications.

This study demonstrates the potential of synthetic data generated by diffusion models to effectively train machine learning models for medical applications. Models pretrained on synthetic data achieved competitive results in classification and segmentation tasks, highlighting their ability to preserve critical medical biomarkers and generalize to real-world scenarios. These findings also align with the growing development of foundation models, showcasing how synthetic data can support scalable and versatile pretraining frameworks that generalize across multiple clinical tasks.

While the results are promising, challenges remain in optimizing synthetic data generation for practical use in healthcare settings. Future work should prioritize addressing computational efficiency and ensuring the ethical use of synthetic data to maintain patient confidentiality.

### ACKNOWLEDGMENT

The authors gratefully acknowledge the support of the IT and administration teams at Weill Cornell Medicine-Qatar for facilitating the computational and operational aspects of this study. They sincerely thank GPT-4 for its role in improving the language of this report. They wrote the initial draft, and GPT-4 was used to refine and enhance its clarity and quality.