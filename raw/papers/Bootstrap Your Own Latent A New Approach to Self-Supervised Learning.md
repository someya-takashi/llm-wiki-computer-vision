---
title: "Bootstrap Your Own Latent A New Approach to Self-Supervised Learning"
source: "https://ar5iv.labs.arxiv.org/html/2006.07733"
author:
published:
created: 2026-05-27
description: "We introduce Bootstrap Your Own Latent (BYOL), a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks, that interact and learn fro…"
tags:
  - "clippings"
---
Jean-Bastien Grill <sup>,1</sup>   Florian Strub <sup>1</sup> <sup>,1</sup>   Florent Altché <sup>1</sup> <sup>,1</sup>   Corentin Tallec <sup>1</sup> <sup>,1</sup>   Pierre H. Richemond <sup>1</sup> <sup>,1,2</sup> Elena Buchatskaya <sup>1</sup>   Carl Doersch <sup>1</sup>   Bernardo Avila Pires <sup>1</sup>   Zhaohan Daniel Guo <sup>1</sup> Mohammad Gheshlaghi Azar <sup>1</sup>   Bilal Piot <sup>1</sup>   Koray Kavukcuoglu <sup>1</sup>   Rémi Munos <sup>1</sup>   Michal Valko <sup>1</sup>  
<sup>1</sup> DeepMind <sup>2</sup> Imperial College  
\[jbgrill,fstrub,altche,corentint,richemond\]@google.com Equal contribution; the order of first authors was randomly selected.

###### Abstract

We introduce Bootstrap Your Own Latent (BYOL), a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network. While state-of-the art methods rely on negative pairs, BYOL achieves a new state of the art without them. BYOL reaches $74.3\%$ top-1 classification accuracy on ImageNet using a linear evaluation with a ResNet-50 architecture and $79.6\%$ with a larger ResNet. We show that BYOL performs on par or better than the current state of the art on both transfer and semi-supervised benchmarks. Our implementation and pretrained models are given on GitHub.<sup>3</sup>

## 1 Introduction

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2006.07733/assets/x1.png)

Figure 1: Performance of BYOL on ImageNet (linear evaluation) using ResNet-50 and our best architecture ResNet-200 ( 2 × 2\\times ), compared to other unsupervised and supervised ( Sup. ) baselines 8.

Learning good image representations is a key challenge in computer vision [^1] [^2] [^3] as it allows for efficient training on downstream tasks [^4] [^5] [^6] [^7]. Many different training approaches have been proposed to learn such representations, usually relying on visual pretext tasks. Among them, state-of-the-art contrastive methods [^8] [^9] [^10] [^11] [^12] are trained by reducing the distance between representations of different augmented views of the same image (‘positive pairs’), and increasing the distance between representations of augmented views from different images (‘negative pairs’). These methods need careful treatment of negative pairs [^13] by either relying on large batch sizes [^8] [^12], memory banks [^9] or customized mining strategies [^14] [^15] to retrieve the negative pairs. In addition, their performance critically depends on the choice of image augmentations [^8] [^12].

In this paper, we introduce Bootstrap Your Own Latent (BYOL), a new algorithm for self-supervised learning of image representations. BYOL achieves higher performance than state-of-the-art contrastive methods without using negative pairs. It iteratively bootstraps <sup>4</sup> the outputs of a network to serve as targets for an enhanced representation. Moreover, BYOL is more robust to the choice of image augmentations than contrastive methods; we suspect that not relying on negative pairs is one of the leading reasons for its improved robustness. While previous methods based on bootstrapping have used pseudo-labels [^16], cluster indices [^17] or a handful of labels [^18] [^19] [^20], we propose to directly bootstrap the representations. In particular, BYOL uses two neural networks, referred to as online and target networks, that interact and learn from each other. Starting from an augmented view of an image, BYOL trains its online network to predict the target network’s representation of another augmented view of the same image. While this objective admits collapsed solutions, e.g., outputting the same vector for all images, we empirically show that BYOL does not converge to such solutions. We hypothesize (see Section 3.2) that the combination of (i) the addition of a predictor to the online network and (ii) the use of a slow-moving average of the online parameters as the target network encourages encoding more and more information within the online projection and avoids collapsed solutions.

We evaluate the representation learned by BYOL on ImageNet [^21] and other vision benchmarks using ResNet architectures [^22]. Under the linear evaluation protocol on ImageNet, consisting in training a linear classifier on top of the frozen representation, BYOL reaches $74.3\%$ top-1 accuracy with a standard ResNet- $50$ and $79.6\%$ top-1 accuracy with a larger ResNet (Figure 1). In the semi-supervised and transfer settings on ImageNet, we obtain results on par or superior to the current state of the art. Our contributions are: (i) We introduce BYOL, a self-supervised representation learning method (Section 3) which achieves state-of-the-art results under the linear evaluation protocol on ImageNet without using negative pairs. (ii) We show that our learned representation outperforms the state of the art on semi-supervised and transfer benchmarks (Section 4). (iii) We show that BYOL is more resilient to changes in the batch size and in the set of image augmentations compared to its contrastive counterparts (Section 5). In particular, BYOL suffers a much smaller performance drop than SimCLR, a strong contrastive baseline, when only using random crops as image augmentations.

## 2 Related work

Most unsupervised methods for representation learning can be categorized as either generative or discriminative [^23] [^8]. Generative approaches to representation learning build a distribution over data and latent embedding and use the learned embeddings as image representations. Many of these approaches rely either on auto-encoding of images [^24] [^25] [^26] or on adversarial learning [^27], jointly modelling data and representation [^28] [^29] [^30] [^31]. Generative methods typically operate directly in pixel space. This however is computationally expensive, and the high level of detail required for image generation may not be necessary for representation learning.

Among discriminative methods, contrastive methods [^9] [^10] [^32] [^33] [^34] [^11] [^35] [^36] currently achieve state-of-the-art performance in self-supervised learning [^37] [^8] [^38] [^12]. Contrastive approaches avoid a costly generation step in pixel space by bringing representation of different views of the same image closer (‘positive pairs’), and spreading representations of views from different images (‘negative pairs’) apart [^39] [^40]. Contrastive methods often require comparing each example with many other examples to work well [^9] [^8] prompting the question of whether using negative pairs is necessary.

DeepCluster [^17] partially answers this question. It uses bootstrapping on previous versions of its representation to produce targets for the next representation; it clusters data points using the prior representation, and uses the cluster index of each sample as a classification target for the new representation. While avoiding the use of negative pairs, this requires a costly clustering phase and specific precautions to avoid collapsing to trivial solutions.

Some self-supervised methods are not contrastive but rely on using auxiliary handcrafted prediction tasks to learn their representation. In particular, relative patch prediction [^23] [^40], colorizing gray-scale images [^41] [^42], image inpainting [^43], image jigsaw puzzle [^44], image super-resolution [^45], and geometric transformations [^46] [^47] have been shown to be useful. Yet, even with suitable architectures [^48], these methods are being outperformed by contrastive methods [^37] [^8] [^12].

Our approach has some similarities with Predictions of Bootstrapped Latents (PBL, [^49]), a self-supervised representation learning technique for reinforcement learning (RL). PBL jointly trains the agent’s history representation and an encoding of future observations. The observation encoding is used as a target to train the agent’s representation, and the agent’s representation as a target to train the observation encoding. Unlike PBL, BYOL uses a slow-moving average of its representation to provide its targets, and does not require a second network.

The idea of using a slow-moving average target network to produce stable targets for the online network was inspired by deep RL [^50] [^51] [^52] [^53]. Target networks stabilize the bootstrapping updates provided by the Bellman equation, making them appealing to stabilize the bootstrap mechanism in BYOL. While most RL methods use fixed target networks, BYOL uses a weighted moving average of previous networks (as in [^54]) in order to provide smoother changes in the target representation.

In the semi-supervised setting [^55] [^56], an unsupervised loss is combined with a classification loss over a handful of labels to ground the training [^19] [^20] [^57] [^58] [^59] [^60] [^61] [^62]. Among these methods, mean teacher (MT) [^20] also uses a slow-moving average network, called teacher, to produce targets for an online network, called student. An $\ell_{2}$ consistency loss between the softmax predictions of the teacher and the student is added to the classification loss. While [^20] demonstrates the effectiveness of MT in the semi-supervised learning case, in Section 5 we show that a similar approach collapses when removing the classification loss. In contrast, BYOL introduces an additional predictor on top of the online network, which prevents collapse.

Finally, in self-supervised learning, MoCo [^9] uses a slow-moving average network (*momentum encoder*) to maintain consistent representations of negative pairs drawn from a memory bank. Instead, BYOL uses a moving average network to produce prediction targets as a means of stabilizing the bootstrap step. We show in Section 5 that this mere stabilizing effect can also improve existing contrastive methods.

## 3 Method

We start by motivating our method before explaining its details in Section 3.1. Many successful self-supervised learning approaches build upon the cross-view prediction framework introduced in [^63]. Typically, these approaches learn representations by predicting different views (e.g., different random crops) of the same image from one another. Many such approaches cast the prediction problem directly in representation space: the representation of an augmented view of an image should be predictive of the representation of another augmented view of the same image. However, predicting directly in representation space can lead to collapsed representations: for instance, a representation that is constant across views is always fully predictive of itself. Contrastive methods circumvent this problem by reformulating the prediction problem into one of discrimination: from the representation of an augmented view, they learn to discriminate between the representation of another augmented view of the same image, and the representations of augmented views of different images. In the vast majority of cases, this prevents the training from finding collapsed representations. Yet, this discriminative approach typically requires comparing each representation of an augmented view with many negative examples, to find ones sufficiently close to make the discrimination task challenging. In this work, we thus tasked ourselves to find out whether these negative examples are indispensable to prevent collapsing while preserving high performance.

To prevent collapse, a straightforward solution is to use a fixed randomly initialized network to produce the targets for our predictions. While avoiding collapse, it empirically does not result in very good representations. Nonetheless, it is interesting to note that the representation obtained using this procedure can already be much better than the initial fixed representation. In our ablation study (Section 5), we apply this procedure by predicting a fixed randomly initialized network and achieve $18.8\%$ top-1 accuracy (Table 5(a)) on the linear evaluation protocol on ImageNet, whereas the randomly initialized network only achieves $1.4\%$ by itself. This experimental finding is the core motivation for BYOL: from a given representation, referred to as target, we can train a new, potentially enhanced representation, referred to as online, by predicting the target representation. From there, we can expect to build a sequence of representations of increasing quality by iterating this procedure, using subsequent online networks as new target networks for further training. In practice, BYOL generalizes this bootstrapping procedure by iteratively refining its representation, but using a slowly moving exponential average of the online network as the target network instead of fixed checkpoints.

### 3.1 Description of BYOL

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2006.07733/assets/x2.png)

Figure 2: BYOL ’s architecture. minimizes a similarity loss between q θ ( z ) subscript 𝑞 𝜃 𝑧 q\_{\\theta}(z\_{\\theta}) and sg ξ ′ superscript 𝜉 \\mathrm{sg}(z^{\\prime}\_{\\xi}), where {\\theta} are the trained weights, \\xi are an exponential moving average of \\mathrm{sg} means stop-gradient. At the end of training, everything but f 𝑓 f\_{\\theta} is discarded, and y 𝑦 y\_{\\theta} is used as the image representation.

BYOL’s goal is to learn a representation $y_{\theta}$ which can then be used for downstream tasks. As described previously, BYOL uses two neural networks to learn: the *online* and *target* networks. The online network is defined by a set of weights ${\theta}$ and is comprised of three stages: an *encoder* $f_{\theta}$, a *projector* $g_{\theta}$ and a *predictor* $q_{\theta}$, as shown in Figure 2 and Figure 8. The target network has the same architecture as the online network, but uses a different set of weights $\xi$. The target network provides the regression targets to train the online network, and its parameters $\xi$ are an exponential moving average of the online parameters ${\theta}$ [^54]. More precisely, given a target decay rate $\tau\in[0,1]$, after each training step we perform the following update,

$$
\xi\leftarrow\tau\xi+(1-\tau){\theta}.
$$

Given a set of images $\mathcal{D}$, an image $x\sim\mathcal{D}$ sampled uniformly from $\mathcal{D}$, and two distributions of image augmentations $\mathcal{T}$ and $\mathcal{T}^{\prime}$, BYOL produces two augmented views $v\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}t(x)$ and $v^{\prime}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}t^{\prime}(x)$ from $x$ by applying respectively image augmentations $t\sim\mathcal{T}$ and $t^{\prime}\sim\mathcal{T}^{\prime}$. From the first augmented view $v$, the online network outputs a *representation* $y_{\theta}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}f_{\theta}(v)$ and a projection $z_{\theta}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}g_{\theta}(y)$. The target network outputs $y^{\prime}_{\xi}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}f_{\xi}(v^{\prime})$ and the *target projection* $z^{\prime}_{\xi}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}g_{\xi}(y^{\prime})$ from the second augmented view $v^{\prime}$. We then output a *prediction* $q_{\theta}(z_{\theta})$ of $z^{\prime}_{\xi}$ and $\ell_{2}$ -normalize both $q_{\theta}(z_{\theta})$ and $z^{\prime}_{\xi}$ to $\overline{q_{\theta}}(z_{\theta})\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}q_{\theta}(z_{\theta})/\mathopen{}\mathclose{{}\left\|q_{\theta}(z_{\theta})}\right\|_{2}$ and $\overline{z}^{\prime}_{\xi}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}z^{\prime}_{\xi}/\|z^{\prime}_{\xi}\|_{2}$. Note that this predictor is only applied to the online branch, making the architecture asymmetric between the online and target pipeline. Finally we define the following mean squared error between the normalized predictions and target projections,<sup>5</sup>

$$
\mathcal{L}_{{\theta},\xi}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\mathopen{}\mathclose{{}\left\|\overline{q_{\theta}}(z_{\theta})-\overline{z}^{\prime}_{\xi}}\right\|_{2}^{2}=2-2\cdot\frac{\langle q_{\theta}(z_{\theta}),z^{\prime}_{\xi}\rangle}{\big{\|}q_{\theta}(z_{\theta})\big{\|}_{2}\cdot\big{\|}z^{\prime}_{\xi}\big{\|}_{2}}\cdot
$$

We symmetrize the loss $\mathcal{L}_{{\theta},\xi}$ in Equation 2 by separately feeding $v^{\prime}$ to the online network and $v$ to the target network to compute $\widetilde{\mathcal{L}}_{{\theta},\xi}$. At each training step, we perform a stochastic optimization step to minimize $\mathcal{L}^{\texttt{BYOL}}_{{\theta},\xi}=\mathcal{L}_{{\theta},\xi}+\widetilde{\mathcal{L}}_{{\theta},\xi}$ with respect to ${\theta}$ only, but not $\xi$, as depicted by the stop-gradient in Figure 2. BYOL’s dynamics are summarized as

$$
\displaystyle{\theta}\leftarrow\mathrm{optimizer}\mathopen{}\mathclose{{}\left({\theta},\nabla_{\theta}\mathcal{L}^{\texttt{BYOL}}_{{\theta},\xi},\eta}\right),
$$
$$
\displaystyle\xi\leftarrow\tau\xi+(1-\tau){\theta},
$$

where $\mathrm{optimizer}$ is an optimizer and $\eta$ is a learning rate.

At the end of training, we only keep the encoder $f_{\theta}$; as in [^9]. When comparing to other methods, we consider the number of inference-time weights only in the final representation $f_{\theta}$. The full training procedure is summarized in Appendix A, and python pseudo-code based on the libraries JAX [^64] and Haiku [^65] is provided in in Appendix J.

### 3.2 Intuitions on BYOL’s behavior

As BYOL does not use an explicit term to prevent collapse (such as negative examples [^10]) while minimizing $\mathcal{L}_{{\theta},\xi}^{\texttt{BYOL}}$ with respect to ${\theta}$, it may seem that BYOL should converge to a minimum of this loss with respect to $({\theta},\xi)$ (*e.g.*, a collapsed constant representation). However BYOL’s target parameters $\xi$ updates are not in the direction of $\nabla_{\xi}\mathcal{L}_{{\theta},\xi}^{\texttt{BYOL}}$. More generally, we hypothesize that there is no loss $L_{{\theta},\xi}$ such that BYOL’s dynamics is a gradient descent on $L$ jointly over ${\theta},\xi$. This is similar to GANs [^66], where there is no loss that is jointly minimized w.r.t. both the discriminator and generator parameters. There is therefore no a priori reason why BYOL’s parameters would converge to a minimum of $\mathcal{L}^{\texttt{BYOL}}_{{\theta},\xi}$.

While BYOL’s dynamics still admit undesirable equilibria, we did not observe convergence to such equilibria in our experiments. In addition, when assuming BYOL’s predictor to be optimal <sup>6</sup> i.e., $q_{\theta}=q^{\star}$ with

$$
q^{\star}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\operatorname*{arg\,min}_{q}\mathbb{E}\mathopen{}\mathclose{{}\left[\mathopen{}\mathclose{{}\left\|q(z_{\theta})-z^{\prime}_{\xi}}\right\|_{2}^{2}}\right],\quad\text{where}\quad q^{\star}(z_{\theta})=\mathbb{E}\mathopen{}\mathclose{{}\left[z^{\prime}_{\xi}|z_{\theta}}\right],
$$

we hypothesize that the undesirable equilibria are unstable. Indeed, in this optimal predictor case, BYOL’s updates on ${\theta}$ follow in expectation the gradient of the expected conditional variance (see Appendix H for details),

$$
\nabla_{\theta}\mathbb{E}\mathopen{}\mathclose{{}\left[\mathopen{}\mathclose{{}\left\|q^{\star}(z_{\theta})-z^{\prime}_{\xi}}\right\|_{2}^{2}}\right]=\nabla_{\theta}\mathbb{E}\mathopen{}\mathclose{{}\left[\mathopen{}\mathclose{{}\left\|\mathbb{E}\mathopen{}\mathclose{{}\left[z^{\prime}_{\xi}|z_{\theta}}\right]-z^{\prime}_{\xi}}\right\|_{2}^{2}}\right]=\nabla_{\theta}\mathbb{E}\mathopen{}\mathclose{{}\left[\sum_{i}\operatorname*{Var}(z^{\prime}_{\xi,i}|z_{\theta})}\right],
$$

where $z^{\prime}_{\xi,i}$ is the $i$ -th feature of $z^{\prime}_{\xi}$.

Note that for any random variables $X,$ $Y,$ and $Z$, $\operatorname*{Var}(X|Y,Z)\leq\operatorname*{Var}(X|Y)$. Let $X$ be the target projection, $Y$ the current online projection, and $Z$ an additional variability on top of the online projection induced by stochasticities in the training dynamics: purely discarding information from the online projection cannot decrease the conditional variance.

In particular, BYOL avoids constant features in $z_{\theta}$ as, for any constant $c$ and random variables $z_{\theta}$ and $z^{\prime}_{\xi}$, $\operatorname*{Var}(z^{\prime}_{\xi}|z_{\theta})\leq\operatorname*{Var}(z^{\prime}_{\xi}|c)$; hence our hypothesis on these collapsed constant equilibria being unstable. Interestingly, if we were to minimize $\mathbb{E}[\sum_{i}\operatorname*{Var}(z^{\prime}_{\xi,i}|z_{\theta})]$ with respect to $\xi$, we would get a collapsed $z^{\prime}_{\xi}$ as the variance is minimized for a constant $z^{\prime}_{\xi}$. Instead, BYOL makes $\xi$ closer to ${\theta}$, incorporating sources of variability captured by the online projection into the target projection.

Furthemore, notice that performing a hard-copy of the online parameters ${\theta}$ into the target parameters $\xi$ would be enough to propagate new sources of variability. However, sudden changes in the target network might break the assumption of an optimal predictor, in which case BYOL’s loss is not guaranteed to be close to the conditional variance. We hypothesize that the main role of BYOL’s moving-averaged target network is to ensure the near-optimality of the predictor over training; Section 5 and Appendix I provide some empirical support of this interpretation.

### 3.3 Implementation details

#### Image augmentations

BYOL uses the same set of image augmentations as in SimCLR [^8]. First, a random patch of the image is selected and resized to $224\times 224$ with a random horizontal flip, followed by a color distortion, consisting of a random sequence of brightness, contrast, saturation, hue adjustments, and an optional grayscale conversion. Finally Gaussian blur and solarization are applied to the patches. Additional details on the image augmentations are in Appendix B.

#### Architecture

We use a convolutional residual network [^22] with 50 layers and post-activation (ResNet- $50(1\times)$ v1) as our base parametric encoders $f_{\theta}$ and $f_{\xi}$. We also use deeper ($50$, $101$, $152$ and $200$ layers) and wider (from $1\times$ to $4\times$) ResNets, as in [^67] [^48] [^8]. Specifically, the representation $y$ corresponds to the output of the final average pooling layer, which has a feature dimension of $2048$ (for a width multiplier of $1\times$). As in SimCLR [^8], the representation $y$ is projected to a smaller space by a *multi-layer perceptron* (MLP) $g_{\theta}$, and similarly for the target projection $g_{\xi}$. This MLP consists in a linear layer with output size $4096$ followed by batch normalization [^68], rectified linear units (ReLU) [^69], and a final linear layer with output dimension $256$. Contrary to SimCLR, the output of this MLP is not batch normalized. The predictor $q_{\theta}$ uses the same architecture as $g_{\theta}$.

#### Optimization

We use the LARS optimizer [^70] with a cosine decay learning rate schedule [^71], without restarts, over $1000$ epochs, with a warm-up period of $10$ epochs. We set the base learning rate to $0.2,$ scaled linearly [^72] with the batch size ($\text{LearningRate}=0.2\times\text{BatchSize}/256$). In addition, we use a global weight decay parameter of $1.5\cdot 10^{-6}$ while excluding the biases and batch normalization parameters from both LARS adaptation and weight decay. For the target network, the exponential moving average parameter $\tau$ starts from $\tau_{\text{base}}=0.996$ and is increased to one during training. Specifically, we set $\tau\triangleq 1-(1-\tau_{\text{base}})\cdot\mathopen{}\mathclose{{}\left(\cos\mathopen{}\mathclose{{}\left(\pi k/K}\right)+1}\right)/2$ with $k$ the current training step and $K$ the maximum number of training steps. We use a batch size of $4096$ split over $512$ Cloud TPU v $3$ cores. With this setup, training takes approximately $8$ hours for a ResNet- $50(\times 1)$. All hyperparameters are summarized in Appendix J; an additional set of hyperparameters for a smaller batch size of $512$ is provided in Appendix G.

## 4 Experimental evaluation

We assess the performance of BYOL’s representation after self-supervised pretraining on the training set of the ImageNet ILSVRC-2012 dataset [^21]. We first evaluate it on ImageNet (IN) in both linear evaluation and semi-supervised setups. We then measure its transfer capabilities on other datasets and tasks, including classification, segmentation, object detection and depth estimation. For comparison, we also report scores for a representation trained using labels from the train ImageNet subset, referred to as Supervised-IN. In Appendix E, we assess the generality of BYOL by pretraining a representation on the Places365-Standard dataset [^73] before reproducing this evaluation protocol.

#### Linear evaluation on ImageNet

We first evaluate BYOL’s representation by training a linear classifier on top of the frozen representation, following the procedure described in [^48] [^74] [^41] [^10] [^8], and section C.1; we report top- $1$ and top- $5$ accuracies in % on the test set in Table 1. With a standard ResNet- $50$ ($\times 1$) BYOL obtains $74.3\%$ top- $1$ accuracy ($91.6\%$ top- $5$ accuracy), which is a $1.3\%$ (resp. $0.5\%$) improvement over the previous self-supervised state of the art [^12]. This tightens the gap with respect to the supervised baseline of [^8], $76.5\%$, but is still significantly below the stronger supervised baseline of [^75], $78.9\%$. With deeper and wider architectures, BYOL consistently outperforms the previous state of the art (Section C.2), and obtains a best performance of $79.6\%$ top- $1$ accuracy, ranking higher than previous self-supervised approaches. On a ResNet- $50$ ($4\times$) BYOL achieves $78.6\%$, similar to the $78.9\%$ of the best supervised baseline in [^8] for the same architecture.

| Method | Top- $1$ | Top- $5$ |
| --- | --- | --- |
| Local Agg. | $60.2$ | \- |
| PIRL [^35] | $63.6$ | \- |
| CPC v2 [^32] | $63.8$ | $85.3$ |
| CMC [^11] | $66.2$ | $87.0$ |
| SimCLR [^8] | $69.3$ | $89.0$ |
| MoCo v2 [^37] | $71.1$ | \- |
| InfoMin Aug. [^12] | $73.0$ | $91.1$ |
| BYOL (ours) | $\pagecolor{pearDark!20}\bf{74.3}$ | $\pagecolor{pearDark!20}\bf{91.6}$ |

(a) ResNet-50 encoder.

| Method | Architecture | Param. | Top- $1$ | Top- $5$ |
| --- | --- | --- | --- | --- |
| SimCLR [^8] | ResNet- $50$ ($2\times$) | $94$ M | $74.2$ | $92.0$ |
| CMC [^11] | ResNet- $50$ ($2\times$) | $94$ M | $70.6$ | $89.7$ |
| BYOL (ours) | ResNet- $50$ ($2\times$) | $94$ M | $77.4$ | $93.6$ |
| CPC v2 [^32] | ResNet- $161$ | $305$ M | $71.5$ | $90.1$ |
| MoCo [^9] | ResNet- $50$ ($4\times$) | $375$ M | $68.6$ | \- |
| SimCLR [^8] | ResNet- $50$ ($4\times$) | $375$ M | $76.5$ | $93.2$ |
| BYOL (ours) | ResNet- $50$ ($4\times$) | $375$ M | $\pagecolor{pearDark!20}78.6$ | $94.2$ |
| BYOL (ours) | ResNet- $200$ ($2\times$) | $250$ M | $\pagecolor{pearDark!20}\bf{79.6}$ | $\bf{94.8}$ |

(b) Other ResNet encoder architectures.

Table 1: Top-1 and top-5 accuracies (in %) under linear evaluation on ImageNet.

#### Semi-supervised training on ImageNet

Next, we evaluate the performance obtained when fine-tuning BYOL’s representation on a classification task with a small subset of ImageNet’s train set, this time using label information. We follow the semi-supervised protocol of [^74] [^76] [^8] [^32] detailed in Section C.1, and use the same fixed splits of respectively $1\%$ and $10\%$ of ImageNet labeled training data as in [^8]. We report both top- $1$ and top- $5$ accuracies on the test set in Table 2. BYOL consistently outperforms previous approaches across a wide range of architectures. Additionally, as detailed in Section C.1, BYOL reaches $77.7\%$ top- $1$ accuracy with ResNet-50 when fine-tuning over $100\%$ of ImageNet labels.

<table><tbody><tr><th>Method</th><td colspan="2">Top- <math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></td><td colspan="2">Top- <math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></td></tr><tr><th></th><td><math><semantics><mrow><mn>1</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>1</cn></apply> <annotation>1\%</annotation></semantics></math></td><td><math><semantics><mrow><mn>10</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>10</cn></apply> <annotation>10\%</annotation></semantics></math></td><td><math><semantics><mrow><mn>1</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>1</cn></apply> <annotation>1\%</annotation></semantics></math></td><td><math><semantics><mrow><mn>10</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>10</cn></apply> <annotation>10\%</annotation></semantics></math></td></tr><tr><th>Supervised <sup><a href="#fn:77">77</a></sup></th><td><math><semantics><mn>25.4</mn> <cn>25.4</cn> <annotation>25.4</annotation></semantics></math></td><td><math><semantics><mn>56.4</mn> <cn>56.4</cn> <annotation>56.4</annotation></semantics></math></td><td><math><semantics><mn>48.4</mn> <cn>48.4</cn> <annotation>48.4</annotation></semantics></math></td><td><math><semantics><mn>80.4</mn> <cn>80.4</cn> <annotation>80.4</annotation></semantics></math></td></tr><tr><th>InstDisc</th><td>-</td><td>-</td><td><math><semantics><mn>39.2</mn> <cn>39.2</cn> <annotation>39.2</annotation></semantics></math></td><td><math><semantics><mn>77.4</mn> <cn>77.4</cn> <annotation>77.4</annotation></semantics></math></td></tr><tr><th>PIRL <sup><a href="#fn:35">35</a></sup></th><td>-</td><td>-</td><td><math><semantics><mn>57.2</mn> <cn>57.2</cn> <annotation>57.2</annotation></semantics></math></td><td><math><semantics><mn>83.8</mn> <cn>83.8</cn> <annotation>83.8</annotation></semantics></math></td></tr><tr><th>SimCLR <sup><a href="#fn:8">8</a></sup></th><td><math><semantics><mn>48.3</mn> <cn>48.3</cn> <annotation>48.3</annotation></semantics></math></td><td><math><semantics><mn>65.6</mn> <cn>65.6</cn> <annotation>65.6</annotation></semantics></math></td><td><math><semantics><mn>75.5</mn> <cn>75.5</cn> <annotation>75.5</annotation></semantics></math></td><td><math><semantics><mn>87.8</mn> <cn>87.8</cn> <annotation>87.8</annotation></semantics></math></td></tr><tr><th>BYOL (ours)</th><td><math><semantics><mn>53.2</mn> <cn>53.2</cn> <annotation>\bf{53.2}</annotation></semantics></math></td><td><math><semantics><mn>68.8</mn> <cn>68.8</cn> <annotation>\bf{68.8}</annotation></semantics></math></td><td><math><semantics><mn>78.4</mn> <cn>78.4</cn> <annotation>\bf{78.4}</annotation></semantics></math></td><td><math><semantics><mn>89.0</mn> <cn>89.0</cn> <annotation>\bf{89.0}</annotation></semantics></math></td></tr></tbody></table>

(a) ResNet-50 encoder.

<table><tbody><tr><th>Method</th><th>Architecture</th><th>Param.</th><td colspan="2">Top- <math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></td><td colspan="2">Top- <math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></td></tr><tr><th></th><th></th><th></th><td><math><semantics><mrow><mn>1</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>1</cn></apply> <annotation>1\%</annotation></semantics></math></td><td><math><semantics><mrow><mn>10</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>10</cn></apply> <annotation>10\%</annotation></semantics></math></td><td><math><semantics><mrow><mn>1</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>1</cn></apply> <annotation>1\%</annotation></semantics></math></td><td><math><semantics><mrow><mn>10</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>10</cn></apply> <annotation>10\%</annotation></semantics></math></td></tr><tr><th>CPC v2 <sup><a href="#fn:32">32</a></sup></th><th>ResNet- <math><semantics><mn>161</mn> <cn>161</cn> <annotation>161</annotation></semantics></math></th><th><math><semantics><mn>305</mn> <cn>305</cn> <annotation>305</annotation></semantics></math> M</th><td>-</td><td>-</td><td><math><semantics><mn>77.9</mn> <cn>77.9</cn> <annotation>77.9</annotation></semantics></math></td><td><math><semantics><mn>91.2</mn> <cn>91.2</cn> <annotation>91.2</annotation></semantics></math></td></tr><tr><th>SimCLR <sup><a href="#fn:8">8</a></sup></th><th>ResNet- <math><semantics><mn>50</mn> <cn>50</cn> <annotation>50</annotation></semantics></math> (<math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math>)</th><th><math><semantics><mn>94</mn> <cn>94</cn> <annotation>94</annotation></semantics></math> M</th><td><math><semantics><mn>58.5</mn> <cn>58.5</cn> <annotation>58.5</annotation></semantics></math></td><td><math><semantics><mn>71.7</mn> <cn>71.7</cn> <annotation>71.7</annotation></semantics></math></td><td><math><semantics><mn>83.0</mn> <cn>83.0</cn> <annotation>{83.0}</annotation></semantics></math></td><td><math><semantics><mn>91.2</mn> <cn>91.2</cn> <annotation>91.2</annotation></semantics></math></td></tr><tr><th>BYOL (ours)</th><th>ResNet- <math><semantics><mn>50</mn> <cn>50</cn> <annotation>50</annotation></semantics></math> (<math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math>)</th><th><math><semantics><mn>94</mn> <cn>94</cn> <annotation>94</annotation></semantics></math> M</th><td><math><semantics><mn>62.2</mn> <cn>62.2</cn> <annotation>{62.2}</annotation></semantics></math></td><td><math><semantics><mn>73.5</mn> <cn>73.5</cn> <annotation>{73.5}</annotation></semantics></math></td><td><math><semantics><mn>84.1</mn> <cn>84.1</cn> <annotation>{84.1}</annotation></semantics></math></td><td><math><semantics><mn>91.7</mn> <cn>91.7</cn> <annotation>{91.7}</annotation></semantics></math></td></tr><tr><th>SimCLR <sup><a href="#fn:8">8</a></sup></th><th>ResNet- <math><semantics><mn>50</mn> <cn>50</cn> <annotation>50</annotation></semantics></math> (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</th><th><math><semantics><mn>375</mn> <cn>375</cn> <annotation>375</annotation></semantics></math> M</th><td><math><semantics><mn>63.0</mn> <cn>63.0</cn> <annotation>63.0</annotation></semantics></math></td><td><math><semantics><mn>74.4</mn> <cn>74.4</cn> <annotation>74.4</annotation></semantics></math></td><td><math><semantics><mn>85.8</mn> <cn>85.8</cn> <annotation>85.8</annotation></semantics></math></td><td><math><semantics><mn>92.6</mn> <cn>92.6</cn> <annotation>{92.6}</annotation></semantics></math></td></tr><tr><th>BYOL (ours)</th><th>ResNet- <math><semantics><mn>50</mn> <cn>50</cn> <annotation>50</annotation></semantics></math> (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</th><th><math><semantics><mn>375</mn> <cn>375</cn> <annotation>375</annotation></semantics></math> M</th><td><math><semantics><mn>69.1</mn> <cn>69.1</cn> <annotation>{69.1}</annotation></semantics></math></td><td><math><semantics><mn>75.7</mn> <cn>75.7</cn> <annotation>{75.7}</annotation></semantics></math></td><td><math><semantics><mn>87.9</mn> <cn>87.9</cn> <annotation>{87.9}</annotation></semantics></math></td><td><math><semantics><mn>92.5</mn> <cn>92.5</cn> <annotation>92.5</annotation></semantics></math></td></tr><tr><th>BYOL (ours)</th><th>ResNet- <math><semantics><mn>200</mn> <cn>200</cn> <annotation>200</annotation></semantics></math> (<math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math>)</th><th><math><semantics><mn>250</mn> <cn>250</cn> <annotation>250</annotation></semantics></math> M</th><td><math><semantics><mn>71.2</mn> <cn>71.2</cn> <annotation>\bf{71.2}</annotation></semantics></math></td><td><math><semantics><mn>77.7</mn> <cn>77.7</cn> <annotation>\pagecolor{pearDark!20}\bf{77.7}</annotation></semantics></math></td><td><math><semantics><mn>89.5</mn> <cn>89.5</cn> <annotation>\bf{89.5}</annotation></semantics></math></td><td><math><semantics><mn>93.7</mn> <cn>93.7</cn> <annotation>\bf{93.7}</annotation></semantics></math></td></tr></tbody></table>

(b) Other ResNet encoder architectures.

Table 2: Semi-supervised training with a fraction of ImageNet labels.

#### Transfer to other classification tasks

We evaluate our representation on other classification datasets to assess whether the features learned on ImageNet (IN) are generic and thus useful across image domains, or if they are ImageNet-specific. We perform linear evaluation and fine-tuning on the same set of classification tasks used in [^8] [^74], and carefully follow their evaluation protocol, as detailed in Appendix D. Performance is reported using standard metrics for each benchmark, and results are provided on a held-out test set after hyperparameter selection on a validation set. We report results in Table 3, both for linear evaluation and fine-tuning. BYOL outperforms SimCLR on all benchmarks and the Supervised-IN baseline on $7$ of the $12$ benchmarks, providing only slightly worse performance on the $5$ remaining benchmarks. BYOL’s representation can be transferred over to small images, e.g., CIFAR [^78], landscapes, e.g., SUN397 [^79] or VOC2007 [^80], and textures, e.g., DTD [^81].

<table><tbody><tr><th>Method</th><td>Food101</td><td>CIFAR10</td><td>CIFAR100</td><td>Birdsnap</td><td>SUN397</td><td>Cars</td><td>Aircraft</td><td>VOC2007</td><td>DTD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td></tr><tr><th colspan="13"><em>Linear evaluation:</em></th></tr><tr><th>BYOL (ours)</th><td><math><semantics><mn>75.3</mn> <cn>75.3</cn> <annotation>\bf{75.3}</annotation></semantics></math></td><td><math><semantics><mn>91.3</mn> <cn>91.3</cn> <annotation>91.3</annotation></semantics></math></td><td><math><semantics><mn>78.4</mn> <cn>78.4</cn> <annotation>\bf{78.4}</annotation></semantics></math></td><td><math><semantics><mn>57.2</mn> <cn>57.2</cn> <annotation>\bf{57.2}</annotation></semantics></math></td><td><math><semantics><mn>62.2</mn> <cn>62.2</cn> <annotation>\bf{62.2}</annotation></semantics></math></td><td><math><semantics><mn>67.8</mn> <cn>67.8</cn> <annotation>\bf{67.8}</annotation></semantics></math></td><td><math><semantics><mn>60.6</mn> <cn>60.6</cn> <annotation>60.6</annotation></semantics></math></td><td><math><semantics><mn>82.5</mn> <cn>82.5</cn> <annotation>82.5</annotation></semantics></math></td><td><math><semantics><mn>75.5</mn> <cn>75.5</cn> <annotation>75.5</annotation></semantics></math></td><td><math><semantics><mn>90.4</mn> <cn>90.4</cn> <annotation>90.4</annotation></semantics></math></td><td><math><semantics><mn>94.2</mn> <cn>94.2</cn> <annotation>94.2</annotation></semantics></math></td><td><math><semantics><mn>96.1</mn> <cn>96.1</cn> <annotation>\bf{96.1}</annotation></semantics></math></td></tr><tr><th>SimCLR (repro)</th><td><math><semantics><mn>72.8</mn> <cn>72.8</cn> <annotation>72.8</annotation></semantics></math></td><td><math><semantics><mn>90.5</mn> <cn>90.5</cn> <annotation>90.5</annotation></semantics></math></td><td><math><semantics><mn>74.4</mn> <cn>74.4</cn> <annotation>74.4</annotation></semantics></math></td><td><math><semantics><mn>42.4</mn> <cn>42.4</cn> <annotation>42.4</annotation></semantics></math></td><td><math><semantics><mn>60.6</mn> <cn>60.6</cn> <annotation>60.6</annotation></semantics></math></td><td><math><semantics><mn>49.3</mn> <cn>49.3</cn> <annotation>49.3</annotation></semantics></math></td><td><math><semantics><mn>49.8</mn> <cn>49.8</cn> <annotation>49.8</annotation></semantics></math></td><td><math><semantics><mn>81.4</mn> <cn>81.4</cn> <annotation>81.4</annotation></semantics></math></td><td><math><semantics><mn>75.7</mn> <cn>75.7</cn> <annotation>\bf{75.7}</annotation></semantics></math></td><td><math><semantics><mn>84.6</mn> <cn>84.6</cn> <annotation>84.6</annotation></semantics></math></td><td><math><semantics><mn>89.3</mn> <cn>89.3</cn> <annotation>89.3</annotation></semantics></math></td><td><math><semantics><mn>92.6</mn> <cn>92.6</cn> <annotation>92.6</annotation></semantics></math></td></tr><tr><th>SimCLR <sup><a href="#fn:8">8</a></sup></th><td><math><semantics><mn>68.4</mn> <cn>68.4</cn> <annotation>68.4</annotation></semantics></math></td><td><math><semantics><mn>90.6</mn> <cn>90.6</cn> <annotation>90.6</annotation></semantics></math></td><td><math><semantics><mn>71.6</mn> <cn>71.6</cn> <annotation>71.6</annotation></semantics></math></td><td><math><semantics><mn>37.4</mn> <cn>37.4</cn> <annotation>37.4</annotation></semantics></math></td><td><math><semantics><mn>58.8</mn> <cn>58.8</cn> <annotation>58.8</annotation></semantics></math></td><td><math><semantics><mn>50.3</mn> <cn>50.3</cn> <annotation>50.3</annotation></semantics></math></td><td><math><semantics><mn>50.3</mn> <cn>50.3</cn> <annotation>50.3</annotation></semantics></math></td><td><math><semantics><mn>80.5</mn> <cn>80.5</cn> <annotation>80.5</annotation></semantics></math></td><td><math><semantics><mn>74.5</mn> <cn>74.5</cn> <annotation>74.5</annotation></semantics></math></td><td><math><semantics><mn>83.6</mn> <cn>83.6</cn> <annotation>83.6</annotation></semantics></math></td><td><math><semantics><mn>90.3</mn> <cn>90.3</cn> <annotation>90.3</annotation></semantics></math></td><td><math><semantics><mn>91.2</mn> <cn>91.2</cn> <annotation>91.2</annotation></semantics></math></td></tr><tr><th>Supervised-IN <sup><a href="#fn:8">8</a></sup></th><td><math><semantics><mn>72.3</mn> <cn>72.3</cn> <annotation>72.3</annotation></semantics></math></td><td><math><semantics><mn>93.6</mn> <cn>93.6</cn> <annotation>\bf{93.6}</annotation></semantics></math></td><td><math><semantics><mn>78.3</mn> <cn>78.3</cn> <annotation>78.3</annotation></semantics></math></td><td><math><semantics><mn>53.7</mn> <cn>53.7</cn> <annotation>53.7</annotation></semantics></math></td><td><math><semantics><mn>61.9</mn> <cn>61.9</cn> <annotation>61.9</annotation></semantics></math></td><td><math><semantics><mn>66.7</mn> <cn>66.7</cn> <annotation>66.7</annotation></semantics></math></td><td><math><semantics><mn>61.0</mn> <cn>61.0</cn> <annotation>\bf{61.0}</annotation></semantics></math></td><td><math><semantics><mn>82.8</mn> <cn>82.8</cn> <annotation>\bf{82.8}</annotation></semantics></math></td><td><math><semantics><mn>74.9</mn> <cn>74.9</cn> <annotation>74.9</annotation></semantics></math></td><td><math><semantics><mn>91.5</mn> <cn>91.5</cn> <annotation>\bf{91.5}</annotation></semantics></math></td><td><math><semantics><mn>94.5</mn> <cn>94.5</cn> <annotation>\bf{94.5}</annotation></semantics></math></td><td><math><semantics><mn>94.7</mn> <cn>94.7</cn> <annotation>94.7</annotation></semantics></math></td></tr><tr><th colspan="13"><em>Fine-tuned:</em></th></tr><tr><th>BYOL (ours)</th><td><math><semantics><mn>88.5</mn> <cn>88.5</cn> <annotation>\bf{88.5}</annotation></semantics></math></td><td><math><semantics><mn>97.8</mn> <cn>97.8</cn> <annotation>\bf{97.8}</annotation></semantics></math></td><td><math><semantics><mn>86.1</mn> <cn>86.1</cn> <annotation>86.1</annotation></semantics></math></td><td><math><semantics><mn>76.3</mn> <cn>76.3</cn> <annotation>\bf{76.3}</annotation></semantics></math></td><td><math><semantics><mn>63.7</mn> <cn>63.7</cn> <annotation>63.7</annotation></semantics></math></td><td><math><semantics><mn>91.6</mn> <cn>91.6</cn> <annotation>91.6</annotation></semantics></math></td><td><math><semantics><mn>88.1</mn> <cn>88.1</cn> <annotation>\bf{88.1}</annotation></semantics></math></td><td><math><semantics><mn>85.4</mn> <cn>85.4</cn> <annotation>\bf{85.4}</annotation></semantics></math></td><td><math><semantics><mn>76.2</mn> <cn>76.2</cn> <annotation>\bf{76.2}</annotation></semantics></math></td><td><math><semantics><mn>91.7</mn> <cn>91.7</cn> <annotation>91.7</annotation></semantics></math></td><td><math><semantics><mn>93.8</mn> <cn>93.8</cn> <annotation>\bf{93.8}</annotation></semantics></math></td><td><math><semantics><mn>97.0</mn> <cn>97.0</cn> <annotation>97.0</annotation></semantics></math></td></tr><tr><th>SimCLR (repro)</th><td><math><semantics><mn>87.5</mn> <cn>87.5</cn> <annotation>87.5</annotation></semantics></math></td><td><math><semantics><mn>97.4</mn> <cn>97.4</cn> <annotation>97.4</annotation></semantics></math></td><td><math><semantics><mn>85.3</mn> <cn>85.3</cn> <annotation>85.3</annotation></semantics></math></td><td><math><semantics><mn>75.0</mn> <cn>75.0</cn> <annotation>75.0</annotation></semantics></math></td><td><math><semantics><mn>63.9</mn> <cn>63.9</cn> <annotation>63.9</annotation></semantics></math></td><td><math><semantics><mn>91.4</mn> <cn>91.4</cn> <annotation>91.4</annotation></semantics></math></td><td><math><semantics><mn>87.6</mn> <cn>87.6</cn> <annotation>87.6</annotation></semantics></math></td><td><math><semantics><mn>84.5</mn> <cn>84.5</cn> <annotation>84.5</annotation></semantics></math></td><td><math><semantics><mn>75.4</mn> <cn>75.4</cn> <annotation>75.4</annotation></semantics></math></td><td><math><semantics><mn>89.4</mn> <cn>89.4</cn> <annotation>89.4</annotation></semantics></math></td><td><math><semantics><mn>91.7</mn> <cn>91.7</cn> <annotation>91.7</annotation></semantics></math></td><td><math><semantics><mn>96.6</mn> <cn>96.6</cn> <annotation>96.6</annotation></semantics></math></td></tr><tr><th>SimCLR <sup><a href="#fn:8">8</a></sup></th><td><math><semantics><mn>88.2</mn> <cn>88.2</cn> <annotation>88.2</annotation></semantics></math></td><td><math><semantics><mn>97.7</mn> <cn>97.7</cn> <annotation>97.7</annotation></semantics></math></td><td><math><semantics><mn>85.9</mn> <cn>85.9</cn> <annotation>85.9</annotation></semantics></math></td><td><math><semantics><mn>75.9</mn> <cn>75.9</cn> <annotation>75.9</annotation></semantics></math></td><td><math><semantics><mn>63.5</mn> <cn>63.5</cn> <annotation>63.5</annotation></semantics></math></td><td><math><semantics><mn>91.3</mn> <cn>91.3</cn> <annotation>91.3</annotation></semantics></math></td><td><math><semantics><mn>88.1</mn> <cn>88.1</cn> <annotation>88.1</annotation></semantics></math></td><td><math><semantics><mn>84.1</mn> <cn>84.1</cn> <annotation>84.1</annotation></semantics></math></td><td><math><semantics><mn>73.2</mn> <cn>73.2</cn> <annotation>73.2</annotation></semantics></math></td><td><math><semantics><mn>89.2</mn> <cn>89.2</cn> <annotation>89.2</annotation></semantics></math></td><td><math><semantics><mn>92.1</mn> <cn>92.1</cn> <annotation>92.1</annotation></semantics></math></td><td><math><semantics><mn>97.0</mn> <cn>97.0</cn> <annotation>97.0</annotation></semantics></math></td></tr><tr><th>Supervised-IN <sup><a href="#fn:8">8</a></sup></th><td><math><semantics><mn>88.3</mn> <cn>88.3</cn> <annotation>88.3</annotation></semantics></math></td><td><math><semantics><mn>97.5</mn> <cn>97.5</cn> <annotation>97.5</annotation></semantics></math></td><td><math><semantics><mn>86.4</mn> <cn>86.4</cn> <annotation>\bf{86.4}</annotation></semantics></math></td><td><math><semantics><mn>75.8</mn> <cn>75.8</cn> <annotation>75.8</annotation></semantics></math></td><td><math><semantics><mn>64.3</mn> <cn>64.3</cn> <annotation>\bf{64.3}</annotation></semantics></math></td><td><math><semantics><mn>92.1</mn> <cn>92.1</cn> <annotation>\bf{92.1}</annotation></semantics></math></td><td><math><semantics><mn>86.0</mn> <cn>86.0</cn> <annotation>86.0</annotation></semantics></math></td><td><math><semantics><mn>85.0</mn> <cn>85.0</cn> <annotation>85.0</annotation></semantics></math></td><td><math><semantics><mn>74.6</mn> <cn>74.6</cn> <annotation>74.6</annotation></semantics></math></td><td><math><semantics><mn>92.1</mn> <cn>92.1</cn> <annotation>\bf{92.1}</annotation></semantics></math></td><td><math><semantics><mn>93.3</mn> <cn>93.3</cn> <annotation>93.3</annotation></semantics></math></td><td><math><semantics><mn>97.6</mn> <cn>97.6</cn> <annotation>\bf{97.6}</annotation></semantics></math></td></tr><tr><th>Random init <sup><a href="#fn:8">8</a></sup></th><td><math><semantics><mn>86.9</mn> <cn>86.9</cn> <annotation>86.9</annotation></semantics></math></td><td><math><semantics><mn>95.9</mn> <cn>95.9</cn> <annotation>95.9</annotation></semantics></math></td><td><math><semantics><mn>80.2</mn> <cn>80.2</cn> <annotation>80.2</annotation></semantics></math></td><td><math><semantics><mn>76.1</mn> <cn>76.1</cn> <annotation>76.1</annotation></semantics></math></td><td><math><semantics><mn>53.6</mn> <cn>53.6</cn> <annotation>53.6</annotation></semantics></math></td><td><math><semantics><mn>91.4</mn> <cn>91.4</cn> <annotation>91.4</annotation></semantics></math></td><td><math><semantics><mn>85.9</mn> <cn>85.9</cn> <annotation>85.9</annotation></semantics></math></td><td><math><semantics><mn>67.3</mn> <cn>67.3</cn> <annotation>67.3</annotation></semantics></math></td><td><math><semantics><mn>64.8</mn> <cn>64.8</cn> <annotation>64.8</annotation></semantics></math></td><td><math><semantics><mn>81.5</mn> <cn>81.5</cn> <annotation>81.5</annotation></semantics></math></td><td><math><semantics><mn>72.6</mn> <cn>72.6</cn> <annotation>72.6</annotation></semantics></math></td><td><math><semantics><mn>92.0</mn> <cn>92.0</cn> <annotation>92.0</annotation></semantics></math></td></tr></tbody></table>

Table 3: Transfer learning results from ImageNet (IN) with the standard ResNet-50 architecture.

#### Transfer to other vision tasks

We evaluate our representation on different tasks relevant to computer vision practitioners, namely semantic segmentation, object detection and depth estimation. With this evaluation, we assess whether BYOL’s representation generalizes beyond classification tasks.

We first evaluate BYOL on the VOC2012 semantic segmentation task as detailed in Section D.4, where the goal is to classify each pixel in the image [^7]. We report the results in Table 4(a). BYOL outperforms both the Supervised-IN baseline ($+1.9$ mIoU) and SimCLR ($+1.1$ mIoU).

Similarly, we evaluate on object detection by reproducing the setup in [^9] using a Faster R-CNN architecture [^82], as detailed in Section D.5. We fine-tune on trainval2007 and report results on test2007 using the standard AP <sub>50</sub> metric; BYOL is significantly better than the Supervised-IN baseline ($+3.1$ AP <sub>50</sub>) and SimCLR ($+2.3$ AP <sub>50</sub>).

Finally, we evaluate on depth estimation on the NYU v2 dataset, where the depth map of a scene is estimated given a single RGB image. Depth prediction measures how well a network represents geometry, and how well that information can be localized to pixel accuracy [^40]. The setup is based on [^83] and detailed in Section D.6. We evaluate on the commonly used test subset of $654$ images and report results using several common metrics in Table 4(b): relative (rel) error, root mean squared (rms) error, and the percent of pixels (pct) where the error, $\max(d_{gt}/d_{p},d_{p}/d_{gt})$, is below $1.25^{n}$ thresholds where $d_{p}$ is the predicted depth and $d_{gt}$ is the ground truth depth [^40]. BYOL is better or on par with other methods for each metric. For instance, the challenging pct. $\!\,\!<\!\!1.25$ measure is respectively improved by $+3.5$ points and $+1.3$ points compared to supervised and SimCLR baselines.

| Method | AP <sub>50</sub> | mIoU |
| --- | --- | --- |
| Supervised-IN [^9] | $74.4$ | $74.4$ |
| MoCo [^9] | $74.9$ | $72.5$ |
| SimCLR (repro) | $75.2$ | $75.2$ |
| BYOL (ours) | $\bf{77.5}$ | $\bf{76.3}$ |

(a) Transfer results in semantic  
segmentation and object detection.

<table><tbody><tr><th></th><td colspan="3">Higher better</td><td colspan="2">Lower better</td></tr><tr><th>Method</th><td>pct. <math><semantics><mrow><mo><</mo> <mn>1.25</mn></mrow> <apply><csymbol>absent</csymbol> <cn>1.25</cn></apply> <annotation>\!<\!1.25</annotation></semantics></math></td><td>pct. <math><semantics><mrow><mo><</mo> <msup><mn>1.25</mn> <mn>2</mn></msup></mrow> <apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>1.25</cn> <cn>2</cn></apply></apply> <annotation>\!<\!1.25^{2}</annotation></semantics></math></td><td>pct. <math><semantics><mrow><mo><</mo> <msup><mn>1.25</mn> <mn>3</mn></msup></mrow> <apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>1.25</cn> <cn>3</cn></apply></apply> <annotation>\!<\!1.25^{3}</annotation></semantics></math></td><td>rms</td><td>rel</td></tr><tr><th>Supervised-IN <sup><a href="#fn:83">83</a></sup></th><td><math><semantics><mn>81.1</mn> <cn>81.1</cn> <annotation>81.1</annotation></semantics></math></td><td><math><semantics><mn>95.3</mn> <cn>95.3</cn> <annotation>95.3</annotation></semantics></math></td><td><math><semantics><mn>98.8</mn> <cn>98.8</cn> <annotation>98.8</annotation></semantics></math></td><td><math><semantics><mn>0.573</mn> <cn>0.573</cn> <annotation>0.573</annotation></semantics></math></td><td><math><semantics><mn>0.127</mn> <cn>0.127</cn> <annotation>\bf{0.127}</annotation></semantics></math></td></tr><tr><th>SimCLR (repro)</th><td><math><semantics><mn>83.3</mn> <cn>83.3</cn> <annotation>83.3</annotation></semantics></math></td><td><math><semantics><mn>96.5</mn> <cn>96.5</cn> <annotation>96.5</annotation></semantics></math></td><td><math><semantics><mn>99.1</mn> <cn>99.1</cn> <annotation>99.1</annotation></semantics></math></td><td><math><semantics><mn>0.557</mn> <cn>0.557</cn> <annotation>0.557</annotation></semantics></math></td><td><math><semantics><mn>0.134</mn> <cn>0.134</cn> <annotation>0.134</annotation></semantics></math></td></tr><tr><th>BYOL (ours)</th><td><math><semantics><mn>84.6</mn> <cn>84.6</cn> <annotation>\bf{84.6}</annotation></semantics></math></td><td><math><semantics><mn>96.7</mn> <cn>96.7</cn> <annotation>\bf{96.7}</annotation></semantics></math></td><td><math><semantics><mn>99.1</mn> <cn>99.1</cn> <annotation>\bf{99.1}</annotation></semantics></math></td><td><math><semantics><mn>0.541</mn> <cn>0.541</cn> <annotation>\bf{0.541}</annotation></semantics></math></td><td><math><semantics><mn>0.129</mn> <cn>0.129</cn> <annotation>0.129</annotation></semantics></math></td></tr></tbody></table>

(b) Transfer results on NYU v2 depth estimation.

Table 4: Results on transferring BYOL’s representation to other vision tasks.

## 5 Building intuitions with ablations

We present ablations on BYOL to give an intuition of its behavior and performance. For reproducibility, we run each configuration of parameters over three seeds, and report the average performance. We also report the half difference between the best and worst runs when it is larger than $0.25$. Although previous works perform ablations at $100$ epochs [^8] [^12], we notice that relative improvements at $100$ epochs do not always hold over longer training. For this reason, we run ablations over $300$ epochs on $64$ TPU v $3$ cores, which yields consistent results compared to our baseline training of $1000$ epochs. For all the experiments in this section, we set the initial learning rate to $0.3$ with batch size $4096$, the weight decay to $10^{-6}$ as in SimCLR [^8] and the base target decay rate $\tau_{\text{base}}$ to $0.99$. In this section we report results in top- $1$ accuracy on ImageNet under the linear evaluation protocol as in Section C.1.

#### Batch size

Among contrastive methods, the ones that draw negative examples from the minibatch suffer performance drops when their batch size is reduced. BYOL does not use negative examples and we expect it to be more robust to smaller batch sizes. To empirically verify this hypothesis, we train both BYOL and SimCLR using different batch sizes from $128$ to $4096$. To avoid re-tuning other hyperparameters, we average gradients over $N$ consecutive steps before updating the online network when reducing the batch size by a factor $N$. The target network is updated once every $N$ steps, after the update of the online network; we accumulate the $N$ -steps in parallel in our runs.

As shown in Figure 3(a), the performance of SimCLR rapidly deteriorates with batch size, likely due to the decrease in the number of negative examples. In contrast, the performance of BYOL remains stable over a wide range of batch sizes from $256$ to $4096$, and only drops for smaller values due to batch normalization layers in the encoder.<sup>7</sup>

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2006.07733/assets/x3.png)

(a) Impact of batch size

#### Image augmentations

Contrastive methods are sensitive to the choice of image augmentations. For instance, SimCLR does not work well when removing color distortion from its image augmentations. As an explanation, SimCLR shows that crops of the same image mostly share their color histograms. At the same time, color histograms vary across images. Therefore, when a contrastive task only relies on random crops as image augmentations, it can be mostly solved by focusing on color histograms alone. As a result the representation is not incentivized to retain information beyond color histograms. To prevent that, SimCLR adds color distortion to its set of image augmentations. Instead, BYOL is incentivized to keep any information captured by the target representation into its online network, to improve its predictions. Therefore, even if augmented views of a same image share the same color histogram, BYOL is still incentivized to retain additional features in its representation. For that reason, we believe that BYOL is more robust to the choice of image augmentations than contrastive methods.

Results presented in Figure 3(b) support this hypothesis: the performance of BYOL is much less affected than the performance of SimCLR when removing color distortions from the set of image augmentations ($-9.1$ accuracy points for BYOL, $-22.2$ accuracy points for SimCLR). When image augmentations are reduced to mere random crops, BYOL still displays good performance ($59.4\%$, *i.e.* $-13.1$ points from $72.5\%$ ), while SimCLR loses more than a third of its performance ($40.3\%$, *i.e.* $-27.6$ points from $67.9\%$). We report additional ablations in Section F.3.

#### Bootstrapping

BYOL uses the projected representation of a target network, whose weights are an exponential moving average of the weights of the online network, as target for its predictions. This way, the weights of the target network represent a delayed and more stable version of the weights of the online network. When the target decay rate is $1$, the target network is never updated, and remains at a constant value corresponding to its initialization. When the target decay rate is $00$, the target network is instantaneously updated to the online network at each step. There is a trade-off between updating the targets too often and updating them too slowly, as illustrated in Table 5(a). Instantaneously updating the target network ($\tau=0$) destabilizes training, yielding very poor performance while never updating the target ($\tau=1$) makes the training stable but prevents iterative improvement, ending with low-quality final representation. All values of the decay rate between $0.9$ and $0.999$ yield performance above $68.4\%$ top- $1$ accuracy at $300$ epochs.

| Target | $\tau_{\text{base}}$ | Top- $1$ |
| --- | --- | --- |
| Constant random network | $1$ | $18.8{\scriptstyle\pm 0.7}$ |
| Moving average of online | $0.999$ | $69.8$ |
| Moving average of online | $0.99$ | $\bf{72.5}$ |
| Moving average of online | $0.9$ | $68.4$ |
| Stop gradient of online <sup>†</sup> | $00$ | $0.3$ |

(a) Results for different target modes. <sup>†</sup> In the *stop gradient of online*, $\tau=\tau_{\text{base}}=0$ is kept constant throughout training.

| Method | Predictor | Target network | $\beta$ | Top- $1$ |
| --- | --- | --- | --- | --- |
| BYOL | ✓ | ✓ | 0 | $\bf{72.5}$ |
| $-$ | ✓ | ✓ | 1 | $70.9$ |
| $-$ |  | ✓ | 1 | $70.7$ |
| SimCLR |  |  | 1 | $69.4$ |
| $-$ | ✓ |  | 1 | $69.1$ |
| $-$ | ✓ |  | 0 | $0.3$ |
| $-$ |  | ✓ | 0 | $0.2$ |
| $-$ |  |  | 0 | $0.1$ |

(b) Intermediate variants between BYOL and SimCLR.

Table 5: Ablations with top- $1$ accuracy (in %) at $300$ epochs under linear evaluation on ImageNet.

#### Ablation to contrastive methods

In this subsection, we recast SimCLR and BYOL using the same formalism to better understand where the improvement of BYOL over SimCLR comes from. Let us consider the following objective that extends the InfoNCE objective [^10] [^84] (see Section F.4),

$$
\displaystyle\text{InfoNCE}^{\alpha,\beta}_{{\theta}}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\frac{2}{B}\sum_{i=1}^{B}S_{\theta}(v_{i},v^{\prime}_{i})\!-\beta\cdot\frac{2\alpha}{B}\sum_{i=1}^{B}\ln\mathopen{}\mathclose{{}\left(\sum\limits_{j\neq i}\exp\frac{S_{\theta}(v_{i},v_{j})}{\alpha}+\sum\limits_{j}\exp\frac{S_{\theta}(v_{i},v^{\prime}_{j})}{\alpha}}\right)\mathbin{\raisebox{2.15277pt}{,}}
$$

where $\alpha>0$ is a fixed temperature, $\beta\in[0,1]$ a weighting coefficient, $B$ the batch size, $v$ and $v^{\prime}$ are batches of augmented views where for any batch index $i$, $v_{i}$ and $v^{\prime}_{i}$ are augmented views from the same image; the real-valued function $S_{\theta}$ quantifies pairwise similarity between augmented views. For any augmented view $u$ we denote $z_{\theta}(u)\triangleq f_{\theta}(g_{\theta}(u))$ and $z_{\xi}(u)\triangleq f_{\xi}(g_{\xi}(u))$. For given $\phi$ and $\psi$, we consider the normalized dot product

$$
S_{\theta}(u_{1},u_{2})\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\frac{\langle\phi(u_{1}),\psi(u_{2})\rangle}{\|\phi(u_{1})\|_{2}\cdot\|\psi(u_{2})\|_{2}}\cdot
$$

Up to minor details (cf. Section F.5), we recover the SimCLR loss with $\phi(u_{1})=z_{\theta}(u_{1})$ (no predictor), $\psi(u_{2})=z_{\theta}(u_{2})$ (no target network) and $\beta=1$. We recover the BYOL loss when using a predictor and a target network, *i.e.,* $\phi(u_{1})=p_{\theta}\mathopen{}\mathclose{{}\left(z_{\theta}(u_{1})}\right)$ and $\psi(u_{2})=z_{\xi}(u_{2})$ with $\beta=0$. To evaluate the influence of the target network, the predictor and the coefficient $\beta$, we perform an ablation over them. Results are presented in Table 5(b) and more details are given in Section F.4.

The only variant that performs well without negative examples (i.e., with $\beta=0$) is BYOL, using both a bootstrap target network and a predictor. Adding the negative pairs to BYOL’s loss without re-tuning the temperature parameter hurts its performance. In Section F.4, we show that we can add back negative pairs and still match the performance of BYOL with proper tuning of the temperature.

Simply adding a target network to SimCLR already improves performance ($+1.6$ points). This sheds new light on the use of the target network in MoCo [^9], where the target network is used to provide more negative examples. Here, we show that by mere stabilization effect, even when using the same number of negative examples, using a target network is beneficial. Finally, we observe that modifying the architecture of $S_{\theta}$ to include a predictor only mildly affects the performance of SimCLR.

#### Network hyperparameters

In Appendix F, we explore how other network parameters may impact BYOL’s performance. We iterate over multiple weight decays, learning rates, and projector/encoder architectures to observe that small hyperparameter changes do not drastically alter the final score. We note that removing the weight decay in either BYOL or SimCLR leads to network divergence, emphasizing the need for weight regularization in the self-supervised setting. Furthermore, we observe that changing the scaling factor in the network initialization [^85] did not impact the performance (higher than $72\%$ top- $1$ accuracy).

#### Relationship with Mean Teacher

Another semi-supervised approach, Mean Teacher (MT) [^20], complements a supervised loss on few labels with an additional consistency loss. In [^20], this consistency loss is the $\ell_{2}$ distance between the logits from a student network, and those of a temporally averaged version of the student network, called teacher. Removing the predictor in BYOL results in an unsupervised version of MT with no classification loss that uses image augmentations instead of the original architectural noise (e.g., dropout). This variant of BYOL collapses (Row 7 of Table 5) which suggests that the additional predictor is critical to prevent collapse in an unsupervised scenario.

#### Importance of a near-optimal predictor

Table 5(b) already shows the importance of combining a predictor and a target network: the representation does collapse when either is removed. We further found that we can remove the target network without collapse by making the predictor near-optimal, either by (i) using an optimal *linear* predictor (obtained by linear regression on the current batch) before back-propagating the error through the network ($52.5\%$ top-1 accuracy), or (ii) increasing the learning rate of the predictor ($66.5\%$ top-1). By contrast, increasing the learning rates of both projector *and* predictor (without target network) yields poor results ($\approx 25\%$ top-1). See Appendix I for more details. This seems to indicate that keeping the predictor near-optimal at all times is important to preventing collapse, which may be one of the roles of BYOL’s target network.

## 6 Conclusion

We introduced BYOL, a new algorithm for self-supervised learning of image representations. BYOL learns its representation by predicting previous versions of its outputs, without using negative pairs. We show that BYOL achieves state-of-the-art results on various benchmarks. In particular, under the linear evaluation protocol on ImageNet with a ResNet- $50$ ($1\times$), BYOL achieves a new state of the art and bridges most of the remaining gap between self-supervised methods and the supervised learning baseline of [^8]. Using a ResNet- $200$ $(2\times)$, BYOL reaches a top- $1$ accuracy of $79.6\%$ which improves over the previous state of the art ($76.8\%$) while using $30\%$ fewer parameters.

Nevertheless, BYOL remains dependent on existing sets of augmentations that are specific to vision applications. To generalize BYOL to other modalities (e.g., audio, video, text, …) it is necessary to obtain similarly suitable augmentations for each of them. Designing such augmentations may require significant effort and expertise. Therefore, automating the search for these augmentations would be an important next step to generalize BYOL to other modalities.

## Broader impact

The presented research should be categorized as research in the field of unsupervised learning. This work may inspire new algorithms, theoretical, and experimental investigation. The algorithm presented here can be used for many different vision applications and a particular use may have both positive or negative impacts, which is known as the dual use problem. Besides, as vision datasets could be biased, the representation learned by BYOL could be susceptible to replicate these biases.

## Acknowledgements

The authors would like to thank the following people for their help throughout the process of writing this paper, in alphabetical order: Aaron van den Oord, Andrew Brock, Jason Ramapuram, Jeffrey De Fauw, Karen Simonyan, Katrina McKinney, Nathalie Beauguerlange, Olivier Henaff, Oriol Vinyals, Pauline Luc, Razvan Pascanu, Sander Dieleman, and the DeepMind team. We especially thank Jason Ramapuram and Jeffrey De Fauw, who provided the JAX SimCLR reproduction used throughout the paper.

## References

## Appendix A Algorithm

Inputs:

| $\mathcal{D}$, $\mathcal{T},$ and $\mathcal{T}^{\prime}$ | set of images and distributions of transformations |
| --- | --- |
| ${\theta}$, $f_{\theta}$, $g_{\theta},$ and $q_{\theta}$ | initial online parameters, encoder, projector, and predictor |
| $\xi$, $f_{\xi}$, $g_{\xi}$ | initial target parameters, target encoder, and target projector |
| $\mathrm{optimizer}$ | optimizer, updates online parameters using the loss gradient |
| $K$ and $N$ | total number of optimization steps and batch size |
| $\{\tau_{k}\}_{k=1}^{K}$ and $\{\eta_{k}\}_{k=1}^{K}$ | target network update schedule and learning rate schedule |

for *$k=1$ to $K$* do

 $\mathcal{B}\leftarrow\{x_{i}\sim\mathcal{D}\}_{i=1}^{N}$

// sample a batch of $N$ images

for *$x_{i}\in\mathcal{B}$* do

 $t\sim\mathcal{T}{\rm\ and\ }t^{\prime}\sim\mathcal{T}^{\prime}$

// sample image transformations

$z_{1}\leftarrow g_{\theta}\mathopen{}\mathclose{{}\left(f_{\theta}(t(x_{i}))}\right)$ and $z_{2}\leftarrow g_{\theta}\mathopen{}\mathclose{{}\left(f_{\theta}(t^{\prime}(x_{i}))}\right)$

// compute projections

$z^{\prime}_{1}\leftarrow g_{\xi}\mathopen{}\mathclose{{}\left(f_{\xi}(t^{\prime}(x_{i}))}\right)$ and $z^{\prime}_{2}\leftarrow g_{\xi}\mathopen{}\mathclose{{}\left(f_{\xi}(t(x_{i}))}\right)$

// compute target projections

 $l_{i}\leftarrow-2\cdot\mathopen{}\mathclose{{}\left(\frac{\langle q_{\theta}(z_{1}),z^{\prime}_{1}\rangle}{\mathopen{}\mathclose{{}\left\|q_{\theta}(z_{1})}\right\|_{2}\cdot\mathopen{}\mathclose{{}\left\|z^{\prime}_{1}}\right\|_{2}}+\frac{\langle q_{\theta}(z_{2}),z^{\prime}_{2}\rangle}{\mathopen{}\mathclose{{}\left\|q_{\theta}(z_{2})}\right\|_{2}\cdot\mathopen{}\mathclose{{}\left\|z^{\prime}_{2}}\right\|_{2}}}\right)$

// compute the loss for $x_{i}$

end for

 ${\delta\theta}\leftarrow\frac{1}{N}\sum\limits_{i=1}^{N}\partial_{\theta}l_{i}$

// compute the total loss gradient w.r.t. $\theta$

 ${\theta}\leftarrow\mathrm{optimizer}({\theta},{\delta\theta},\eta_{k})$

// update online parameters

 $\xi\leftarrow\tau_{k}\xi+(1-\tau_{k}){\theta}$

// update target parameters

end for

Output: encoder $f_{\theta}$

Algorithm 1 BYOL: Bootstrap Your Own Latent

## Appendix B Image augmentations

During self-supervised training, BYOL uses the following image augmentations (which are a subset of the ones presented in [^8]):

- random cropping: a random patch of the image is selected, with an area uniformly sampled between 8% and 100% of that of the original image, and an aspect ratio logarithmically sampled between $3/4$ and $4/3$. This patch is then resized to the target size of $224\times 224$ using bicubic interpolation;
- optional left-right flip;
- color jittering: the brightness, contrast, saturation and hue of the image are shifted by a uniformly random offset applied on all the pixels of the same image. The order in which these shifts are performed is randomly selected for each patch;
- color dropping: an optional conversion to grayscale. When applied, output intensity for a pixel $(r,g,b)$ corresponds to its luma component, computed as $0.2989r+0.5870g+0.1140b$;
- Gaussian blurring: for a $224\times 224$ image, a square Gaussian kernel of size $23\times 23$ is used, with a standard deviation uniformly sampled over $[0.1,2.0]$;
- solarization: an optional color transformation $x\mapsto x\cdot{\bf 1}_{\{x<0.5\}}+(1-x)\cdot{\bf 1}_{\{x\geq 0.5\}}$ for pixels with values in $[0,1]$.

Augmentations from the sets $\mathcal{T}$ and $\mathcal{T}^{\prime}$ (introduced in Section 3) are compositions of the above image augmentations in the listed order, each applied with a predetermined probability. The image augmentations parameters are listed in Table 6.

During evaluation, we use a center crop similar to [^8]: images are resized to $256$ pixels along the shorter side using bicubic resampling, after which a $224\times 224$ center crop is applied. In both training and evaluation, we normalize color channels by subtracting the average color and dividing by the standard deviation, computed on ImageNet, after applying the augmentations.

| Parameter | $\mathcal{T}$ | $\mathcal{T}^{\prime}$ |
| --- | --- | --- |
| Random crop probability | $1.0$ | $1.0$ |
| Flip probability | $0.5$ | $0.5$ |
| Color jittering probability | $0.8$ | $0.8$ |
| Brightness adjustment max intensity | $0.4$ | $0.4$ |
| Contrast adjustment max intensity | $0.4$ | $0.4$ |
| Saturation adjustment max intensity | $0.2$ | $0.2$ |
| Hue adjustment max intensity | $0.1$ | $0.1$ |
| Color dropping probability | $0.2$ | $0.2$ |
| Gaussian blurring probability | $1.0$ | $0.1$ |
| Solarization probability | $0.0$ | $0.2$ |

Table 6: Parameters used to generate image augmentations.

## Appendix C Evaluation on ImageNet training

### C.1 Self-supervised learning evaluation on ImageNet

#### Linear evaluation protocol on ImageNet

As in [^48] [^74] [^8] [^37], we use the standard linear evaluation protocol on ImageNet, which consists in training a linear classifier on top of the frozen representation, *i.e.*, without updating the network parameters nor the batch statistics. At training time, we apply spatial augmentations, i.e., random crops with resize to $224\times 224$ pixels, and random flips. At test time, images are resized to $256$ pixels along the shorter side using bicubic resampling, after which a $224\times 224$ center crop is applied. In both cases, we normalize the color channels by subtracting the average color and dividing by the standard deviation (computed on ImageNet), after applying the augmentations. We optimize the cross-entropy loss using SGD with Nesterov momentum over $80$ epochs, using a batch size of $1024$ and a momentum of $0.9$. We do not use any regularization methods such as weight decay, gradient clipping [^86], tclip [^34], or logits regularization. We finally sweep over $5$ learning rates $\{0.4,0.3,0.2,0.1,0.05\}$ on a local validation set (10009 images from ImageNet train set), and report the accuracy of the best validation hyperparameter on the test set (which is the public validation set of the original ILSVRC2012 ImageNet dataset).

#### Variant on linear evaluation on ImageNet

In this paragraph only, we deviate from the protocol of [^8] [^37] and propose another way of performing linear evaluation on top of a frozen representation. This method achieves better performance both in top-1 and top-5 accuracy.

- We replace the spatial augmentations (random crops with resize to $224\times 224$ pixels and random flips) with the pre-train augmentations of Appendix B. This method was already used in [^32] with a different subset of pre-train augmentations.
- We regularize the linear classifier as in [^34] <sup>8</sup> by clipping the logits using a hyperbolic tangent function
	$$
	{\rm tclip}(x)\triangleq\alpha\cdot\tanh(x/\alpha),
	$$
	where $\alpha$ is a positive scalar, and by adding a logit-regularization penalty term in the loss
	$$
	{\rm Loss}(x,y)\triangleq{\rm cross\_entropy}({\rm tclip}(x),y)+\beta\cdot{\rm average}({\rm tclip}(x)^{2}),
	$$
	where $x$ are the logits, $y$ are the target labels, and $\beta$ is the regularization parameter. We set $\alpha=20$ and $\beta=1e{-2}$.

We report in Table 7 the top-1 and top-5 accuracy on ImageNet using this modified protocol. These modifications in the evaluation protocol increase the BYOL’s top- $1$ accuracy from $74.3\%$ to $74.8\%$ with a ResNet- $50$ ($1\times$).

<table><thead><tr><th>Architecture</th><th>Pre-train augmentations</th><th>Logits regularization</th><th>Top- <math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></th><th>Top- <math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></th></tr></thead><tbody><tr><td rowspan="4">ResNet- <math><semantics><mn>50</mn> <cn>50</cn> <annotation>50</annotation></semantics></math> (<math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math>)</td><td></td><td></td><td><math><semantics><mn>74.3</mn> <cn>74.3</cn> <annotation>74.3</annotation></semantics></math></td><td><math><semantics><mn>91.6</mn> <cn>91.6</cn> <annotation>91.6</annotation></semantics></math></td></tr><tr><td>✓</td><td></td><td><math><semantics><mn>74.4</mn> <cn>74.4</cn> <annotation>74.4</annotation></semantics></math></td><td><math><semantics><mn>91.8</mn> <cn>91.8</cn> <annotation>91.8</annotation></semantics></math></td></tr><tr><td></td><td>✓</td><td><math><semantics><mn>74.7</mn> <cn>74.7</cn> <annotation>74.7</annotation></semantics></math></td><td><math><semantics><mn>91.8</mn> <cn>91.8</cn> <annotation>91.8</annotation></semantics></math></td></tr><tr><td>✓</td><td>✓</td><td><math><semantics><mn>74.8</mn> <cn>74.8</cn> <annotation>\bf{74.8}</annotation></semantics></math></td><td><math><semantics><mn>91.8</mn> <cn>91.8</cn> <annotation>\bf{91.8}</annotation></semantics></math></td></tr><tr><td rowspan="4">ResNet- <math><semantics><mn>50</mn> <cn>50</cn> <annotation>50</annotation></semantics></math> (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</td><td></td><td></td><td><math><semantics><mn>78.6</mn> <cn>78.6</cn> <annotation>78.6</annotation></semantics></math></td><td><math><semantics><mn>94.2</mn> <cn>94.2</cn> <annotation>94.2</annotation></semantics></math></td></tr><tr><td>✓</td><td></td><td><math><semantics><mn>78.6</mn> <cn>78.6</cn> <annotation>78.6</annotation></semantics></math></td><td><math><semantics><mn>94.3</mn> <cn>94.3</cn> <annotation>94.3</annotation></semantics></math></td></tr><tr><td></td><td>✓</td><td><math><semantics><mn>78.9</mn> <cn>78.9</cn> <annotation>78.9</annotation></semantics></math></td><td><math><semantics><mn>94.3</mn> <cn>94.3</cn> <annotation>94.3</annotation></semantics></math></td></tr><tr><td>✓</td><td>✓</td><td><math><semantics><mn>79.0</mn> <cn>79.0</cn> <annotation>\bf{79.0}</annotation></semantics></math></td><td><math><semantics><mn>94.5</mn> <cn>94.5</cn> <annotation>\bf{94.5}</annotation></semantics></math></td></tr><tr><td rowspan="4">ResNet- <math><semantics><mn>200</mn> <cn>200</cn> <annotation>200</annotation></semantics></math> (<math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math>)</td><td></td><td></td><td><math><semantics><mn>79.6</mn> <cn>79.6</cn> <annotation>79.6</annotation></semantics></math></td><td><math><semantics><mn>94.8</mn> <cn>94.8</cn> <annotation>94.8</annotation></semantics></math></td></tr><tr><td>✓</td><td></td><td><math><semantics><mn>79.6</mn> <cn>79.6</cn> <annotation>79.6</annotation></semantics></math></td><td><math><semantics><mn>94.8</mn> <cn>94.8</cn> <annotation>94.8</annotation></semantics></math></td></tr><tr><td></td><td>✓</td><td><math><semantics><mn>79.8</mn> <cn>79.8</cn> <annotation>79.8</annotation></semantics></math></td><td><math><semantics><mn>95.0</mn> <cn>95.0</cn> <annotation>95.0</annotation></semantics></math></td></tr><tr><td>✓</td><td>✓</td><td><math><semantics><mn>80.0</mn> <cn>80.0</cn> <annotation>\bf{80.0}</annotation></semantics></math></td><td><math><semantics><mn>95.0</mn> <cn>95.0</cn> <annotation>\bf{95.0}</annotation></semantics></math></td></tr></tbody></table>

Table 7: Different linear evaluation protocols on ResNet architectures by either replacing the spatial augmentations with pre-train augmentations, or regularizing the linear classifier. No pre-train augmentations and no logits regularization correspond to the evaluation protocol of the main paper, which is the same as in [^8] [^37].

#### Semi-supervised learning on ImageNet

We follow the semi-supervised learning protocol of [^8] [^77]. We first initialize the network with the parameters of the pretrained representation, and fine-tune it with a subset of ImageNet labels. At training time, we apply spatial augmentations, i.e., random crops with resize to $224\times 224$ pixels and random flips. At test time, images are resized to $256$ pixels along the shorter side using bicubic resampling, after which a $224\times 224$ center crop is applied. In both cases, we normalize the color channels by subtracting the average color and dividing by the standard deviation (computed on ImageNet), after applying the augmentations. We optimize the cross-entropy loss using SGD with Nesterov momentum. We used a batch size of $1024$, a momentum of $0.9$. We do not use any regularization methods such as weight decay, gradient clipping [^86], tclip [^34], or logits rescaling. We sweep over the learning rate $\{0.01,0.02,0.05,0.1,0.005\}$ and the number of epochs $\{30,50\}$ and select the hyperparameters achieving the best performance on our local validation set to report test performance.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2006.07733/assets/x5.png)

(a) Top-1 accuracy

<table><thead><tr><th colspan="3"><em>Supervised:</em></th><th colspan="3"><em>Semi-supervised (<math><semantics><mrow><mn>100</mn> <mo>%</mo></mrow> <apply><csymbol>percent</csymbol> <cn>100</cn></apply> <annotation>100\%</annotation></semantics></math>):</em></th></tr></thead><tbody><tr><th>Method</th><td>Top- <math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></td><td>Top- <math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></td><th>Method</th><td>Top- <math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></td><td>Top- <math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></td></tr><tr><th>Supervised <sup><a href="#fn:8">8</a></sup></th><td><math><semantics><mn>76.5</mn> <cn>76.5</cn> <annotation>76.5</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><th>SimCLR <sup><a href="#fn:8">8</a></sup></th><td><math><semantics><mn>76.0</mn> <cn>76.0</cn> <annotation>76.0</annotation></semantics></math></td><td><math><semantics><mn>93.1</mn> <cn>93.1</cn> <annotation>93.1</annotation></semantics></math></td></tr><tr><th>AutoAugment <sup><a href="#fn:87">87</a></sup></th><td><math><semantics><mn>77.6</mn> <cn>77.6</cn> <annotation>77.6</annotation></semantics></math></td><td><math><semantics><mn>93.8</mn> <cn>93.8</cn> <annotation>93.8</annotation></semantics></math></td><th>SimCLR (repro)</th><td><math><semantics><mn>76.5</mn> <cn>76.5</cn> <annotation>76.5</annotation></semantics></math></td><td><math><semantics><mn>93.5</mn> <cn>93.5</cn> <annotation>93.5</annotation></semantics></math></td></tr><tr><th>MaxUp <sup><a href="#fn:75">75</a></sup></th><td><math><semantics><mn>78.9</mn> <cn>78.9</cn> <annotation>\bf{78.9}</annotation></semantics></math></td><td><math><semantics><mn>94.2</mn> <cn>94.2</cn> <annotation>\bf{94.2}</annotation></semantics></math></td><th>BYOL</th><td><math><semantics><mn>77.7</mn> <cn>77.7</cn> <annotation>\bf{77.7}</annotation></semantics></math></td><td><math><semantics><mn>93.9</mn> <cn>93.9</cn> <annotation>\bf{93.9}</annotation></semantics></math></td></tr></tbody></table>

Table 8: Semi-supervised training with the full ImageNet on a ResNet- $50$ ($\times 1$). We also report other fully supervised methods for extensive comparisons.

In Table 2 presented in the main text, we fine-tune the representation over the $1$ % and $10$ % ImageNet splits from [^8] with various ResNet architectures.

In Figure 4, we fine-tune the representation over $1$ %, $2$ %, $5$ %, $10$ %, $20$ %, $50$ %, and $100$ % of the ImageNet dataset as in [^32] with a ResNet- $50$ ($1\times$) architecture, and compare them with a supervised baseline and a fine-tuned SimCLR representation. In this case and contrary to Table 2 we don’t reuse the splits from SimCLR but we create our own via a balanced selection. In this setting, we observed that tuning a BYOL representation always outperforms a supervised baseline trained from scratch. In Figure 5, we then fine-tune the representation over multiple ResNet architectures. We observe that the largest networks are prone to overfitting as they are outperformed by ResNets with identical depth but smaller scaling factor. This overfitting is further confirmed when looking at the training and evaluation loss: large networks have lower training losses, but higher validation losses than some of their slimmer counterparts. Regularization methods are thus recommended when tuning on large architectures.

Finally, we fine-tune the representation over the full ImageNet dataset. We report the results in Table 8 along with supervised baselines trained on ImageNet. We observe that fine-tuning the SimCLR checkpoint does not yield better results (in our reproduction, which matches the results reported in the original paper [^8]) than using a random initialization ($76.5$ top- $1$). Instead, BYOL’s initialization checkpoint leads to a high final score ($77.7$ top- $1$), higher than the vanilla supervised baseline of [^8], matching the strong supervised baseline of AutoAugment [^87] but still $1.2$ points below the stronger supervised baseline [^75], which uses advanced supervised learning techniques.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2006.07733/assets/x7.png)

Figure 5: Semi-supervised training with a fraction of ImageNet labels on multiple ResNets architecture pretrained with BYOL. Note that large networks are facing overfitting problems.

### C.2 Linear evaluation on larger architectures and supervised baselines

<table><tbody><tr><td></td><th></th><th></th><th colspan="2">BYOL</th><th colspan="2">Supervised (ours)</th><th>Supervised <sup><a href="#fn:8">8</a></sup></th></tr><tr><th>Architecture</th><th>Multiplier</th><th>Weights</th><th>Top-1</th><th>Top-5</th><th>Top-1</th><th>Top-5</th><th>Top-1</th></tr><tr><td>ResNet-50</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math></td><td>24M</td><td><math><semantics><mn>74.3</mn> <cn>74.3</cn> <annotation>74.3</annotation></semantics></math></td><td><math><semantics><mn>91.6</mn> <cn>91.6</cn> <annotation>91.6</annotation></semantics></math></td><td><math><semantics><mn>76.4</mn> <cn>76.4</cn> <annotation>76.4</annotation></semantics></math></td><td><math><semantics><mn>92.9</mn> <cn>92.9</cn> <annotation>92.9</annotation></semantics></math></td><td><math><semantics><mn>76.5</mn> <cn>76.5</cn> <annotation>76.5</annotation></semantics></math></td></tr><tr><td>ResNet-101</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math></td><td>43M</td><td><math><semantics><mn>76.4</mn> <cn>76.4</cn> <annotation>76.4</annotation></semantics></math></td><td><math><semantics><mn>93.0</mn> <cn>93.0</cn> <annotation>93.0</annotation></semantics></math></td><td><math><semantics><mn>78.0</mn> <cn>78.0</cn> <annotation>78.0</annotation></semantics></math></td><td><math><semantics><mn>94.0</mn> <cn>94.0</cn> <annotation>94.0</annotation></semantics></math></td><td>-</td></tr><tr><td>ResNet-152</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math></td><td>58M</td><td><math><semantics><mn>77.3</mn> <cn>77.3</cn> <annotation>77.3</annotation></semantics></math></td><td><math><semantics><mn>93.7</mn> <cn>93.7</cn> <annotation>93.7</annotation></semantics></math></td><td><math><semantics><mn>79.1</mn> <cn>79.1</cn> <annotation>79.1</annotation></semantics></math></td><td><math><semantics><mn>94.5</mn> <cn>94.5</cn> <annotation>94.5</annotation></semantics></math></td><td>-</td></tr><tr><td>ResNet-200</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math></td><td>63M</td><td><math><semantics><mn>77.8</mn> <cn>77.8</cn> <annotation>77.8</annotation></semantics></math></td><td><math><semantics><mn>93.9</mn> <cn>93.9</cn> <annotation>93.9</annotation></semantics></math></td><td><math><semantics><mn>79.3</mn> <cn>79.3</cn> <annotation>79.3</annotation></semantics></math></td><td><math><semantics><mn>94.6</mn> <cn>94.6</cn> <annotation>94.6</annotation></semantics></math></td><td>-</td></tr><tr><td>ResNet-50</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math></td><td>94M</td><td><math><semantics><mn>77.4</mn> <cn>77.4</cn> <annotation>77.4</annotation></semantics></math></td><td><math><semantics><mn>93.6</mn> <cn>93.6</cn> <annotation>93.6</annotation></semantics></math></td><td><math><semantics><mn>79.9</mn> <cn>79.9</cn> <annotation>79.9</annotation></semantics></math></td><td><math><semantics><mn>95.0</mn> <cn>95.0</cn> <annotation>95.0</annotation></semantics></math></td><td><math><semantics><mn>77.8</mn> <cn>77.8</cn> <annotation>77.8</annotation></semantics></math></td></tr><tr><td>ResNet-101</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math></td><td>170M</td><td><math><semantics><mn>78.7</mn> <cn>78.7</cn> <annotation>78.7</annotation></semantics></math></td><td><math><semantics><mn>94.3</mn> <cn>94.3</cn> <annotation>94.3</annotation></semantics></math></td><td><math><semantics><mn>80.3</mn> <cn>80.3</cn> <annotation>80.3</annotation></semantics></math></td><td><math><semantics><mn>95.0</mn> <cn>95.0</cn> <annotation>95.0</annotation></semantics></math></td><td>-</td></tr><tr><td>ResNet-50</td><td><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math></td><td>211M</td><td><math><semantics><mn>78.2</mn> <cn>78.2</cn> <annotation>78.2</annotation></semantics></math></td><td><math><semantics><mn>93.9</mn> <cn>93.9</cn> <annotation>93.9</annotation></semantics></math></td><td><math><semantics><mn>80.2</mn> <cn>80.2</cn> <annotation>80.2</annotation></semantics></math></td><td><math><semantics><mn>95.0</mn> <cn>95.0</cn> <annotation>95.0</annotation></semantics></math></td><td>-</td></tr><tr><td>ResNet-152</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math></td><td>232M</td><td><math><semantics><mn>79.0</mn> <cn>79.0</cn> <annotation>79.0</annotation></semantics></math></td><td><math><semantics><mn>94.6</mn> <cn>94.6</cn> <annotation>94.6</annotation></semantics></math></td><td><math><semantics><mn>80.6</mn> <cn>80.6</cn> <annotation>80.6</annotation></semantics></math></td><td><math><semantics><mn>95.3</mn> <cn>95.3</cn> <annotation>95.3</annotation></semantics></math></td><td>-</td></tr><tr><td>ResNet-200</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math></td><td>250M</td><td><math><semantics><mn>79.6</mn> <cn>79.6</cn> <annotation>\bf{79.6}</annotation></semantics></math></td><td><math><semantics><mn>94.9</mn> <cn>94.9</cn> <annotation>\bf{94.9}</annotation></semantics></math></td><td><math><semantics><mn>80.1</mn> <cn>80.1</cn> <annotation>80.1</annotation></semantics></math></td><td><math><semantics><mn>95.2</mn> <cn>95.2</cn> <annotation>95.2</annotation></semantics></math></td><td>-</td></tr><tr><td>ResNet-50</td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math></td><td>375M</td><td><math><semantics><mn>78.6</mn> <cn>78.6</cn> <annotation>78.6</annotation></semantics></math></td><td><math><semantics><mn>94.2</mn> <cn>94.2</cn> <annotation>94.2</annotation></semantics></math></td><td><math><semantics><mn>80.7</mn> <cn>80.7</cn> <annotation>80.7</annotation></semantics></math></td><td><math><semantics><mn>95.3</mn> <cn>95.3</cn> <annotation>\bf{95.3}</annotation></semantics></math></td><td><math><semantics><mn>78.9</mn> <cn>78.9</cn> <annotation>78.9</annotation></semantics></math></td></tr><tr><td>ResNet-101</td><td><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math></td><td>382M</td><td><math><semantics><mn>78.4</mn> <cn>78.4</cn> <annotation>78.4</annotation></semantics></math></td><td><math><semantics><mn>94.2</mn> <cn>94.2</cn> <annotation>94.2</annotation></semantics></math></td><td><math><semantics><mn>80.7</mn> <cn>80.7</cn> <annotation>80.7</annotation></semantics></math></td><td><math><semantics><mn>95.3</mn> <cn>95.3</cn> <annotation>\bf{95.3}</annotation></semantics></math></td><td>-</td></tr><tr><td>ResNet-152</td><td><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math></td><td>522M</td><td><math><semantics><mn>79.5</mn> <cn>79.5</cn> <annotation>79.5</annotation></semantics></math></td><td><math><semantics><mn>94.6</mn> <cn>94.6</cn> <annotation>94.6</annotation></semantics></math></td><td><math><semantics><mn>80.9</mn> <cn>80.9</cn> <annotation>\bf{80.9}</annotation></semantics></math></td><td><math><semantics><mn>95.2</mn> <cn>95.2</cn> <annotation>95.2</annotation></semantics></math></td><td>-</td></tr></tbody></table>

Table 9: Linear evaluation of BYOL on ImageNet using larger encoders. Top- $1$ and top- $5$ accuracies are reported in %.

Here we investigate the performance of BYOL with deeper and wider ResNet architectures. We compare ourselves to the best supervised baselines from [^8] when available (rightmost column in table 9), which are also presented in Figure 1. Importantly, we close in on those baselines using the ResNet- $50$ ($2\times$) and the ResNet- $50$ ($4\times$) architectures, where we are within $0.4$ accuracy points of the supervised performance. To the best of our knowledge, this is the first time that the gap to supervised has been closed to such an extent using a self-supervised method under the linear evaluation protocol. Therefore, in order to ensure fair comparison, and suspecting that the supervised baselines’ performance in [^8] could be even further improved with appropriate data augmentations, we also report on our own reproduction of strong supervised baselines. We use RandAugment [^87] data augmentation for all large ResNet architectures (which are all version $1$, as per [^22]). We train our supervised baselines for up to $200$ epochs, using SGD with a Nesterov momentum value of $0.9$, a cosine-annealed learning rate after a $5$ epochs linear warmup period, weight decay with a value of $1e-4$, and a label smoothing [^88] value of $0.1$. Results are presented in Figure 6.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2006.07733/assets/x8.png)

Figure 6: Results for linear evaluation of BYOL compared to fully supervised baselines with various ResNet architectures. Our supervised baselines are ran with RandAugment 87 augmentations.

## Appendix D Transfer to other datasets

### D.1 Datasets

| Dataset | Classes | Original train examples | Train examples | Valid. examples | Test examples | Accuracy measure | Test provided |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ImageNet [^21] | $1000$ | $1281167$ | $1271158$ | $10009$ | $50000$ | Top-1 accuracy | \- |
| Food101 [^89] | $101$ | $75750$ | $68175$ | $7575$ | $25250$ | Top-1 accuracy | \- |
| CIFAR-10 [^78] | $10$ | $50000$ | $45000$ | $5000$ | $10000$ | Top-1 accuracy | \- |
| CIFAR-100 [^78] | $100$ | $50000$ | $44933$ | $5067$ | $10000$ | Top-1 accuracy | \- |
| Birdsnap [^90] | $500$ | $47386$ | $42405$ | $4981$ | $2443$ | Top-1 accuracy | \- |
| Sun397 (split 1) [^79] | $397$ | $19850$ | $15880$ | $3970$ | $19850$ | Top-1 accuracy | \- |
| Cars [^91] | $196$ | $8144$ | $6494$ | $1650$ | $8041$ | Top-1 accuracy | \- |
| Aircraft [^92] | $100$ | $3334$ | $3334$ | $3333$ | $3333$ | Mean per-class accuracy | Yes |
| PASCAL-VOC2007 [^80] | $20$ | $5011$ | $2501$ | $2510$ | $4952$ | 11-point mAP / AP50 | \- |
| PASCAL-VOC2012 [^80] | $21$ | $10582$ | $-$ | $2119$ | $1449$ | Mean IoU | \- |
| DTD (split 1) [^81] | $47$ | $1880$ | $1880$ | $1880$ | $1880$ | Top-1 accuracy | Yes |
| Pets [^93] | $37$ | $3680$ | $2940$ | $740$ | $3669$ | Mean per-class accuracy | \- |
| Caltech-101 [^94] | $101$ | $3060$ | $2550$ | $510$ | $6084$ | Mean per-class accuracy | \- |
| Places365 [^73] | $365$ | $1803460$ | $1803460$ | $-$ | $36500$ | Top-1 accuracy | \- |
| Flowers [^95] | $102$ | $1020$ | $1020$ | $1020$ | $6149$ | Mean per-class accuracy | Yes |

Table 10: Characteristics of image datasets used in transfer learning. When an official test split with labels is not publicly available, we use the official validation split as test set, and create a held-out validation set from the training examples.

We perform transfer via linear classification and fine-tuning on the same set of datasets as in [^8], namely Food- $101$ dataset [^89], CIFAR- $10$ [^78] and CIFAR- $100$ [^78], Birdsnap [^90], the SUN $397$ scene dataset [^79], Stanford Cars [^91], FGVC Aircraft [^92], the PASCAL VOC $2007$ classification task [^80], the Describable Textures Dataset (DTD) [^81], Oxford-IIIT Pets [^93], Caltech- $101$ [^94], and Oxford $102$ Flowers [^95]. As in [^8], we used the validation sets specified by the dataset creators to select hyperparameters for FGVC Aircraft, PASCAL VOC 2007, DTD, and Oxford 102 Flowers. On other datasets, we use the validation examples as test set, and hold out a subset of the training examples that we use as validation set. We use standard metrics for each datasets:

- Top- $1$: We compute the proportion of correctly classified examples.
- Mean per class: We compute the top- $1$ accuracy for each class separately and then compute the empirical mean over the classes.
- Point 11-mAP: We compute the empirical mean *average precision* as defined in [^80].
- Mean IoU: We compute the empirical mean Intersection-Over-Union as defined in [^80].
- AP50: We compute the Average Precision as defined in [^80].

We detail the validation procedures for some specific datasets:

- For Sun397 [^79], the original dataset specifies 10 train/test splits, all of which contain 50 examples/images of 397 different classes. We use the first train/test split. The original dataset specifies no validation split and therefore, the training images have been further subdivided into 40 images per class for the train split and 10 images per class for the valid split.
- For Birdsnap [^90], we use a random selection of valid images with the same number of images per category as the test split.
- For DTD [^81], the original dataset specifies 10 train/validation/test splits, we only use the first split.
- For Caltech-101 [^94], the original does not dataset specifies any train/test splits. We have followed the approach used in [^96]: This file defines datasets for 5 random splits of 25 training images per category, with 5 validation images per category and the remaining images used for testing.
- For ImageNet, we took the last $10009$ last images of the official tensorflow ImageNet split.
- For Oxford-IIIT Pets, the valid set consists of 20 randomly selected images per class.

Information about the dataset are summarized in Table 10.

### D.2 Transfer via linear classification

We follow the linear evaluation protocol of [^48] [^74] [^8] that we detail next for completeness. We train a regularized multinomial logistic regression classifier on top of the frozen representation, i.e., with frozen pretrained parameters and without re-computing batch-normalization statistics. In training and testing, we do not perform any image augmentations; images are resized to $224$ pixels along the shorter side using bicubic resampling and then normalized with ImageNet statistics. Finally, we minimize the cross-entropy objective using LBFGS with $\ell_{2}$ -regularization, where we select the regularization parameters from a range of $45$ logarithmically-spaced values between $10^{-6}$ and $10^{5}$. After choosing the best-performing hyperparameters on the validation set, the model is retrained on combined training and validation images together, using the chosen parameters. The final accuracy is reported on the test set.

### D.3 Transfer via fine-tuning

We follow the same fine-tuning protocol as in [^32] [^48] [^76] [^8] that we also detail for completeness. Specifically, we initialize the network with the parameters of the pretrained representation. At training time, we apply spatial transformation, i.e., random crops with resize to $224\times 224$ pixels and random flips. At test time, images are resized to $256$ pixels along the shorter side using bicubic resampling, after which a $224\times 224$ center crop is extracted. In both cases, we normalize the color channels by subtracting the average color and dividing by the standard deviation (computed on ImageNet), after applying the augmentations. We optimize the loss using SGD with Nesterov momentum for $20000$ steps with a batch size of $256$ and with a momentum of $0.9$. We set the momentum parameter for the batch normalization statistics to $\max(1-10/s,0.9)$ where $s$ is the number of steps per epoch. The learning rate and weight decay are selected respectively with a grid of seven logarithmically spaced learning rates between $0.0001$ and $0.1$, and $7$ logarithmically-spaced values of weight decay between $10^{-6}$ and $10^{-3}$, as well as no weight decay. These values of weight decay are divided by the learning rate. After choosing the best-performing hyperparameters on the validation set, the model is retrained on combined training and validation images together, using the chosen parameters. The final accuracy is reported on the test set.

### D.4 Implementation details for semantic segmentation

We use the same fully-convolutional network (FCN)-based [^7] architecture as [^9]. The backbone consists of the convolutional layers in ResNet- $50$. The $3\times 3$ convolutions in the conv $5$ blocks use dilation $2$ and stride $1$. This is followed by two extra $3\times 3$ convolutions with $256$ channels, each followed by batch normalization and ReLU activations, and a $1\times 1$ convolution for per-pixel classification. The dilation is set to $6$ in the two extra $3\times 3$ convolutions. The total stride is $16$ (FCN- $16$ s [^7]).

We train on the train\_aug2012 set and report results on val2012. Hyperparameters are selected on a $2119$ images held-out validation set. We use a standard per-pixel softmax cross-entropy loss to train the FCN. Training is done with random scaling (by a ratio in $\mathopen{}\mathclose{{}\left[0.5,2.0}\right]$), cropping, and horizontal flipping. The crop size is $513$. Inference is performed on the $\mathopen{}\mathclose{{}\left[513,513}\right]$ central crop. For training we use a batch size of $16$ and weight decay of $0.0001$. We select the base learning rate by sweeping across $5$ logarithmically spaced values between $10^{-3}$ and $10^{-1}$. The learning rate is multiplied by $0.1$ at the $70$ -th and $90$ -th percentile of training. We train for $30000$ iterations, and average the results on 5 seeds.

### D.5 Implementation details for object detection

For object detection, we follow prior work on Pascal detection transfer [^40] [^23] wherever possible. We use a Faster R-CNN [^82] detector with a R $50$ -C $4$ backbone with a frozen representation. The R $50$ -C $4$ backbone ends with the conv $4$ stage of a ResNet- $50$, and the box prediction head consists of the conv $5$ stage (including global pooling). We preprocess the images by applying multi-scale augmentation (rescaling the image so its longest edge is between $480$ and $1024$ pixels) but no other augmentation. We use an asynchronous SGD optimizer with $9$ workers and train for $1.5$ M steps. We used an initial learning rate of $10^{-3}$, which is reduced to $10^{-4}$ at 1M steps and to $10^{-5}$ at $1.2$ M steps.

### D.6 Implementation details for depth estimation

For depth estimation, we follow the same protocol as in [^83], and report its core components for completeness. We use a standard ResNet- $50$ backbone and feed the conv $5$ features into $4$ fast up-projection blocks with respective filter sizes $512$, $256$, $128$, and $64$. We use a reverse Huber loss function for training [^83] [^97].

The original NYU Depth v $2$ frames of size $\mathopen{}\mathclose{{}\left[640,480}\right]$ are down-sampled by a factor $0.5$ and center-cropped to $\mathopen{}\mathclose{{}\left[304,228}\right]$ pixels. Input images are randomly horizontally flipped and the following color transformations are applied:

- Grayscale with an application probability of $0.3$.
- Brightness with a maximum brightness difference of $0.1255$.
- Saturation with a saturation factor randomly picked in the interval $\mathopen{}\mathclose{{}\left[0.5,1.5}\right]$.
- Hue with a hue adjustment factor randomly picked in the interval $\mathopen{}\mathclose{{}\left[-0.2,0.2}\right]$.

We train for $7500$ steps with batch size $256$, weight decay $0.0005,$ and learning rate $0.16$ (scaled linearly from the setup of [^83] to account for the bigger batch size).

### D.7 Further comparisons on PASCAL and NYU v2 Depth

For completeness, Table 11 and 12 extends Table 4 with other published baselines which use comparable networks. We see that in almost all settings, BYOL outperforms these baselines, even when those baselines use more data or deeper models. One notable exception is RMS error for NYU Depth prediction, which is a metric that’s sensitive to outliers. The reason for this is unclear, but one possibility is that the network is producing higher-variance predictions due to being more confident about a test-set scene’s similarities with those in the training set.

| Method | AP <sub>50</sub> | mIoU |
| --- | --- | --- |
| Supervised-IN [^9] | $74.4$ | $74.4$ |
| RelPos [^23], by [^40] <sup>∗</sup> | $66.8$ | \- |
| Multi-task [^40] <sup>∗</sup> | $70.5$ | \- |
| LocalAgg [^98] | $69.1$ | \- |
| MoCo [^9] | $74.9$ | $72.5$ |
| MoCo + IG-1B [^9] | $75.6$ | $73.6$ |
| CPC [^32] <sup>∗∗</sup> | $76.6$ | \- |
| SimCLR (repro) | $75.2$ | $75.2$ |
| BYOL (ours) | $\bf{77.5}$ | $\bf{76.3}$ |

Table 11: Transfer results in semantic segmentation and object detection.

<sup>∗</sup> uses a larger model (ResNet- $101$). <sup>∗∗</sup> uses an even larger model (ResNet- $161$).

<table><thead><tr><th></th><th colspan="3">Higher better</th><th colspan="2">Lower better</th></tr><tr><th>Method</th><th>pct. <math><semantics><mrow><mo><</mo> <mn>1.25</mn></mrow> <apply><csymbol>absent</csymbol> <cn>1.25</cn></apply> <annotation>\!<\!1.25</annotation></semantics></math></th><th>pct. <math><semantics><mrow><mo><</mo> <msup><mn>1.25</mn> <mn>2</mn></msup></mrow> <apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>1.25</cn> <cn>2</cn></apply></apply> <annotation>\!<\!1.25^{2}</annotation></semantics></math></th><th>pct. <math><semantics><mrow><mo><</mo> <msup><mn>1.25</mn> <mn>3</mn></msup></mrow> <apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>1.25</cn> <cn>3</cn></apply></apply> <annotation>\!<\!1.25^{3}</annotation></semantics></math></th><th>rms</th><th>rel</th></tr><tr><th>Supervised-IN <sup><a href="#fn:83">83</a></sup></th><th><math><semantics><mn>81.1</mn> <cn>81.1</cn> <annotation>81.1</annotation></semantics></math></th><th><math><semantics><mn>95.3</mn> <cn>95.3</cn> <annotation>95.3</annotation></semantics></math></th><th><math><semantics><mn>98.8</mn> <cn>98.8</cn> <annotation>98.8</annotation></semantics></math></th><th><math><semantics><mn>0.573</mn> <cn>0.573</cn> <annotation>0.573</annotation></semantics></math></th><th><math><semantics><mn>0.127</mn> <cn>0.127</cn> <annotation>\bf{0.127}</annotation></semantics></math></th></tr></thead><tbody><tr><th>RelPos <sup><a href="#fn:23">23</a></sup>, by <sup><a href="#fn:40">40</a></sup> <sup>∗</sup></th><td><math><semantics><mn>80.6</mn> <cn>80.6</cn> <annotation>80.6</annotation></semantics></math></td><td><math><semantics><mn>94.7</mn> <cn>94.7</cn> <annotation>94.7</annotation></semantics></math></td><td><math><semantics><mn>98.3</mn> <cn>98.3</cn> <annotation>98.3</annotation></semantics></math></td><td><math><semantics><mn>0.399</mn> <cn>0.399</cn> <annotation>\bf{0.399}</annotation></semantics></math></td><td><math><semantics><mn>0.146</mn> <cn>0.146</cn> <annotation>0.146</annotation></semantics></math></td></tr><tr><th>Color <sup><a href="#fn:41">41</a></sup>, by <sup><a href="#fn:40">40</a></sup> <sup>∗</sup></th><td><math><semantics><mn>76.8</mn> <cn>76.8</cn> <annotation>76.8</annotation></semantics></math></td><td><math><semantics><mn>93.5</mn> <cn>93.5</cn> <annotation>93.5</annotation></semantics></math></td><td><math><semantics><mn>97.7</mn> <cn>97.7</cn> <annotation>97.7</annotation></semantics></math></td><td><math><semantics><mn>0.444</mn> <cn>0.444</cn> <annotation>0.444</annotation></semantics></math></td><td><math><semantics><mn>0.164</mn> <cn>0.164</cn> <annotation>0.164</annotation></semantics></math></td></tr><tr><th>Exemplar <sup><a href="#fn:46">46</a></sup> <sup><a href="#fn:40">40</a></sup> <sup>∗</sup></th><td><math><semantics><mn>71.3</mn> <cn>71.3</cn> <annotation>71.3</annotation></semantics></math></td><td><math><semantics><mn>90.6</mn> <cn>90.6</cn> <annotation>90.6</annotation></semantics></math></td><td><math><semantics><mn>96.5</mn> <cn>96.5</cn> <annotation>96.5</annotation></semantics></math></td><td><math><semantics><mn>0.513</mn> <cn>0.513</cn> <annotation>0.513</annotation></semantics></math></td><td><math><semantics><mn>0.191</mn> <cn>0.191</cn> <annotation>0.191</annotation></semantics></math></td></tr><tr><th>Mot. Seg. <sup><a href="#fn:99">99</a></sup>, by <sup><a href="#fn:40">40</a></sup> <sup>∗</sup></th><td><math><semantics><mn>74.2</mn> <cn>74.2</cn> <annotation>74.2</annotation></semantics></math></td><td><math><semantics><mn>92.4</mn> <cn>92.4</cn> <annotation>92.4</annotation></semantics></math></td><td><math><semantics><mn>97.4</mn> <cn>97.4</cn> <annotation>97.4</annotation></semantics></math></td><td><math><semantics><mn>0.473</mn> <cn>0.473</cn> <annotation>0.473</annotation></semantics></math></td><td><math><semantics><mn>0.177</mn> <cn>0.177</cn> <annotation>0.177</annotation></semantics></math></td></tr><tr><th>Multi-task <sup><a href="#fn:40">40</a></sup> <sup>∗</sup></th><td><math><semantics><mn>79.3</mn> <cn>79.3</cn> <annotation>79.3</annotation></semantics></math></td><td><math><semantics><mn>94.2</mn> <cn>94.2</cn> <annotation>94.2</annotation></semantics></math></td><td><math><semantics><mn>98.1</mn> <cn>98.1</cn> <annotation>98.1</annotation></semantics></math></td><td><math><semantics><mn>0.422</mn> <cn>0.422</cn> <annotation>0.422</annotation></semantics></math></td><td><math><semantics><mn>0.152</mn> <cn>0.152</cn> <annotation>0.152</annotation></semantics></math></td></tr><tr><th>SimCLR (repro)</th><td><math><semantics><mn>83.3</mn> <cn>83.3</cn> <annotation>83.3</annotation></semantics></math></td><td><math><semantics><mn>96.5</mn> <cn>96.5</cn> <annotation>96.5</annotation></semantics></math></td><td><math><semantics><mn>99.1</mn> <cn>99.1</cn> <annotation>99.1</annotation></semantics></math></td><td><math><semantics><mn>0.557</mn> <cn>0.557</cn> <annotation>0.557</annotation></semantics></math></td><td><math><semantics><mn>0.134</mn> <cn>0.134</cn> <annotation>0.134</annotation></semantics></math></td></tr><tr><th>BYOL (ours)</th><td><math><semantics><mn>84.6</mn> <cn>84.6</cn> <annotation>\bf{84.6}</annotation></semantics></math></td><td><math><semantics><mn>96.7</mn> <cn>96.7</cn> <annotation>\bf{96.7}</annotation></semantics></math></td><td><math><semantics><mn>99.1</mn> <cn>99.1</cn> <annotation>\bf{99.1}</annotation></semantics></math></td><td><math><semantics><mn>0.541</mn> <cn>0.541</cn> <annotation>0.541</annotation></semantics></math></td><td><math><semantics><mn>0.129</mn> <cn>0.129</cn> <annotation>0.129</annotation></semantics></math></td></tr></tbody></table>

Table 12: Transfer results on NYU v2 depth estimation.

## Appendix E Pretraining on Places 365

To ascertain that BYOL learns good representations on other datasets, we applied our representation learning protocol on the scene recognition dataset Places $365$ -Standard [^73] before performing linear evaluation. This dataset contains $1.80$ million training images and $36500$ validation images with labels, making it roughly similar to ImageNet in scale. We reuse the *exact* same parameters as in Section 4 and train the representation for $1000$ epochs, using BYOL and our SimCLR reproduction. Results for the linear evaluation setup (using the protocol of Section C.1 for ImageNet and Places365, and that of Appendix D on other datasets) are reported in Table 13.

Interestingly, the representation trained by using BYOL on Places365 (BYOL-PL) consistently outperforms that of SimCLR on the same dataset, but underperforms the BYOL representation trained on ImageNet (BYOL-IN) on all tasks except Places365 and SUN397 [^79], another scene understanding dataset. Interestingly, all three unsupervised representation learning methods achieve a relatively high performance on the Places365 task; for comparison, reference [^73] (in its linked repository) reports a top-1 accuracy of $55.2\%$ for a ResNet- $50$ v2 trained from scratch using labels on this dataset.

| Method | Places365 | ImageNet | Food101 | CIFAR10 | CIFAR100 | Birdsnap | SUN397 | Cars | Aircraft | DTD | Pets | Caltech-101 | Flowers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BYOL-IN | $51.0$ | $74.3$ | $75.3$ | $91.3$ | $78.4$ | $57.3$ | $62.6$ | $67.2$ | $60.6$ | $76.5$ | $90.4$ | $94.3$ | $96.1$ |
| BYOL-PL | $\bf{53.2}$ | $\bf{58.5}$ | $\bf{64.7}$ | $\bf{84.5}$ | $\bf{66.1}$ | $\bf{28.8}$ | $\bf{64.2}$ | $\bf{55.6}$ | $\bf{55.9}$ | $\bf{68.5}$ | $\bf{66.1}$ | $\bf{84.3}$ | $\bf{90.0}$ |
| SimCLR-PL | $53.0$ | $56.5$ | $61.7$ | $80.8$ | $61.1$ | $21.2$ | $62.5$ | $40.1$ | $44.3$ | $64.3$ | $59.4$ | $77.1$ | $85.9$ |

Table 13: Transfer learning results (linear evaluation, ResNet- $50$) from Places365 (PL). For comparison purposes, we also report the results from BYOL trained on ImageNet (BYOL-IN).

## Appendix F Additional ablation results

To extend on the above results, we provide additional ablations obtained using the same experimental setup as in Section 5, *i.e.,* 300 epochs, averaged over $3$ seeds with the initial learning rate set to $0.3$, the batch size to $4096$, the weight decay to $10^{-6}$ and the base target decay rate $\tau_{\text{base}}$ to $0.99$ unless specified otherwise. Confidence intervals correspond to the half-difference between the maximum and minimum score of these seeds; we omit them for half-differences lower than $0.25$ accuracy points.

### F.1 Architecture settings

Table 14(a) shows the influence of projector and predictor architecture on BYOL. We examine the effect of different depths for both the projector and predictor, as well as the effect of the projection size. We do not apply a ReLU activation nor a batch normalization on the final linear layer of our MLPs such that a depth of $1$ corresponds to a linear layer. Using the default projector and predictor of depth $2$ yields the best performance.

| Proj. $g_{\theta}$   depth | Pred. $q_{\theta}$   depth | Top- $1$ | Top- $5$ |
| --- | --- | --- | --- |
| $1$ | $1$ | $61.9$ | $86.0$ |
|  | $2$ | $65.0$ | $86.8$ |
|  | $3$ | $65.7$ | $86.8$ |
| $2$ | $1$ | $71.5$ | $90.7$ |
|  | $2$ | $\bf{72.5}$ | $\bf{90.8}$ |
|  | $3$ | $71.4$ | $90.4$ |
| $3$ | $1$ | $71.4$ | $90.4$ |
|  | $2$ | $72.1$ | $90.5$ |
|  | $3$ | $72.1$ | $90.5$ |

(a) Projector and predictor depth (i.e. the number of Linear layers).

Table 14: Effect of architectural settings where top- $1$ and top- $5$ accuracies are reported in %.

Projector  $g_{\theta}$  
output dim Top- $1$ Top- $5$ $16$ $69.9{\scriptstyle\pm 0.3}$ $89.9$ $32$ $71.3$ $90.6$ $64$ $72.2$ $90.9$ $128$ $72.5$ $91.0$ $256$ $72.5$ $90.8$ $512$ $\bf{72.6}$ $\bf{91.0}$ (b) Projection dimension.

Table 15(a) shows the influence of the initial learning rate on BYOL. Note that the optimal value depends on the number of training epochs. Table 15(b) displays the influence of the weight decay on BYOL.

| Learning |  |  |
| --- | --- | --- |
| rate | Top- $1$ | Top- $5$ |
| $0.01$ | $34.8{\scriptstyle\pm 3.0}$ | $60.8{\scriptstyle\pm 3.2}$ |
| $0.1$ | $65.0$ | $87.0$ |
| $0.2$ | $71.7$ | $90.6$ |
| $0.3$ | $\bf{72.5}$ | $\bf{90.8}$ |
| $0.4$ | $72.3$ | $90.6$ |
| $0.5$ | $71.5$ | $90.1$ |
| $1$ | $69.4$ | $89.2$ |

(a) Base learning rate.

| Weight decay   coefficient | Top- $1$ | Top- $5$ |
| --- | --- | --- |
| $1\cdot 10^{-7}$ | $72.1$ | $90.4$ |
| $5\cdot 10^{-7}$ | $\bf{72.6}$ | $\bf{91.0}$ |
| $1\cdot 10^{-6}$ | $72.5$ | $90.8$ |
| $5\cdot 10^{-6}$ | $71.0{\scriptstyle\pm 0.3}$ | $90.0$ |
| $1\cdot 10^{-5}$ | $69.6{\scriptstyle\pm 0.4}$ | $89.3$ |

(b) Weight decay.

Table 15: Effect of learning rate and weight decay. We note that BYOL’s performance is quite robust within a range of hyperparameters. We also observe that setting the weight decay to zero may lead to unstable results (as in SimCLR).

### F.2 Batch size

We run a sweep over the batch size for both BYOL and our reproduction of SimCLR. As explained in Section 5, when reducing the batch size by a factor $N$, we average gradients over $N$ consecutive steps and update the target network once every $N$ steps. We report in Table 16, the performance of both our reproduction of SimCLR and BYOL for batch sizes between $4096$ (BYOL and SimCLR default) down to $64$. We observe that the performance of SimCLR deteriorates faster than the one of BYOL which stays mostly constant for batch sizes larger than $256$. We believe that the performance at batch size $256$ could match the performance of the large $4096$ batch size with proper parameter tuning when accumulating the gradient. We think that the drop in performance at batch size $64$ in table 16 is mainly related to the ill behaviour of batch normalization at low batch sizes [^100].

<table><thead><tr><th>Batch</th><th colspan="2">Top- <math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></th><th colspan="2">Top- <math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></th></tr><tr><th>size</th><th>BYOL (ours)</th><th>SimCLR (repro)</th><th>BYOL (ours)</th><th>SimCLR (repro)</th></tr></thead><tbody><tr><th><math><semantics><mn>4096</mn> <cn>4096</cn> <annotation>4096</annotation></semantics></math></th><td><math><semantics><mn>72.5</mn> <cn>72.5</cn> <annotation>\bf{72.5}</annotation></semantics></math></td><td><math><semantics><mn>67.9</mn> <cn>67.9</cn> <annotation>67.9</annotation></semantics></math></td><td><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>\bf{90.8}</annotation></semantics></math></td><td><math><semantics><mn>88.5</mn> <cn>88.5</cn> <annotation>88.5</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>2048</mn> <cn>2048</cn> <annotation>2048</annotation></semantics></math></th><td><math><semantics><mn>72.4</mn> <cn>72.4</cn> <annotation>72.4</annotation></semantics></math></td><td><math><semantics><mn>67.8</mn> <cn>67.8</cn> <annotation>67.8</annotation></semantics></math></td><td><math><semantics><mn>90.7</mn> <cn>90.7</cn> <annotation>90.7</annotation></semantics></math></td><td><math><semantics><mn>88.5</mn> <cn>88.5</cn> <annotation>88.5</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>1024</mn> <cn>1024</cn> <annotation>1024</annotation></semantics></math></th><td><math><semantics><mn>72.2</mn> <cn>72.2</cn> <annotation>72.2</annotation></semantics></math></td><td><math><semantics><mn>67.4</mn> <cn>67.4</cn> <annotation>67.4</annotation></semantics></math></td><td><math><semantics><mn>90.7</mn> <cn>90.7</cn> <annotation>90.7</annotation></semantics></math></td><td><math><semantics><mn>88.1</mn> <cn>88.1</cn> <annotation>88.1</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>512</mn> <cn>512</cn> <annotation>512</annotation></semantics></math></th><td><math><semantics><mn>72.2</mn> <cn>72.2</cn> <annotation>72.2</annotation></semantics></math></td><td><math><semantics><mn>66.5</mn> <cn>66.5</cn> <annotation>66.5</annotation></semantics></math></td><td><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>90.8</annotation></semantics></math></td><td><math><semantics><mn>87.6</mn> <cn>87.6</cn> <annotation>87.6</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>256</mn> <cn>256</cn> <annotation>256</annotation></semantics></math></th><td><math><semantics><mn>71.8</mn> <cn>71.8</cn> <annotation>71.8</annotation></semantics></math></td><td><math><semantics><mrow><mn>64.3</mn> <mo>±</mo> <mn>2.1</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>64.3</cn> <cn>2.1</cn></apply> <annotation>64.3{\scriptstyle\pm 2.1}</annotation></semantics></math></td><td><math><semantics><mn>90.7</mn> <cn>90.7</cn> <annotation>90.7</annotation></semantics></math></td><td><math><semantics><mrow><mn>86.3</mn> <mo>±</mo> <mn>1.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>86.3</cn> <cn>1.0</cn></apply> <annotation>86.3{\scriptstyle\pm 1.0}</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>128</mn> <cn>128</cn> <annotation>128</annotation></semantics></math></th><td><math><semantics><mrow><mn>69.6</mn> <mo>±</mo> <mn>0.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>69.6</cn> <cn>0.5</cn></apply> <annotation>69.6{\scriptstyle\pm 0.5}</annotation></semantics></math></td><td><math><semantics><mn>63.6</mn> <cn>63.6</cn> <annotation>63.6</annotation></semantics></math></td><td><math><semantics><mn>89.6</mn> <cn>89.6</cn> <annotation>89.6</annotation></semantics></math></td><td><math><semantics><mn>85.9</mn> <cn>85.9</cn> <annotation>85.9</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>64</mn> <cn>64</cn> <annotation>64</annotation></semantics></math></th><td><math><semantics><mrow><mn>59.7</mn> <mo>±</mo> <mn>1.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>59.7</cn> <cn>1.5</cn></apply> <annotation>59.7{\scriptstyle\pm 1.5}</annotation></semantics></math></td><td><math><semantics><mrow><mn>59.2</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>59.2</cn> <cn>2.9</cn></apply> <annotation>59.2{\scriptstyle\pm 2.9}</annotation></semantics></math></td><td><math><semantics><mrow><mn>83.2</mn> <mo>±</mo> <mn>1.2</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>83.2</cn> <cn>1.2</cn></apply> <annotation>83.2{\scriptstyle\pm 1.2}</annotation></semantics></math></td><td><math><semantics><mrow><mn>83.0</mn> <mo>±</mo> <mn>1.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>83.0</cn> <cn>1.9</cn></apply> <annotation>83.0{\scriptstyle\pm 1.9}</annotation></semantics></math></td></tr></tbody></table>

Table 16: Influence of the batch size.

### F.3 Image augmentations

Table 17 compares the impact of individual image transformations on BYOL and SimCLR. BYOL is more resilient to changes of image augmentations across the board. For completeness, we also include an ablation with symmetric parameters across both views; for this ablation, we use a Gaussian blurring w.p. of $0.5$ and a solarization w.p. of $0.2$ for both $\mathcal{T}$ and $\mathcal{T}^{\prime}$, and recover very similar results compared to our baseline choice of parameters.

<table><thead><tr><th></th><th colspan="2">Top- <math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></th><th colspan="2">Top- <math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></th></tr><tr><th>Image augmentation</th><th>BYOL (ours)</th><th>SimCLR (repro)</th><th>BYOL (ours)</th><th>SimCLR (repro)</th></tr><tr><th>Baseline</th><th><math><semantics><mn>72.5</mn> <cn>72.5</cn> <annotation>\bf{72.5}</annotation></semantics></math></th><th><math><semantics><mn>67.9</mn> <cn>67.9</cn> <annotation>67.9</annotation></semantics></math></th><th><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>\bf{90.8}</annotation></semantics></math></th><th><math><semantics><mn>88.5</mn> <cn>88.5</cn> <annotation>88.5</annotation></semantics></math></th></tr></thead><tbody><tr><th>Remove flip</th><td><math><semantics><mn>71.9</mn> <cn>71.9</cn> <annotation>71.9</annotation></semantics></math></td><td><math><semantics><mn>67.3</mn> <cn>67.3</cn> <annotation>67.3</annotation></semantics></math></td><td><math><semantics><mn>90.6</mn> <cn>90.6</cn> <annotation>90.6</annotation></semantics></math></td><td><math><semantics><mn>88.2</mn> <cn>88.2</cn> <annotation>88.2</annotation></semantics></math></td></tr><tr><th>Remove blur</th><td><math><semantics><mn>71.2</mn> <cn>71.2</cn> <annotation>71.2</annotation></semantics></math></td><td><math><semantics><mn>65.2</mn> <cn>65.2</cn> <annotation>65.2</annotation></semantics></math></td><td><math><semantics><mn>90.3</mn> <cn>90.3</cn> <annotation>90.3</annotation></semantics></math></td><td><math><semantics><mn>86.6</mn> <cn>86.6</cn> <annotation>86.6</annotation></semantics></math></td></tr><tr><th>Remove color (jittering and grayscale)</th><td><math><semantics><mrow><mn>63.4</mn> <mo>±</mo> <mn>0.7</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>63.4</cn> <cn>0.7</cn></apply> <annotation>63.4{\scriptstyle\pm 0.7}</annotation></semantics></math></td><td><math><semantics><mn>45.7</mn> <cn>45.7</cn> <annotation>45.7</annotation></semantics></math></td><td><math><semantics><mrow><mn>85.3</mn> <mo>±</mo> <mn>0.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>85.3</cn> <cn>0.5</cn></apply> <annotation>85.3{\scriptstyle\pm 0.5}</annotation></semantics></math></td><td><math><semantics><mn>70.6</mn> <cn>70.6</cn> <annotation>70.6</annotation></semantics></math></td></tr><tr><th>Remove color jittering</th><td><math><semantics><mn>71.8</mn> <cn>71.8</cn> <annotation>71.8</annotation></semantics></math></td><td><math><semantics><mn>63.7</mn> <cn>63.7</cn> <annotation>63.7</annotation></semantics></math></td><td><math><semantics><mn>90.7</mn> <cn>90.7</cn> <annotation>90.7</annotation></semantics></math></td><td><math><semantics><mn>85.9</mn> <cn>85.9</cn> <annotation>85.9</annotation></semantics></math></td></tr><tr><th>Remove grayscale</th><td><math><semantics><mn>70.3</mn> <cn>70.3</cn> <annotation>70.3</annotation></semantics></math></td><td><math><semantics><mn>61.9</mn> <cn>61.9</cn> <annotation>61.9</annotation></semantics></math></td><td><math><semantics><mn>89.8</mn> <cn>89.8</cn> <annotation>89.8</annotation></semantics></math></td><td><math><semantics><mn>84.1</mn> <cn>84.1</cn> <annotation>84.1</annotation></semantics></math></td></tr><tr><th>Remove blur in <math><semantics><msup><mi>𝒯</mi> <mo>′</mo></msup> <apply><csymbol>superscript</csymbol> <ci>𝒯</ci> <ci>′</ci></apply> <annotation>\mathcal{T}^{\prime}</annotation></semantics></math></th><td><math><semantics><mn>72.4</mn> <cn>72.4</cn> <annotation>72.4</annotation></semantics></math></td><td><math><semantics><mn>67.5</mn> <cn>67.5</cn> <annotation>67.5</annotation></semantics></math></td><td><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>90.8</annotation></semantics></math></td><td><math><semantics><mn>88.4</mn> <cn>88.4</cn> <annotation>88.4</annotation></semantics></math></td></tr><tr><th>Remove solarize in <math><semantics><msup><mi>𝒯</mi> <mo>′</mo></msup> <apply><csymbol>superscript</csymbol> <ci>𝒯</ci> <ci>′</ci></apply> <annotation>\mathcal{T}^{\prime}</annotation></semantics></math></th><td><math><semantics><mn>72.3</mn> <cn>72.3</cn> <annotation>72.3</annotation></semantics></math></td><td><math><semantics><mn>67.7</mn> <cn>67.7</cn> <annotation>67.7</annotation></semantics></math></td><td><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>90.8</annotation></semantics></math></td><td><math><semantics><mn>88.2</mn> <cn>88.2</cn> <annotation>88.2</annotation></semantics></math></td></tr><tr><th>Remove blur and solarize in <math><semantics><msup><mi>𝒯</mi> <mo>′</mo></msup> <apply><csymbol>superscript</csymbol> <ci>𝒯</ci> <ci>′</ci></apply> <annotation>\mathcal{T}^{\prime}</annotation></semantics></math></th><td><math><semantics><mn>72.2</mn> <cn>72.2</cn> <annotation>72.2</annotation></semantics></math></td><td><math><semantics><mn>67.4</mn> <cn>67.4</cn> <annotation>67.4</annotation></semantics></math></td><td><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>90.8</annotation></semantics></math></td><td><math><semantics><mn>88.1</mn> <cn>88.1</cn> <annotation>88.1</annotation></semantics></math></td></tr><tr><th>Symmetric blurring/solarization</th><td><math><semantics><mn>72.5</mn> <cn>72.5</cn> <annotation>72.5</annotation></semantics></math></td><td><math><semantics><mn>68.1</mn> <cn>68.1</cn> <annotation>68.1</annotation></semantics></math></td><td><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>90.8</annotation></semantics></math></td><td><math><semantics><mn>88.4</mn> <cn>88.4</cn> <annotation>88.4</annotation></semantics></math></td></tr><tr><th>Crop only</th><td><math><semantics><mrow><mn>59.4</mn> <mo>±</mo> <mn>0.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>59.4</cn> <cn>0.3</cn></apply> <annotation>59.4{\scriptstyle\pm 0.3}</annotation></semantics></math></td><td><math><semantics><mrow><mn>40.3</mn> <mo>±</mo> <mn>0.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>40.3</cn> <cn>0.3</cn></apply> <annotation>40.3{\scriptstyle\pm 0.3}</annotation></semantics></math></td><td><math><semantics><mn>82.4</mn> <cn>82.4</cn> <annotation>82.4</annotation></semantics></math></td><td><math><semantics><mrow><mn>64.8</mn> <mo>±</mo> <mn>0.4</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>64.8</cn> <cn>0.4</cn></apply> <annotation>64.8{\scriptstyle\pm 0.4}</annotation></semantics></math></td></tr><tr><th>Crop and flip only</th><td><math><semantics><mrow><mn>60.1</mn> <mo>±</mo> <mn>0.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>60.1</cn> <cn>0.3</cn></apply> <annotation>60.1{\scriptstyle\pm 0.3}</annotation></semantics></math></td><td><math><semantics><mn>40.2</mn> <cn>40.2</cn> <annotation>40.2</annotation></semantics></math></td><td><math><semantics><mrow><mn>83.0</mn> <mo>±</mo> <mn>0.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>83.0</cn> <cn>0.3</cn></apply> <annotation>83.0{\scriptstyle\pm 0.3}</annotation></semantics></math></td><td><math><semantics><mn>64.8</mn> <cn>64.8</cn> <annotation>64.8</annotation></semantics></math></td></tr><tr><th>Crop and color only</th><td><math><semantics><mn>70.7</mn> <cn>70.7</cn> <annotation>70.7</annotation></semantics></math></td><td><math><semantics><mn>64.2</mn> <cn>64.2</cn> <annotation>64.2</annotation></semantics></math></td><td><math><semantics><mn>90.0</mn> <cn>90.0</cn> <annotation>90.0</annotation></semantics></math></td><td><math><semantics><mn>86.2</mn> <cn>86.2</cn> <annotation>86.2</annotation></semantics></math></td></tr><tr><th>Crop and blur only</th><td><math><semantics><mrow><mn>61.1</mn> <mo>±</mo> <mn>0.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>61.1</cn> <cn>0.3</cn></apply> <annotation>61.1{\scriptstyle\pm 0.3}</annotation></semantics></math></td><td><math><semantics><mn>41.7</mn> <cn>41.7</cn> <annotation>41.7</annotation></semantics></math></td><td><math><semantics><mn>83.9</mn> <cn>83.9</cn> <annotation>83.9</annotation></semantics></math></td><td><math><semantics><mn>66.4</mn> <cn>66.4</cn> <annotation>66.4</annotation></semantics></math></td></tr></tbody></table>

Table 17: Ablation on image transformations.

<table><tbody><tr><th>Loss weight <math><semantics><mi>β</mi> <ci>𝛽</ci> <annotation>\beta</annotation></semantics></math></th><td>Temperature <math><semantics><mi>α</mi> <ci>𝛼</ci> <annotation>\alpha</annotation></semantics></math></td><td>Top- <math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></td><td>Top- <math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>0</mn> <cn>0</cn></semantics></math></th><td><math><semantics><mn>0.1</mn> <cn>0.1</cn> <annotation>0.1</annotation></semantics></math></td><td><math><semantics><mn>72.5</mn> <cn>72.5</cn> <annotation>72.5</annotation></semantics></math></td><td><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>90.8</annotation></semantics></math></td></tr><tr><th rowspan="6"><math><semantics><mn>0.1</mn> <cn>0.1</cn> <annotation>0.1</annotation></semantics></math></th><td><math><semantics><mn>0.01</mn> <cn>0.01</cn> <annotation>0.01</annotation></semantics></math></td><td><math><semantics><mn>72.2</mn> <cn>72.2</cn> <annotation>72.2</annotation></semantics></math></td><td><math><semantics><mn>90.7</mn> <cn>90.7</cn> <annotation>90.7</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>0.1</mn> <cn>0.1</cn> <annotation>0.1</annotation></semantics></math></td><td><math><semantics><mn>72.4</mn> <cn>72.4</cn> <annotation>72.4</annotation></semantics></math></td><td><math><semantics><mn>90.9</mn> <cn>90.9</cn> <annotation>90.9</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>0.3</mn> <cn>0.3</cn> <annotation>0.3</annotation></semantics></math></td><td><math><semantics><mn>72.7</mn> <cn>72.7</cn> <annotation>\bf{72.7}</annotation></semantics></math></td><td><math><semantics><mn>91.0</mn> <cn>91.0</cn> <annotation>91.0</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></td><td><math><semantics><mn>72.6</mn> <cn>72.6</cn> <annotation>72.6</annotation></semantics></math></td><td><math><semantics><mn>90.9</mn> <cn>90.9</cn> <annotation>90.9</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>3</mn> <cn>3</cn> <annotation>3</annotation></semantics></math></td><td><math><semantics><mn>72.5</mn> <cn>72.5</cn> <annotation>72.5</annotation></semantics></math></td><td><math><semantics><mn>90.9</mn> <cn>90.9</cn> <annotation>90.9</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>10</mn> <cn>10</cn> <annotation>10</annotation></semantics></math></td><td><math><semantics><mn>72.5</mn> <cn>72.5</cn> <annotation>72.5</annotation></semantics></math></td><td><math><semantics><mn>90.9</mn> <cn>90.9</cn> <annotation>90.9</annotation></semantics></math></td></tr><tr><th rowspan="6"><math><semantics><mn>0.5</mn> <cn>0.5</cn> <annotation>0.5</annotation></semantics></math></th><td><math><semantics><mn>0.01</mn> <cn>0.01</cn> <annotation>0.01</annotation></semantics></math></td><td><math><semantics><mn>70.9</mn> <cn>70.9</cn> <annotation>70.9</annotation></semantics></math></td><td><math><semantics><mn>90.2</mn> <cn>90.2</cn> <annotation>90.2</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>0.1</mn> <cn>0.1</cn> <annotation>0.1</annotation></semantics></math></td><td><math><semantics><mn>72.0</mn> <cn>72.0</cn> <annotation>72.0</annotation></semantics></math></td><td><math><semantics><mn>90.8</mn> <cn>90.8</cn> <annotation>90.8</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>0.3</mn> <cn>0.3</cn> <annotation>0.3</annotation></semantics></math></td><td><math><semantics><mn>72.7</mn> <cn>72.7</cn> <annotation>\bf{72.7}</annotation></semantics></math></td><td><math><semantics><mn>91.2</mn> <cn>91.2</cn> <annotation>\bf{91.2}</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></td><td><math><semantics><mn>72.7</mn> <cn>72.7</cn> <annotation>72.7</annotation></semantics></math></td><td><math><semantics><mn>91.1</mn> <cn>91.1</cn> <annotation>91.1</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>3</mn> <cn>3</cn> <annotation>3</annotation></semantics></math></td><td><math><semantics><mn>72.6</mn> <cn>72.6</cn> <annotation>72.6</annotation></semantics></math></td><td><math><semantics><mn>91.1</mn> <cn>91.1</cn> <annotation>91.1</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>10</mn> <cn>10</cn> <annotation>10</annotation></semantics></math></td><td><math><semantics><mn>72.5</mn> <cn>72.5</cn> <annotation>72.5</annotation></semantics></math></td><td><math><semantics><mn>91.0</mn> <cn>91.0</cn> <annotation>91.0</annotation></semantics></math></td></tr><tr><th rowspan="6"><math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></th><td><math><semantics><mn>0.01</mn> <cn>0.01</cn> <annotation>0.01</annotation></semantics></math></td><td><math><semantics><mrow><mn>53.9</mn> <mo>±</mo> <mn>0.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>53.9</cn> <cn>0.5</cn></apply> <annotation>53.9{\scriptstyle\pm 0.5}</annotation></semantics></math></td><td><math><semantics><mrow><mn>77.5</mn> <mo>±</mo> <mn>0.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>77.5</cn> <cn>0.5</cn></apply> <annotation>77.5{\scriptstyle\pm 0.5}</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>0.1</mn> <cn>0.1</cn> <annotation>0.1</annotation></semantics></math></td><td><math><semantics><mn>70.9</mn> <cn>70.9</cn> <annotation>70.9</annotation></semantics></math></td><td><math><semantics><mn>90.3</mn> <cn>90.3</cn> <annotation>90.3</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>0.3</mn> <cn>0.3</cn> <annotation>0.3</annotation></semantics></math></td><td><math><semantics><mn>72.7</mn> <cn>72.7</cn> <annotation>\bf{72.7}</annotation></semantics></math></td><td><math><semantics><mn>91.1</mn> <cn>91.1</cn> <annotation>91.1</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></td><td><math><semantics><mn>72.7</mn> <cn>72.7</cn> <annotation>\bf{72.7}</annotation></semantics></math></td><td><math><semantics><mn>91.1</mn> <cn>91.1</cn> <annotation>91.1</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>3</mn> <cn>3</cn> <annotation>3</annotation></semantics></math></td><td><math><semantics><mn>72.6</mn> <cn>72.6</cn> <annotation>72.6</annotation></semantics></math></td><td><math><semantics><mn>91.0</mn> <cn>91.0</cn> <annotation>91.0</annotation></semantics></math></td></tr><tr><td><math><semantics><mn>10</mn> <cn>10</cn> <annotation>10</annotation></semantics></math></td><td><math><semantics><mn>72.6</mn> <cn>72.6</cn> <annotation>72.6</annotation></semantics></math></td><td><math><semantics><mn>91.1</mn> <cn>91.1</cn> <annotation>91.1</annotation></semantics></math></td></tr></tbody></table>

Table 18: Top-1 accuracy in % under linear evaluation protocol at 300 epochs of sweep over the temperature $\alpha$ and the dispersion term weight $\beta$ when using a predictor and a target network.

### F.4 Details on the relation to contrastive methods

As mentioned in Section 5, the BYOL loss Equation 2 can be derived from the InfoNCE loss

$$
\text{InfoNCE}^{\alpha,\beta}_{{\theta}}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\frac{2}{B}\sum_{i=1}^{B}S_{\theta}(v_{i},v^{\prime}_{i})\!-\!\frac{2\alpha\cdot\beta}{B}\sum_{i=1}^{B}\ln\mathopen{}\mathclose{{}\left(\sum\limits_{j\neq i}\exp\frac{S_{\theta}(v_{i},v_{j})}{\alpha}+\sum\limits_{j}\exp\frac{S_{\theta}(v_{i},v^{\prime}_{j})}{\alpha}}\right)\mathbin{\raisebox{2.15277pt}{,}}
$$

with

$$
S_{\theta}(u_{1},u_{2})\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\frac{\langle\phi(u_{1}),\psi(u_{2})\rangle}{\|\phi(u_{1})\|_{2}\cdot\|\psi(u_{2})\|_{2}}\cdot
$$

The InfoNCE loss, introduced in [^10], can be found in factored form in [^84] as

$$
\text{InfoNCE}_{{\theta}}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\frac{1}{B}\sum_{i=1}^{B}\ln\cfrac{f(v_{i},v^{\prime}_{i})}{\frac{1}{B}\sum\limits_{j}\exp f(v_{i},v^{\prime}_{j})}\,\cdot
$$

As in SimCLR [^8] we also use negative examples given by $(v_{i},v_{j})_{j\neq i}$ to get

$$
\displaystyle\frac{1}{B}\sum_{i=1}^{B}\ln\cfrac{\exp f(v_{i},v^{\prime}_{i})}{\frac{1}{B}\sum\limits_{j\neq i}\exp f(v_{i},v_{j})+\frac{1}{B}\sum\limits_{j}\exp f(v_{i},v^{\prime}_{j})}
$$
 
$$
\displaystyle\qquad=\ln B+\frac{1}{B}\sum_{i=1}^{B}f(v_{i},v^{\prime}_{i})-\frac{1}{B}\sum_{i=1}^{B}\ln\mathopen{}\mathclose{{}\left(\sum\limits_{j\neq i}\exp f(v_{i},v_{j})+\sum\limits_{j}\exp f(v_{i},v^{\prime}_{j})}\right).
$$

To obtain Equation 8 from Equation 12, we subtract $\ln B$ (which is independent of ${\theta}$), multiply by $2\alpha$, take $f(x,y)=S_{\theta}(x,y)/\alpha$ and finally multiply the second (negative examples) term by $\beta$. Using $\beta=1$ and dividing by $2\alpha$ gets us back to the usual InfoNCE loss as used by SimCLR.

In our ablation in Table 5(b), we set the temperature $\alpha$ to its best value in the SimCLR setting (i.e., $\alpha=0.1$). With this value, setting $\beta$ to 1 (which adds negative examples), in the BYOL setting (i.e., with both a predictor and a target network) hurts the performances. In Table 18, we report results of a sweep over both the temperature $\alpha$ and the weight parameter $\beta$ with a predictor and a target network where BYOL corresponds to $\beta=0$. No run significantly outperforms BYOL and some values of $\alpha$ and $\beta$ hurt the performance. While the best temperature for SimCLR (without the target network and a predictor) is $0.1$, after adding a predictor and a target network the best temperature $\alpha$ is higher than $0.3$.

Using a target network in the loss has two effects: stopping the gradient through the prediction targets and stabilizing the targets with averaging. Stopping the gradient through the target change the objective while averaging makes the target stable and stale. In Table 5(b) we only shows results of the ablation when either using the online network as the prediction target (and flowing the gradient through it) or with a target network (both stopping the gradient into the prediction targets and computing the prediction targets with a moving average of the online network). We shown in Table 5(b) that using a target network is beneficial but it has two distinct effects we would like to understand from which effect the improvement comes from. We report in Table 19 the results already in Table 5(b) but also when the prediction target is computed with a stop gradient of the online network (the gradient does not flow into the prediction targets). This shows that making the prediction targets stable and stale is the main cause of the improvement rather than the change in the objective due to the stop gradient.

| Method | Predictor | Target parameters | $\beta$ | Top-1 |
| --- | --- | --- | --- | --- |
| BYOL | ✓ | $\xi$ | 0 | $\bf{72.5}$ |
|  | ✓ | $\xi$ | 1 | $70.9$ |
|  |  | $\xi$ | 1 | $70.7$ |
|  | ✓ | $\mathrm{sg}({\theta})$ | 1 | $70.2$ |
| SimCLR |  | ${\theta}$ | 1 | $69.4$ |
|  | ✓ | $\mathrm{sg}({\theta})$ | 1 | $70.1$ |
|  |  | $\mathrm{sg}({\theta})$ | 1 | $69.2$ |
|  | ✓ | ${\theta}$ | 1 | $69.0$ |
|  | ✓ | $\mathrm{sg}({\theta})$ | 0 | $5.5$ |
|  | ✓ | ${\theta}$ | 0 | $0.3$ |
|  |  | $\xi$ | 0 | $0.2$ |
|  |  | $\mathrm{sg}({\theta})$ | 0 | $0.1$ |
|  |  | ${\theta}$ | 0 | $0.1$ |

Table 19: Top-1 accuracy in %, under linear evaluation protocol at 300 epochs, of intermediate variants between BYOL and SimCLR (with caveats discussed in Section F.5). $\mathrm{sg}$ means stop gradient.

### F.5 SimCLR baseline of Section

The SimCLR baseline in Section 5 ($\beta=1$, without predictor nor target network) is slightly different from the original one in [^8]. First we multiply the original loss by $2\alpha$. For comparaison here is the original SimCLR loss,

$$
\displaystyle\text{InfoNCE}_{{\theta}}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\frac{1}{B}\sum_{i=1}^{B}\frac{S_{\theta}(v_{i},v^{\prime}_{i})}{\alpha}\!-\!\frac{1}{B}\sum_{i=1}^{B}\ln\mathopen{}\mathclose{{}\left(\sum\limits_{j\neq i}\exp\frac{S_{\theta}(v_{i},v_{j})}{\alpha}+\sum\limits_{j}\exp\frac{S_{\theta}(v_{i},v^{\prime}_{j})}{\alpha}}\right)\cdot
$$

Note that this multiplication by $2\alpha$ matters as the LARS optimizer is not completely invariant with respect to the scale of the loss. Indeed, LARS applies a preconditioning to gradient updates on all weights, except for biases and batch normalization parameters. Updates on preconditioned weights are invariant by multiplicative scaling of the loss. However, the bias and batch normalization parameter updates remain sensitive to multiplicative scaling of the loss.

We also increase the original SimCLR hidden and output size of the projector to respectively 4096 and 256. In our reproduction of SimCLR, these three combined changes improves the top-1 accuracy at 300 epochs from $67.9\%$ (without the changes) to $69.2\%$ (with the changes).

### F.6 Ablation on the normalization in the loss function

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2006.07733/assets/x9.png)

(a) Representation ℓ 2 subscript \\ell\_{2} -norm

| Normalization | Top-1 | Top-5 |
| --- | --- | --- |
| $\ell_{2}$ -norm | $\bf{72.5}$ | $\bf{90.8}$ |
| LayerNorm | $72.5{\scriptstyle\pm 0.4}$ | $90.1$ |
| No normalization | $67.4$ | $87.1$ |
| BatchNorm | $65.3$ | $85.3$ |

Table 20: Top-1 accuracy in % under linear evaluation protocol at 300 epochs for different normalizations in the loss.

BYOL minimizes a squared error between the $\ell_{2}$ -normalized prediction and target. We report results of BYOL at 300 epochs using different normalization function and no normalization at all. More precisely, given batch of prediction and targets in $\mathbb{R}^{d}$, $(p_{i},t_{i})_{i\leq B}$ with $B$ the batch size, BYOL uses the loss function $\frac{1}{B}\sum_{i=1}^{B}\|n_{\ell_{2}}(p_{i})-n_{\ell_{2}}(z_{i})\|_{2}^{2}$ with $n_{\ell_{2}}:x\rightarrow x/\|x\|_{2}$. We run BYOL with other normalization functions: non-trainable batch-normalization and layer-normalization and no normalization. We divide the batch normalization and layer normalization by $\sqrt{d}$ to have a consistent scale with the $\ell_{2}$ -normalization. We report results in Table 20 where $\ell_{2}$, LayerNorm, no normalization and BatchNorm respectively denote using $n_{\ell_{2}}$, $n_{\text{{B\hskip-0.70004ptN}}}$, $n_{\text{{L\hskip-0.70004ptN}}}$ and $n_{\text{{I\hskip-0.70004ptd}}}$ with

$$
\displaystyle{n_{\text{{B\hskip-0.70004ptN}}}^{j}}_{i}
$$
 
$$
\displaystyle:x\rightarrow\frac{x_{i}^{j}-\mu_{\text{{B\hskip-0.70004ptN}}}^{j}(x)}{\sigma_{\text{BN}}^{j}(x)\cdot\sqrt{d}}\mathbin{\raisebox{2.15277pt}{,}}\quad{n_{\text{{L\hskip-0.70004ptN}}}^{j}}_{i}:x\rightarrow\frac{x_{i}^{j}-{\mu_{\text{{L\hskip-0.70004ptN}}}}_{i}(x)}{{\sigma_{\text{LN}}}_{i}(x)\cdot\sqrt{d}}\mathbin{\raisebox{2.15277pt}{,}}\quad n_{\text{{I\hskip-0.70004ptd}}}:x\rightarrow x,
$$
$$
\displaystyle\mu_{\text{{B\hskip-0.70004ptN}}}^{j}
$$
 
$$
\displaystyle:x\rightarrow\frac{1}{B}\sum_{i=1}^{B}x_{i}^{j},\quad\sigma_{\text{{B\hskip-0.70004ptN}}}^{j}:x\rightarrow\sqrt{\frac{1}{B}\sum_{i=1}^{B}\mathopen{}\mathclose{{}\left({x_{i}^{j}}}\right)^{2}-\mu_{\text{{B\hskip-0.70004ptN}}}^{j}(x)^{2}},\quad
$$
 
$$
\displaystyle{\mu_{\text{{L\hskip-0.70004ptN}}}}_{i}
$$
 
$$
\displaystyle:x\rightarrow\frac{1}{d}\sum_{j=1}^{d}x_{i}^{j},\quad{\sigma_{\text{{L\hskip-0.70004ptN}}}}_{i}:x\rightarrow\frac{\|x_{i}-{\mu_{\textsc{L\hskip-0.70004ptN}}}_{i}(x)\|_{2}}{\sqrt{d}}
$$

When using no normalization at all, the projection $\ell_{2}$ norm rapidly increases during the first 100 epochs and stabilizes at around $3\cdot 10^{6}$ as shown in Figure 7. Despite this behaviour, using no normalization still performs reasonably well $(67.4\%)$. The $\ell_{2}$ normalization performs the best.

## Appendix G Training with smaller batch sizes

The results described in Section 4 were obtained using a batch size of $4096$ split over $512$ TPU cores. Due to its increased robustness, BYOL can also be trained using smaller batch sizes without significantly decreasing performance. Using the same linear evaluation setup, BYOL achieves $73.7\%$ top-1 accuracy when trained over $1000$ epochs with a batch size of $512$ split over $64$ TPU cores (approximately $4$ days of training). For this setup, we reuse the same setting as in Section 3, but use a base learning rate of $0.4$ (appropriately scaled by the batch size) and $\tau_{\text{base}}=0.9995$ with the same weight decay coefficient of $1.5\cdot 10^{-6}$.

## Appendix H Details on Equation in Section

In this section we clarify why BYOL’s update is related to Equation 5 from Section 3.2,

$$
\displaystyle\nabla_{\theta}\mathbb{E}\mathopen{}\mathclose{{}\left[\mathopen{}\mathclose{{}\left\|q^{\star}(z_{\theta})-z^{\prime}_{\xi}}\right\|_{2}^{2}}\right]=\nabla_{\theta}\mathbb{E}\mathopen{}\mathclose{{}\left[\mathopen{}\mathclose{{}\left\|\mathbb{E}\mathopen{}\mathclose{{}\left[z^{\prime}_{\xi}|z_{\theta}}\right]-z^{\prime}_{\xi}}\right\|_{2}^{2}}\right]=\nabla_{\theta}\mathbb{E}\mathopen{}\mathclose{{}\left[\sum_{i}\operatorname*{Var}(z^{\prime}_{\xi,i}|z_{\theta})}\right].
$$

Recall that $q^{\star}$ is defined as

$$
\displaystyle q^{\star}\mathrel{\ensurestackMath{\stackon[1pt]{=}{\scriptscriptstyle\Delta}}}\operatorname*{arg\,min}_{q}\mathbb{E}\mathopen{}\mathclose{{}\left[\mathopen{}\mathclose{{}\left\|q(z_{\theta})-z^{\prime}_{\xi}}\right\|_{2}^{2}}\right],\quad\text{where}\quad q^{\star}(z_{\theta})=\mathbb{E}\mathopen{}\mathclose{{}\left[z^{\prime}_{\xi}|z_{\theta}}\right],
$$

and implicitly depends on ${\theta}$ and $\xi$; therefore, it should be denoted as $q^{\star}({\theta},\xi)$ instead of just $q^{\star}$. For simplicity we write $q^{\star}({\theta},\xi)(z_{\theta})$ as $q^{\star}({\theta},\xi,z_{\theta})$ the output of the optimal predictor for any parameters ${\theta}$ and $\xi$ and input $z_{\theta}$.

BYOL updates its online parameters following the gradient of Equation 5, but considering only the gradients of $q$ with respect to its third argument $z$ when applying the chain rule. If we rewrite

$$
\displaystyle\mathbb{E}\mathopen{}\mathclose{{}\left[\mathopen{}\mathclose{{}\left\|q^{\star}({\theta},\xi,z_{\theta})-z^{\prime}_{\xi}}\right\|_{2}^{2}}\right]=\mathbb{E}\mathopen{}\mathclose{{}\left[L(q^{\star}({\theta},\xi,z_{\theta}),z^{\prime}_{\xi})}\right],
$$

the gradient of this quantity w.r.t. ${\theta}$ is

$$
\displaystyle\frac{\partial}{\partial{\theta}}\mathbb{E}\mathopen{}\mathclose{{}\left[L(q^{\star}({\theta},\xi,z_{\theta}),z^{\prime}_{\xi})}\right]=\mathbb{E}\mathopen{}\mathclose{{}\left[\frac{\partial L}{\partial q}\cdot\frac{\partial q^{\star}}{\partial{\theta}}+\frac{\partial L}{\partial q}\cdot\frac{\partial q^{\star}}{\partial z}\cdot\frac{\partial z_{\theta}}{\partial{\theta}}}\right],
$$

where $\frac{\partial q^{\star}}{\partial{\theta}}$ and $\frac{\partial q^{\star}}{\partial z}$ are the gradients of $q^{\star}$ with respect to its first and last argument. Using the envelope theorem, and thanks to the optimality condition of the predictor, the term $\mathbb{E}\mathopen{}\mathclose{{}\left[\frac{\partial L}{\partial q}\cdot\frac{\partial q^{\star}}{\partial{\theta}}}\right]=0$. Therefore, the remaining term $\mathbb{E}\mathopen{}\mathclose{{}\left[\frac{\partial L}{\partial q}\cdot\frac{\partial q^{\star}}{\partial z}\cdot\frac{\partial z_{\theta}}{\partial{\theta}}}\right]$ where gradients are only back-propagated through the predictor’s input is exactly the direction followed by BYOL.

## Appendix I Importance of a near-optimal predictor

In this part we build upon the intuitions of Section 3.2 on the importance of keeping the predictor near-optimal. Specifically, we show that it is possible to remove the exponential moving average in BYOL’s target network (*i.e.*, simply copy weights of the online network into the target) without causing the representation to collapse, provided the predictor remains sufficiently good.

### I.1 Predictor learning rate

In this setup, we remove the exponential moving average (*i.e.*, set $\tau=0$ over the full training in Equation 1), and multiply the learning rate of the predictor by a constant $\lambda$ compared to the learning rate used for the rest of the network; all other hyperparameters are unchanged. As shown in Table 21, using sufficiently large values of $\lambda$ provides a reasonably good level of performance and the performance sharply decreases with $\lambda$ to $0.01\%$ top-1 accuracy (no better than random) for $\lambda=0$.

To show that this effect is directly related to a change of behavior in the predictor, and not only to a change of learning rate in any subpart of the network, we perform a similar experiment by using a multiplier $\lambda$ on the *predictor* ’s learning rate, and a different multiplier $\mu$ for the *projector*. In Table 22, we show that the representation typically collapses or performs poorly when the predictor learning rate is lower or equal to that of the projector. As mentioned in Section 3.2, we further hypothesize that one of the contributions of the target network is to maintain a near optimal predictor at all times.

### I.2 Optimal linear predictor in closed form

Similarly, we can get rid of the slowly moving target network if we use a closed form optimal predictor on the batch, instead of a learned, non-optimal one. In this case we restrict ourselves to a linear predictor,

$$
\displaystyle q^{\star}=\operatorname*{arg\,min}_{Q}\mathopen{}\mathclose{{}\left\|Z_{\theta}Q-Z^{\prime}_{\xi}}\right\|_{2}^{2}=\mathopen{}\mathclose{{}\left(Z_{\theta}^{\mathsf{\scriptscriptstyle T}}Z_{\theta}}\right)^{-1}Z_{\theta}^{\mathsf{\scriptscriptstyle T}}Z^{\prime}_{\xi}
$$

with $\mathopen{}\mathclose{{}\left\|\cdot}\right\|_{2}$ being the Frobenius norm, $Z_{\theta}$ and $Z^{\prime}_{\xi}$ of shape $(B,F)$ respectively the online and target projections, where $B$ is the batch size and $F$ the number of features; and $q^{\star}$ of shape $(F,F)$, the optimal linear predictor for the batch.

At $300$ epochs, when using the closed form optimal predictor, and directly hard copying the weights of the online network into the target, we obtain a top- $1$ accuracy of $fill$.

| $\lambda$ | Top- $1$ |
| --- | --- |
| $00$ | $0.01$ |
| $1$ | $5.5$ |
| $2$ | $62.8{\scriptstyle\pm 1.5}$ |
| $10$ | $66.6$ |
| $20$ | $66.3{\scriptstyle\pm 0.3}$ |
| Baseline | $72.5$ |

Table 21: Top- $1$ accuracy at 300 epochs when removing the slowly moving target network, directly hard copying the weights of the online network into the target network, and applying a multiplier to the predictor learning rate.

<table><thead><tr><th></th><th></th><th colspan="5"><math><semantics><msub><mi>λ</mi> <mrow><mi>p</mi> <mo></mo><mi>r</mi> <mo></mo><mi>e</mi> <mo></mo><mi>d</mi></mrow></msub> <apply><csymbol>subscript</csymbol> <ci>𝜆</ci> <apply><ci>𝑝</ci> <ci>𝑟</ci> <ci>𝑒</ci> <ci>𝑑</ci></apply></apply> <annotation>\lambda_{pred}</annotation></semantics></math></th></tr><tr><th></th><th></th><th><math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></th><th><math><semantics><mn>1.5</mn> <cn>1.5</cn> <annotation>1.5</annotation></semantics></math></th><th><math><semantics><mn>2</mn> <cn>2</cn> <annotation>2</annotation></semantics></math></th><th><math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></th><th><math><semantics><mn>10</mn> <cn>10</cn> <annotation>10</annotation></semantics></math></th></tr></thead><tbody><tr><th rowspan="5"><math><semantics><msub><mi>μ</mi> <mrow><mi>p</mi> <mo></mo><mi>r</mi> <mo></mo><mi>o</mi> <mo></mo><mi>j</mi></mrow></msub> <apply><csymbol>subscript</csymbol> <ci>𝜇</ci> <apply><ci>𝑝</ci> <ci>𝑟</ci> <ci>𝑜</ci> <ci>𝑗</ci></apply></apply> <annotation>\mu_{proj}</annotation></semantics></math></th><th><math><semantics><mn>1</mn> <cn>1</cn> <annotation>1</annotation></semantics></math></th><td><math><semantics><mrow><mn>3.2</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>3.2</cn> <cn>2.9</cn></apply> <annotation>3.2{\scriptstyle\pm 2.9}</annotation></semantics></math></td><td><math><semantics><mrow><mn>25.7</mn> <mo>±</mo> <mn>6.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>25.7</cn> <cn>6.6</cn></apply> <annotation>25.7{\scriptstyle\pm 6.6}</annotation></semantics></math></td><td><math><semantics><mrow><mn>60.8</mn> <mo>±</mo> <mn>2.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>60.8</cn> <cn>2.9</cn></apply> <annotation>60.8{\scriptstyle\pm 2.9}</annotation></semantics></math></td><td><math><semantics><mrow><mn>66.7</mn> <mo>±</mo> <mn>0.4</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>66.7</cn> <cn>0.4</cn></apply> <annotation>66.7{\scriptstyle\pm 0.4}</annotation></semantics></math></td><td><math><semantics><mn>66.9</mn> <cn>66.9</cn> <annotation>66.9</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>1.5</mn> <cn>1.5</cn> <annotation>1.5</annotation></semantics></math></th><td><math><semantics><mrow><mn>1.4</mn> <mo>±</mo> <mn>2.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>1.4</cn> <cn>2.0</cn></apply> <annotation>1.4{\scriptstyle\pm 2.0}</annotation></semantics></math></td><td><math><semantics><mrow><mn>9.2</mn> <mo>±</mo> <mn>7.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>9.2</cn> <cn>7.0</cn></apply> <annotation>9.2{\scriptstyle\pm 7.0}</annotation></semantics></math></td><td><math><semantics><mrow><mn>55.2</mn> <mo>±</mo> <mn>5.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>55.2</cn> <cn>5.8</cn></apply> <annotation>55.2{\scriptstyle\pm 5.8}</annotation></semantics></math></td><td><math><semantics><mrow><mn>61.5</mn> <mo>±</mo> <mn>0.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>61.5</cn> <cn>0.6</cn></apply> <annotation>61.5{\scriptstyle\pm 0.6}</annotation></semantics></math></td><td><math><semantics><mrow><mn>66.0</mn> <mo>±</mo> <mn>0.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>66.0</cn> <cn>0.3</cn></apply> <annotation>66.0{\scriptstyle\pm 0.3}</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>2</mn> <cn>2</cn> <annotation>2</annotation></semantics></math></th><td><math><semantics><mrow><mn>2.0</mn> <mo>±</mo> <mn>2.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>2.0</cn> <cn>2.8</cn></apply> <annotation>2.0{\scriptstyle\pm 2.8}</annotation></semantics></math></td><td><math><semantics><mrow><mn>5.3</mn> <mo>±</mo> <mn>1.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>5.3</cn> <cn>1.9</cn></apply> <annotation>5.3{\scriptstyle\pm 1.9}</annotation></semantics></math></td><td><math><semantics><mrow><mn>15.8</mn> <mo>±</mo> <mn>13.4</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>15.8</cn> <cn>13.4</cn></apply> <annotation>15.8{\scriptstyle\pm 13.4}</annotation></semantics></math></td><td><math><semantics><mrow><mn>60.9</mn> <mo>±</mo> <mn>0.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>60.9</cn> <cn>0.8</cn></apply> <annotation>60.9{\scriptstyle\pm 0.8}</annotation></semantics></math></td><td><math><semantics><mn>66.3</mn> <cn>66.3</cn> <annotation>66.3</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>5</mn> <cn>5</cn> <annotation>5</annotation></semantics></math></th><td><math><semantics><mrow><mn>1.5</mn> <mo>±</mo> <mn>0.9</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>1.5</cn> <cn>0.9</cn></apply> <annotation>1.5{\scriptstyle\pm 0.9}</annotation></semantics></math></td><td><math><semantics><mrow><mn>2.5</mn> <mo>±</mo> <mn>1.5</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>2.5</cn> <cn>1.5</cn></apply> <annotation>2.5{\scriptstyle\pm 1.5}</annotation></semantics></math></td><td><math><semantics><mrow><mn>2.5</mn> <mo>±</mo> <mn>1.4</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>2.5</cn> <cn>1.4</cn></apply> <annotation>2.5{\scriptstyle\pm 1.4}</annotation></semantics></math></td><td><math><semantics><mrow><mn>20.5</mn> <mo>±</mo> <mn>2.0</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>20.5</cn> <cn>2.0</cn></apply> <annotation>20.5{\scriptstyle\pm 2.0}</annotation></semantics></math></td><td><math><semantics><mrow><mn>60.5</mn> <mo>±</mo> <mn>0.6</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>60.5</cn> <cn>0.6</cn></apply> <annotation>60.5{\scriptstyle\pm 0.6}</annotation></semantics></math></td></tr><tr><th><math><semantics><mn>10</mn> <cn>10</cn> <annotation>10</annotation></semantics></math></th><td><math><semantics><mn>0.1</mn> <cn>0.1</cn> <annotation>0.1</annotation></semantics></math></td><td><math><semantics><mrow><mn>2.1</mn> <mo>±</mo> <mn>0.3</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>2.1</cn> <cn>0.3</cn></apply> <annotation>2.1{\scriptstyle\pm 0.3}</annotation></semantics></math></td><td><math><semantics><mrow><mn>1.9</mn> <mo>±</mo> <mn>0.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>1.9</cn> <cn>0.8</cn></apply> <annotation>1.9{\scriptstyle\pm 0.8}</annotation></semantics></math></td><td><math><semantics><mrow><mn>2.8</mn> <mo>±</mo> <mn>0.4</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>2.8</cn> <cn>0.4</cn></apply> <annotation>2.8{\scriptstyle\pm 0.4}</annotation></semantics></math></td><td><math><semantics><mrow><mn>8.3</mn> <mo>±</mo> <mn>6.8</mn></mrow> <apply><csymbol>plus-or-minus</csymbol> <cn>8.3</cn> <cn>6.8</cn></apply> <annotation>8.3{\scriptstyle\pm 6.8}</annotation></semantics></math></td></tr></tbody></table>

Table 22: Top- $1$ accuracy at 300 epochs when removing the slowly moving target network, directly hard copying the weights of the online network in the target network, and applying a multiplier $\mu$ to the projector and $\lambda$ to the predictor learning rate. The predictor learning rate needs to be higher than the projector learning rate in order to successfully remove the target network. This further suggests that the learning dynamic of predictor is central to BYOL’s stability.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2006.07733/assets/x11.png)

Figure 8: BYOL sketch summarizing the method by emphasizing the neural architecture.

See pages 1-3 of [lalala.pdf](https://ar5iv.labs.arxiv.org/html/lalala.pdf)

[^1]: Kunihiko Fukushima. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics, 36(4):193–202, 1980.

[^2]: Laurenz Wiskott and Terrence J Sejnowski. Slow feature analysis: Unsupervised learning of invariances. Neural Computation, 14(4), 2002.

[^3]: Geoffrey E Hinton, Simon Osindero, and Yee-Whye Teh. A fast learning algorithm for deep belief nets. Neural Computation, 18(7):1527–1554, 2006.

[^4]: Maxime Oquab, Leon Bottou, Ivan Laptev, and Josef Sivic. Learning and transferring mid-level image representations using convolutional neural networks. In Computer Vision and Pattern Recognition, 2014.

[^5]: Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

[^6]: Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Computer Vision and Pattern Recognition, 2014.

[^7]: Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Computer Vision and Pattern Recognition, 2015.

[^8]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709, 2020.

[^9]: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. Momentum contrast for unsupervised visual representation learning. arXiv preprint arXiv:1911.05722, 2019.

[^10]: Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[^11]: Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv preprint arXiv:1906.05849v4, 2019.

[^12]: Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning. arXiv preprint arXiv:2005.10243, 2020.

[^13]: Nikunj Saunshi, Orestis Plevrakis, Sanjeev Arora, Mikhail Khodak, and Hrishikesh Khandeparkar. A theoretical analysis of contrastive unsupervised representation learning. In International Conference on Machine Learning, 2019.

[^14]: R. Manmatha, Chao-Yuan Wu, Alexander J. Smola, and Philipp Krähenbühl. Sampling matters in deep embedding learning. In International Conference on Computer Vision, 2017.

[^15]: Ben Harwood, Vijay B. G. Kumar, Gustavo Carneiro, Ian Reid, and Tom Drummond. Smart mining for deep metric learning. In International Conference on Computer Vision, 2017.

[^16]: Dong-Hyun Lee. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In International Conference on Machine Learning, 2013.

[^17]: Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In European Conference on Computer Vision, 2018.

[^18]: Philip Bachman, Ouais Alsharif, and Doina Precup. Learning with pseudo-ensembles. In Advances in neural information processing systems, pages 3365–3373, 2014.

[^19]: Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. arXiv preprint arXiv:1610.02242, 2016.

[^20]: Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in neural information processing systems, pages 1195–1204, 2017.

[^21]: Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision, 115(3):211–252, 2015.

[^22]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Computer Vision and Pattern Recognition, 2016.

[^23]: Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In Computer Vision and Pattern Recognition, 2015.

[^24]: Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In International Conference on Machine Learning, 2008.

[^25]: Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[^26]: Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic back-propagation and variational inference in deep latent gaussian models. arXiv preprint arXiv:1401.4082, 2014.

[^27]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Neural Information Processing Systems, 2014.

[^28]: Jeff Donahue, Philipp Krähenbühl, and Trevor Darrell. Adversarial feature learning. arXiv preprint arXiv:1605.09782, 2016.

[^29]: Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Alex Lamb, Martín Arjovsky, Olivier Mastropietro, and Aaron C. Courville. Adversarially learned inference. arXiv preprint arXiv:1606.00704, 2017.

[^30]: Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. In Neural Information Processing Systems, 2019.

[^31]: Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.

[^32]: Olivier J. Hénaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, S. M. Ali Eslami, and Aäron van den Oord. Data-efficient image recognition with contrastive predictive coding. In International Conference on Machine Learning, 2019.

[^33]: R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670, 2019.

[^34]: Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In Neural Information Processing Systems, 2019.

[^35]: Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. arXiv preprint arXiv:1912.01991, 2019.

[^36]: Junnan Li, Pan Zhou, Caiming Xiong, Richard Socher, and Steven CH Hoi. Prototypical contrastive learning of unsupervised representations. arXiv preprint arXiv:2005.04966, 2020.

[^37]: Rishabh Jain, Haoqi Fan, Ross B. Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.

[^38]: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. arXiv preprint arXiv:2006.10029, 2020.

[^39]: Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In Computer Vision and Pattern Recognition, 2018.

[^40]: Carl Doersch and Andrew Zisserman. Multi-task self-supervised visual learning. In International Conference on Computer Vision, 2017.

[^41]: Richard Zhang, Phillip Isola, and Alexei A. Efros. Colorful image colorization. In European Conference on Computer Vision, 2016.

[^42]: Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Learning representations for automatic colorization. In European Conference on Computer Vision, 2016.

[^43]: Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting. In Computer Vision and Pattern Recognition, 2016.

[^44]: Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European Conference on Computer Vision, 2016.

[^45]: Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photo-realistic single image super-resolution using a generative adversarial network. In Computer Vision and Pattern Recognition, 2017.

[^46]: Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. In Neural Information Processing Systems, 2014.

[^47]: Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018.

[^48]: Alexander Kolesnikov, Xiaohua Zhai, and Lucas Beyer. Revisiting self-supervised visual representation learning. In Computer Vision and Pattern Recognition, 2019.

[^49]: Daniel Guo, Bernardo Avila Pires, Bilal Piot, Jean-Bastien Grill, Florent Altché, Rémi Munos, and Mohammad Gheshlaghi Azar. Bootstrap latent-predictive representations for multitask reinforcement learning. In International Conference on Machine Learning, 2020.

[^50]: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin A. Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen. King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518:529–533, 2015.

[^51]: Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, 2016.

[^52]: Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Gheshlaghi Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In AAAI Conference on Artificial Intelligence, 2018.

[^53]: Hado Van Hasselt, Yotam Doron, Florian Strub, Matteo Hessel, Nicolas Sonnerat, and Joseph Modayil. Deep reinforcement learning and the deadly triad. Deep Reinforcement Learning Workshop NeurIPS, 2018.

[^54]: Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

[^55]: Olivier Chapelle, Bernhard Scholkopf, and Alexander Zien. Semi-supervised learning. IEEE Transactions on Neural Networks, 20(3):542–542, 2009.

[^56]: Xiaojin Zhu and Andrew B Goldberg. Introduction to semi-supervised learning. Synthesis lectures on artificial intelligence and machine learning, 3(1):1–130, 2009.

[^57]: Durk P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Advances in neural information processing systems, 2014.

[^58]: Antti Rasmus, Mathias Berglund, Mikko Honkala, Harri Valpola, and Tapani Raiko. Semi-supervised learning with ladder networks. In Advances in neural information processing systems, 2015.

[^59]: David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. In Advances in Neural Information Processing Systems, 2019.

[^60]: Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a regularization method for supervised and semi-supervised learning. IEEE transactions on pattern analysis and machine intelligence, 41(8):1979–1993, 2018.

[^61]: David Berthelot, N. Carlini, E. D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, and Colin Raffel. Remixmatch: Semi-supervised learning with distribution matching and augmentation anchoring. In ICLR, 2020.

[^62]: Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. arXiv preprint arXiv:2001.07685, 2020.

[^63]: Suzanna Becker and Geoffrey E. Hinton. Self-organizing neural network that discovers surfaces in random-dot stereograms. Nature, 355(6356):161–163, 1992.

[^64]: James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, and Skye Wanderman-Milne. JAX: composable transformations of Python+NumPy programs, 2018.

[^65]: Tom Hennigan, Trevor Cai, Tamara Norman, and Igor Babuschkin. Haiku: Sonnet for JAX, 2020.

[^66]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pages 2672–2680, 2014.

[^67]: Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.

[^68]: Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, 2015.

[^69]: Vinod Nair and Geoffrey E. Hinton. Rectified linear units improve restricted boltzmann machines. In International Conference on Machine Learning, 2010.

[^70]: Yang You, Igor Gitman, and Boris Ginsburg. Scaling SGD batch size to 32k for imagenet training. arXiv preprint arXiv:1708.03888, 2017.

[^71]: Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017.

[^72]: Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

[^73]: Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. Transactions on Pattern Analysis and Machine Intelligence, 2017.

[^74]: Simon Kornblith, Jonathon Shlens, and Quoc V Le. Do better ImageNet models transfer better? In Computer Cision and Pattern Recognition, 2019.

[^75]: Chengyue Gong, Tongzheng Ren, Mao Ye, and Qiang Liu. Maxup: A simple way to improve generalization of neural network training. arXiv preprint arXiv:2002.09024, 2020.

[^76]: Xiaohua Zhai, Joan Puigcerver, Alexander I Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, André Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, Lucas Beyer, Olivier Bachem, Michael Tschannen, Marcin Michalski, Olivier Bousquet, Sylvain Gelly, and Neil Houlsby. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv: Computer Vision and Pattern Recognition, 2019.

[^77]: Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. S4L: Self-supervised semi-supervised learning. In International Conference on Computer Vision, 2019.

[^78]: Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[^79]: Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In Computer Vision and Pattern Recognition, 2010.

[^80]: Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The Pascal visual object classes (VOC) challenge. International Journal of Computer Vision, 88(2):303–338, 2010.

[^81]: Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Computer Vision and Pattern Recognition, 2014.

[^82]: Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In Neural Information Processing Systems, 2015.

[^83]: I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and N. Navab. Deeper depth prediction with fully convolutional residual networks. In International Conference on 3D Vision, 2016.

[^84]: Ben Poole, Sherjil Ozair, Aaron van den Oord, Alexander A Alemi, and George Tucker. On variational bounds of mutual information. arXiv preprint arXiv:1905.06922, 2019.

[^85]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, 2015.

[^86]: Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning, 2013.

[^87]: Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. arXiv preprint arXiv:1909.13719, 2019.

[^88]: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Computer Vision and Pattern Recognition, pages 2818–2826, 2016.

[^89]: Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In European Conference on Computer Vision, 2014.

[^90]: Thomas Berg, Jiongxin Liu, Seung Woo Lee, Michelle L. Alexander, David W. Jacobs, and Peter N. Belhumeur. Birdsnap: Large-scale fine-grained visual categorization of birds. In Computer Vision and Pattern Recognition, 2014.

[^91]: Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3D object representations for fine-grained categorization. In Workshop on 3D Representation and Recognition, Sydney, Australia, 2013.

[^92]: Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew B. Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

[^93]: O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar. Cats and dogs. In Computer Vision and Pattern Recognition, 2012.

[^94]: Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. Computer Vision and Pattern Recognition Workshop, 2004.

[^95]: Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics and Image Processing, 2008.

[^96]: Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In International Conference on Machine Learning, 2014.

[^97]: Art B Owen. A robust hybrid of lasso and ridge regression. Contemporary Mathematics, 443(7):59–72, 2007.

[^98]: Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In International Conference on Computer Vision, 2019.

[^99]: Deepak Pathak, Ross Girshick, Piotr Dollár, Trevor Darrell, and Bharath Hariharan. Learning features by watching objects move. In Conference on Computer Vision and Pattern Recognition, 2017.

[^100]: Yuxin Wu and Kaiming He. Group normalization. In European Conference on Computer Vision, 2018.