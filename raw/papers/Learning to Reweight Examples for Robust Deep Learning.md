---
title: "Learning to Reweight Examples for Robust Deep Learning"
source: "https://ar5iv.labs.arxiv.org/html/1803.09050"
author:
published:
created: 2026-08-22
description: "Deep neural networks have been shown to be very powerful modeling tools for many supervised learningtasks involving complex input patterns. However, they can also easily overfit to training setbiases and label noises…"
tags:
  - "clippings"
---
Mengye Ren Affiliation: Uber Advanced Technologies Group, Toronto ON, CANADA Affiliation: Department of Computer Science, University of Toronto, Toronto ON, CANADA Correspondence to: [mren3@uber.com](mailto:mren3@uber.com)    Wenyuan Zeng Affiliation: Uber Advanced Technologies Group, Toronto ON, CANADA Affiliation: Department of Computer Science, University of Toronto, Toronto ON, CANADA    Bin Yang Affiliation: Uber Advanced Technologies Group, Toronto ON, CANADA Affiliation: Department of Computer Science, University of Toronto, Toronto ON, CANADA    Raquel Urtasun Affiliation: Uber Advanced Technologies Group, Toronto ON, CANADA Affiliation: Department of Computer Science, University of Toronto, Toronto ON, CANADA

###### Abstract

Deep neural networks have been shown to be very powerful modeling tools for many supervised learning tasks involving complex input patterns. However, they can also easily overfit to training set biases and label noises. In addition to various regularizers, example reweighting algorithms are popular solutions to these problems, but they require careful tuning of additional hyperparameters, such as example mining schedules and regularization hyperparameters. In contrast to past reweighting methods, which typically consist of functions of the cost value of each example, in this work we propose a novel meta-learning algorithm that learns to assign weights to training examples based on their gradient directions. To determine the example weights, our method performs a meta gradient descent step on the current mini-batch example weights (which are initialized from zero) to minimize the loss on a clean unbiased validation set. Our proposed method can be easily implemented on any type of deep network, does not require any additional hyperparameter tuning, and achieves impressive performance on class imbalance and corrupted label problems where only a small amount of clean validation data is available.

###### Keywords:

Example reweighting, meta-Learning, deep learning, machine learning

## 1 Introduction

Deep neural networks (DNNs) have been widely used for machine learning applications due to their powerful capacity for modeling complex input patterns. Despite their success, it has been shown that DNNs are prone to training set biases, i.e. the training set is drawn from a joint distribution $p(x,y)$ that is different from the distribution $p(x^{v},y^{v})$ of the evaluation set. This distribution mismatch could have many different forms. Class imbalance in the training set is a very common example. In applications such as object detection in the context of autonomous driving, the vast majority of the training data is composed of standard vehicles but models also need to recognize rarely seen classes such as emergency vehicles or animals with very high accuracy. This will sometime lead to biased training models that do not perform well in practice.

Another popular type of training set bias is label noise. To train a reasonable supervised deep model, we ideally need a large dataset with high-quality labels, which require many passes of expensive human quality assurance (QA). Although coarse labels are cheap and of high availability, the presence of noise will hurt the model performance, e.g. [^44] has shown that a standard CNN can fit any ratio of label flipping noise in the training set and eventually leads to poor generalization performance.

Training set biases and misspecification can sometimes be addressed with dataset resampling [^7], i.e. choosing the correct proportion of labels to train a network on, or more generally by assigning a weight to each example and minimizing a weighted training loss. The example weights are typically calculated based on the training loss, as in many classical algorithms such as AdaBoost [^12], hard negative mining [^28], self-paced learning [^22], and other more recent work [^6] [^18].

However, there exist two contradicting ideas in training loss based approaches. In noisy label problems, we prefer examples with smaller training losses as they are more likely to be clean images; yet in class imbalance problems, algorithms such as hard negative mining [^28] prioritize examples with higher training loss since they are more likely to be the minority class. In cases when the training set is both imbalanced and noisy, these existing methods would have the wrong model assumptions. In fact, without a proper definition of an unbiased test set, solving the training set bias problem is inherently ill-defined. As the model cannot distinguish the right from the wrong, stronger regularization can usually work surprisingly well in certain synthetic noise settings. Here we argue that in order to learn general forms of training set biases, it is necessary to have a small unbiased validation to guide training. It is actually not uncommon to construct a dataset with two parts - one relatively small but very accurately labeled, and another massive but coarsely labeled. Coarse labels can come from inexpensive crowdsourcing services or weakly supervised data [^9] [^35] [^8].

Different from existing training loss based approaches, we follow a meta-learning paradigm and model the most basic assumption instead: the best example weighting should minimize the loss of a set of unbiased clean validation examples that are consistent with the evaluation procedure. Traditionally, validation is performed at the end of training, which can be prohibitively expensive if we treat the example weights as some hyperparameters to optimize; to circumvent this, we perform validation at every training iteration to dynamically determine the example weights of the current batch. Towards this goal, we propose an online reweighting method that leverages an additional small validation set and adaptively assigns importance weights to examples in every iteration. We experiment with both class imbalance and corrupted label problems and find that our approach significantly increases the robustness to training set biases.

## 2 Related Work

The idea of weighting each training example has been well studied in the literature. Importance sampling [^19], a classical method in statistics, assigns weights to samples in order to match one distribution to another. Boosting algorithms such as AdaBoost [^12], select harder examples to train subsequent classifiers. Similarly, hard example mining [^28], downsamples the majority class and exploits the most difficult examples. Focal loss [^25] adds a soft weighting scheme that emphasizes harder examples.

Hard examples are not always preferred in the presence of outliers and noise processes. Robust loss estimators typically downweigh examples with high loss. In self-paced learning [^22], example weights are obtained through optimizing the weighted training loss encouraging learning easier examples first. In each step, the learning algorithm jointly solves a mixed integer program that iterates optimizing over model parameters and binary example weights. Various regularization terms on the example weights have since been proposed to prevent overfitting and trivial solutions of assigning weights to be all zeros [^22] [^27] [^17]. [^40] proposed a Bayesian method that infers the example weights as latent variables. More recently, [^18] proposed to use a meta-learning LSTM to output the weights of the examples based on the training loss. Reweighting examples is also related to curriculum learning [^5], where the model reweights among many available tasks. Similar to self-paced learning, typically it is beneficial to start with easier examples.

One crucial advantage of reweighting examples is robustness against training set bias. There has also been a multitude of prior studies on class imbalance problems, including using dataset resampling [^7] [^10], cost-sensitive weighting [^38] [^20], and structured margin based objectives [^16]. Meanwhile, the noisy label problem has been thoroughly studied by the learning theory community [^30] [^3] and practical methods have also been proposed [^33] [^36] [^42] [^4] [^13] [^24] [^18] [^39] [^15]. In addition to corrupted data, [^21] [^29] demonstrate the possibility of a dataset adversarial attack (i.e. dataset poisoning).

Our method improves the training objective through a weighted loss rather than an average loss and is an instantiation of meta-learning [^37] [^23] [^2], i.e. learning to learn better. Using validation loss as the meta-objective has been explored in recent meta-learning literature for few-shot learning [^31] [^34] [^26], where only a handful of examples are available for each class. Our algorithm also resembles MAML [^11] by taking one gradient descent step on the meta-objective for each iteration. However, different from these meta-learning approaches, our reweighting method does not have any additional hyper-parameters and circumvents an expensive offline training stage. Hence, our method can work in an online fashion during regular training.

## 3 Learning to Reweight Examples

In this section, we derive our model from a meta-learning objective towards an online approximation that can fit into any regular supervised training. We give a practical implementation suitable for any deep network type and provide theoretical guarantees under mild conditions that our algorithm has a convergence rate of $O(1/\epsilon^{2})$. Note that this is the same as that of stochastic gradient descent (SGD).

### 3.1 From a meta-learning objective to an online approximation

Let $(x,y)$ be an input-target pair, and $\{(x_{i},y_{i}),1\leq i\leq N\}$ be the training set. We assume that there is a small unbiased and clean validation set $\{(x^{v}_{i},y^{v}_{i}),1\leq i\leq M\}$, and $M\ll N$. Hereafter, we will use superscript $v$ to denote validation set and subscript $i$ to denote the $i^{th}$ data. We also assume that the training set contains the validation set; otherwise, we can always add this small validation set into the training set and leverage more information during training.

Let $\Phi(x,\theta)$ be our neural network model, and $\theta$ be the model parameters. We consider a loss function $C(\hat{y},y)$ to minimize during training, where $\hat{y}=\Phi(x,\theta)$.

In standard training, we aim to minimize the expected loss for the training set: $\frac{1}{N}\sum_{i=1}^{N}C(\hat{y}_{i},y_{i})=\frac{1}{N}\sum_{i=1}^{N}f_{i}(\theta)$, where each input example is weighted equally, and $f_{i}(\theta)$ stands for the loss function associating with data $x_{i}$. Here we aim to learn a reweighting of the inputs, where we minimize a weighted loss:

$$
\theta^{*}(w)=\arg\,\min_{\theta}\sum_{i=1}^{N}w_{i}f_{i}(\theta),
$$

with $w_{i}$ unknown upon beginning. Note that $\{w_{i}\}_{i=1}^{N}$ can be understood as training hyperparameters, and the optimal selection of $w$ is based on its validation performance:

$$
w^{*}=\arg\,\min_{w,w\geq 0}\frac{1}{M}\sum_{i=1}^{M}f_{i}^{v}(\theta^{*}(w)).
$$

It is necessary that $w_{i}\geq 0$ for all $i$, since minimizing the negative training loss can usually result in unstable behavior.

#### Online approximation

Calculating the optimal $w_{i}$ requires two nested loops of optimization, and every single loop can be very expensive. The motivation of our approach is to adapt online $w$ through a single optimization loop. For each training iteration, we inspect the descent direction of some training examples locally on the training loss surface and reweight them according to their similarity to the descent direction of the validation loss surface.

For most training of deep neural networks, SGD or its variants are used to optimize such loss functions. At every step $t$ of training, a mini-batch of training examples $\{(x_{i},y_{i}),1\leq i\leq n\}$ is sampled, where $n$ is the mini-batch size, $n\ll N$. Then the parameters are adjusted according to the descent direction of the expected loss on the mini-batch. Let’s consider vanilla SGD:

$$
\displaystyle\theta_{t+1}
$$
 
$$
\displaystyle=\theta_{t}-\alpha\nabla\left(\frac{1}{n}\sum_{i=1}^{n}f_{i}(\theta_{t})\right),
$$

where $\alpha$ is the step size.

We want to understand what would be the impact of training example $i$ towards the performance of the validation set at training step $t$. Following a similar analysis to [^21], we consider perturbing the weighting by $\epsilon_{i}$ for each training example in the mini- batch,

$$
\displaystyle f_{i,\epsilon}(\theta)
$$
 
$$
\displaystyle=\epsilon_{i}f_{i}(\theta),
$$
$$
\displaystyle\hat{\theta}_{t+1}(\epsilon)
$$
 
$$
\displaystyle=\theta_{t}-\alpha\nabla\sum_{i=1}^{n}f_{i,\epsilon}(\theta)\Bigr|_{\theta=\theta_{t}}.
$$

We can then look for the optimal $\epsilon^{*}$ that minimizes the validation loss $f^{v}$ locally at step $t$:

$$
\displaystyle\epsilon^{*}_{t}=\arg\,\min_{\epsilon}\frac{1}{M}\sum_{i=1}^{M}f^{v}_{i}(\theta_{t+1}(\epsilon)).
$$

Unfortunately, this can still be quite time-consuming. To get a cheap estimate of $w_{i}$ at step $t$, we take a single gradient descent step on a mini-batch of validation samples wrt. $\epsilon_{t}$, and then rectify the output to get a non-negative weighting:

$$
\displaystyle u_{i,t}
$$
 
$$
\displaystyle=-\eta\frac{\partial}{\partial\epsilon_{i,t}}\frac{1}{m}\sum_{j=1}^{m}f_{j}^{v}(\theta_{t+1}(\epsilon))\Bigr|_{\epsilon_{i,t}=0},
$$
$$
\displaystyle\tilde{w}_{i,t}
$$
 
$$
\displaystyle=\max(u_{i,t},0).
$$

where $\eta$ is the descent step size on $\epsilon$.

To match the original training step size, in practice, we can consider normalizing the weights of all examples in a training batch so that they sum up to one. In other words, we choose to have a hard constraint within the set $\{w:\lVert w\rVert_{1}=1\}\cup\{0\}$.

$$
\displaystyle w_{i,t}=\frac{\tilde{w}_{i,t}}{(\sum_{j}\tilde{w}_{j,t})+\delta(\sum_{j}\tilde{w}_{j,t})},
$$

where $\delta(\cdot)$ is to prevent the degenerate case when all $w_{i}$ ’s in a mini-batch are zeros, i.e. $\delta(a)=1$ if $a=0$, and equals to $0$ otherwise. Without the batch-normalization step, it is possible that the algorithm modifies its effective learning rate of the training progress, and our one-step look ahead may be too conservative in terms of the choice of learning rate [^41]. Moreover, with batch normalization, we effectively cancel the meta learning rate parameter $\eta$.

### 3.2 Example: learning to reweight examples in a multi-layer perceptron network

In this section, we study how to compute $w_{i,t}$ in a multi-layer perceptron (MLP) network. One of the core steps is to compute the gradients of the validation loss wrt. the local perturbation $\epsilon$, We can consider a multi-layered network where we have parameters for each layer $\theta=\{\theta_{l}\}_{l=1}^{L}$, and at every layer, we first compute $z_{l}$ the pre-activation, a weighted sum of inputs to the layer, and afterwards we apply a non-linear activation function $\sigma$ to obtain $\tilde{z}_{l}$ the post-activation:

$$
\displaystyle z_{l}
$$
 
$$
\displaystyle=\theta_{l}^{\top}\tilde{z}_{l-1},
$$
$$
\displaystyle\tilde{z}_{l}
$$
 
$$
\displaystyle=\sigma(z_{l}).
$$

During backpropagation, let $g_{l}$ be the gradients of loss wrt. $z_{l}$, and the gradients wrt. $\theta_{l}$ is given by $\tilde{z}_{l-1}g_{l}^{\top}$. We can further express the gradients towards $\epsilon$ as a sum of local dot products.

$$
\displaystyle\begin{split}&\frac{\partial}{\partial\epsilon_{i,t}}\mathbb{E}\left[f^{v}(\theta_{t+1}(\epsilon))\Bigr|_{\epsilon_{i,t}=0}\right]\\
\propto&-\frac{1}{m}\sum_{j=1}^{m}\frac{\partial f_{j}^{v}(\theta)}{\partial\theta}\Bigr|_{\theta=\theta_{t}}^{\top}\frac{\partial f_{i}(\theta)}{\partial\theta}\Bigr|_{\theta=\theta_{t}}\\
=&-\frac{1}{m}\sum_{j=1}^{m}\sum_{l=1}^{L}(\tilde{z}^{v}_{j,l-1}{}^{\top}\tilde{z}_{i,l-1})(g^{v}_{j,l}{}^{\top}g_{i,l}).\end{split}
$$

Detailed derivations can be found in Appendix A. Eq. 12 suggests that the meta-gradient on $\epsilon$ is composed of the sum of the products of two terms: $z^{\top}z^{v}$ and $g^{\top}g^{v}$. The first dot product computes the similarity between the training and validation inputs to the layer, while the second computes the similarity between the training and validation gradient directions. In other words, suppose that a pair of training and validation examples are very similar, and they also provide similar gradient directions, then this training example is helpful and should be up-weighted, and conversely, if they provide opposite gradient directions, this training example is harmful and should be downweighed.

### 3.3 Implementation using automatic differentiation

Figure 1: Computation graph of our algorithm in a deep neural network, which can be efficiently implemented using second order automatic differentiation.

In an MLP and a CNN, the unnormalized weights can be calculated based on the sum of the correlations of layerwise activation gradients and input activations. In more general networks, we can leverage automatic differentiation techniques to compute the gradient of the validation loss wrt. the example weights of the current batch. As shown in Figure 1, to get the gradients of the example weights, one needs to first unroll the gradient graph of the training batch, and then use backward-on-backward automatic differentiation to take a second order gradient pass (see Step 5 in Figure 1). We list detailed step-by-step pseudo-code in Algorithm 1. This implementation can be generalized to any deep learning architectures and can be very easily implemented using popular deep learning frameworks such as TensorFlow [^1].

Algorithm 1 Learning to Reweight Examples using Automatic Differentiation

  $\theta_{0}$, $\mathcal{D}_{f}$, $\mathcal{D}_{g}$, $n$, $m$

  $\theta_{T}$

 for $t=0$ … $T-1$ do

   $\{X_{f},y_{f}\}\leftarrow$ SampleMiniBatch($\mathcal{D}_{f}$, $n$)

   $\{X_{g},y_{g}\}\leftarrow$ SampleMiniBatch($\mathcal{D}_{g}$, $m$)

   $\hat{y}_{f}\leftarrow\text{Forward}(X_{f},y_{f},\theta_{t})$

   $\epsilon\leftarrow 0$; $l_{f}\leftarrow\sum_{i=1}^{n}\epsilon_{i}C(y_{f,i},\hat{y}_{f,i})$

   $\nabla\theta_{t}\leftarrow\text{BackwardAD}(l_{f},\theta_{t})$    $\hat{\theta}_{t}\leftarrow\theta_{t}-\alpha\nabla\theta_{t}$    $\hat{y}_{g}\leftarrow\text{Forward}(X_{g},y_{g},\hat{\theta}_{t})$    $l_{g}\leftarrow\frac{1}{m}\sum_{i=1}^{m}C(y_{g,i},\hat{y}_{g,i})$    $\nabla\epsilon\leftarrow\text{BackwardAD}(l_{g},\epsilon)$

   $\tilde{w}\leftarrow\max(-\nabla\epsilon,0)$; $w\leftarrow\frac{\tilde{w}}{\sum_{j}\tilde{w}+\delta(\sum_{j}\tilde{w})}$

   $\hat{l}_{f}\leftarrow\sum_{i=1}^{n}w_{i}C(y_{i},\hat{y}_{f,i})$    $\nabla\theta_{t}\leftarrow\text{BackwardAD}(\hat{l}_{f},\theta_{t})$    $\theta_{t+1}\leftarrow\text{OptimizerStep}(\theta_{t},\nabla\theta_{t})$

 end for

#### Training time

Our automatic reweighting method will introduce a constant factor of overhead. First, it requires two full forward and backward passes of the network on training and validation respectively, and then another backward on backward pass (Step 5 in Figure 1), to get the gradients to the example weights, and finally a backward pass to minimize the reweighted objective. In modern networks, a backward-on-backward pass usually takes about the same time as a forward pass, and therefore compared to regular training, our method needs approximately 3 $\times$ training time; it is also possible to reduce the batch size of the validation pass for speedup. We expect that it is worthwhile to spend the extra time to avoid the irritation of choosing early stopping, finetuning schedules, and other hyperparameters.

### 3.4 Analysis: convergence of the reweighted training

Convergence results of SGD based optimization methods are well-known [^32]. However it is still meaningful to establish a convergence result about our method since it involves optimization of two-level objectives (Eq. 1, 2) rather than one, and we further make some first-order approximation by introducing Eq. 7. Here, we show theoretically that our method converges to the critical point of the validation loss function under some mild conditions, and we also give its convergence rate. More detailed proofs can be found in the Appendix B, C.

###### Definition 1.

A function $f(x):\mathbb{R}^{d}\to\mathbb{R}$ is said to be Lipschitz-smooth with constant $L$ if

$$
\displaystyle\lVert\nabla f(x)-\nabla f(y)\rVert\leq L\lVert x-y\rVert,\forall x,y\in\mathbb{R}^{d}.
$$

###### Definition 2.

$f(x)$ has $\sigma$ -bounded gradients if $\lVert\nabla f(x)\rVert\leq\sigma$ for all $x\in\mathbb{R}^{d}$.

In most real-world cases, the high-quality validation set is really small, and thus we could set the mini-batch size $m$ to be the same as the size of the validation set $M$. Under this condition, the following lemma shows that our algorithm always converges to a critical point of the validation loss. However, our method is not equivalent to training a model only on this small validation set. Because directly training a model on a small validation set will lead to severe overfitting issues. On the contrary, our method can leverage useful information from a larger training set, and still converge to an appropriate distribution favored by this clean and balanced validation dataset. This helps both generalization and robustness to biases in the training set, which will be shown in our experiments.

###### Lemma 1.

Suppose the validation loss function is Lipschitz-smooth with constant $L$, and the train loss function $f_{i}$ of training data $x_{i}$ have $\sigma$ -bounded gradients. Let the learning rate $\alpha_{t}$ satisfies $\alpha_{t}\leq\frac{2n}{L\sigma^{2}}$, where $n$ is the training batch size. Then, following our algorithm, the validation loss always monotonically decreases for any sequence of training batches, namely,

$$
\displaystyle G(\theta_{t+1})\leq G(\theta_{t}),
$$

where $G(\theta)$ is the total validation loss

$$
\displaystyle G(\theta)=\frac{1}{M}\sum_{i=1}^{M}f^{v}_{i}(\theta_{t+1}(\epsilon)).
$$

Furthermore, in expectation, the equality in Eq. 13 holds only when the gradient of validation loss becomes 0 at some time step $t$, namely $\mathop{\mathbb{E}}_{t}\left[G(\theta_{t+1})\right]=G(\theta_{t})$ if and only if $\nabla G(\theta_{t})=0$, where the expectation is taking over possible training batches at time step $t$.

Moreover, we can prove the convergence rate of our method to be $O(1/\epsilon^{2})$.

###### Theorem 2.

Suppose $G$, $f_{i}$ and $\alpha_{t}$ satisfy the aforementioned conditions, then Algorithm 1 achieves $\mathbb{E}\left[\lVert\nabla G(\theta_{t})\rVert^{2}\right]\leq\epsilon$ in $O(1/\epsilon^{2})$ steps. More specifically,

$$
\displaystyle\min\limits_{0<t<T}\mathbb{E}\left[\lVert\nabla G(\theta_{t})\rVert^{2}\right]\leq\frac{C}{\sqrt{T}},
$$

where $C$ is some constant independent of the convergence process.

## 4 Experiments

To test the effectiveness of our reweighting algorithm, we designed both class imbalance and noisy label settings, and a combination of both, on standard MNIST and CIFAR benchmarks for image classification using deep CNNs. <sup>1</sup>

### 4.1 MNIST data imbalance experiments

We use the standard MNIST handwritten digit classification dataset and subsample the dataset to generate a class imbalance binary classification task. We select a total of 5,000 images of size 28 $\times$ 28 on class 4 and 9, where 9 dominates the training data distribution. We train a standard LeNet on this task and we compare our method with a suite of commonly used tricks for class imbalance: 1) Proportion weights each example by the inverse frequency 2) Resample samples a class-balanced mini-batch for each iteration 3) Hard Mining selects the highest loss examples from the majority class and 4) Random is a random example weight baseline that assigns weights based on a rectified Gaussian distribution:

$$
w_{i}^{\text{rnd}}=\frac{\max(z_{i},0)}{\sum_{i}\max(z_{i},0)},\ \ \ \text{where}\ z_{i}\sim\mathcal{N}(0,1).
$$

To make sure that our method does not have the privilege of training on more data, we split the balanced validation set of 10 images directly from the training set. The network is trained with SGD with a learning rate of 1e-3 and mini-batch size of 100 for a total of 8,000 steps.

Figure 2 plots the test error rate across various imbalance ratios averaged from 10 runs with random splits. Note that our method significantly outperforms all the baselines. With class imbalance ratio of 200:1, our method only reports a small increase of error rate around 2%, whereas other methods suffer terribly under this setting. Compared with resampling and hard negative mining baselines, our approach does not throw away samples based on its class or training loss - as long as a sample is helpful towards the validation loss, it will be included as a part of the training loss.

Figure 2: MNIST 4-9 binary classification error using a LeNet on imbalanced classes. Our method uses a small balanced validation split of 10 examples.

### 4.2 CIFAR noisy label experiments

Reweighting algorithm can also be useful on datasets where the labels are noisy. We study two settings of label noise here:

- UniformFlip: All label classes can uniformly flip to any other label classes, which is the most studied in the literature.
- BackgroundFlip: All label classes can flip to a single background class. This noise setting is very realistic. For instance, human annotators may not have recognized all the positive instances, while the rest remain in the background class. This is also a combination of label imbalance and label noise since the background class usually dominates the label distribution.

We compare our method with prior work on the noisy label problem.

- Reed, proposed by [^33], is a bootstrapping technique where the training target is a convex combination of the model prediction and the label.
- S-Model, proposed by [^13], adds a fully connected softmax layer after the regular classification output layer to model the noise transition matrix.
- MentorNet, proposed by [^18], is an RNN-based meta-learning model that takes in a sequence of loss values and outputs the example weights. We compare numbers reported in their paper with a base model that achieves similar test accuracy under 0% noise.

In addition, we propose two simple baselines: 1) Random, which assigns weights according to a rectified Gaussian (see Eq. 16); 2) Weighted, designed for BackgroundFlip, where the model knows the oracle noise ratio for each class and reweights the training loss proportional to the percentage of clean images of that label class.

#### Clean validation set

For UniformFlip, we use 1,000 clean images in the validation set; for BackgroundFlip, we use 10 clean images per label class. Since our method uses information from the clean validation, for a fair comparison, we conduct an additional finetuning on the clean data based on the pre-trained baselines. We also study the effect on the size of the clean validation set in an ablation study.

#### Hyper-validation set

For monitoring training progress and tuning baseline hyperparameters, we split out another 5,000 hyper-validation set from the 50,000 training images. We also corrupt the hyper-validation set with the same noise type.

Table 1: CIFAR UniformFlip under 40% noise ratio using a WideResNet-28-10 model. Test accuracy shown in percentage. Top rows use only noisy data, and bottom uses additional 1000 clean images. “FT” denotes fine-tuning on clean data.

<table><tbody><tr><th>Model</th><td>CIFAR-10</td><td>CIFAR-100</td></tr><tr><th>Baseline</th><td>67.97 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.62</td><td>50.66 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.24</td></tr><tr><th>Reed-Hard</th><td>69.66 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.21</td><td>51.34 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.17</td></tr><tr><th>S-Model</th><td>70.64 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 3.09</td><td>49.10 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.58</td></tr><tr><th>MentorNet</th><td>76.6</td><td>56.9</td></tr><tr><th>Random</th><td>86.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.32.</td><td>58.01 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.37</td></tr><tr><th colspan="3">Using 1,000 clean images</th></tr><tr><th>Clean Only</th><td>46.64 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 3.90</td><td>9.94 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.82</td></tr><tr><th>Baseline +FT</th><td>78.66 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.44</td><td>54.52 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.40</td></tr><tr><th>MentorNet +FT</th><td>78</td><td>59</td></tr><tr><th>Random +FT</th><td>86.55 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.24</td><td>58.54 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.52</td></tr><tr><th>Ours</th><td>86.92 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.19</td><td>61.34 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.06</td></tr></tbody></table>

Table 2: CIFAR BackgroundFlip under 40% noise ratio using a ResNet-32 model. Test accuracy shown in percentage. Top rows use only noisy data, and bottom rows use additional 10 clean images per class. “+ES” denotes early stopping; “FT” denotes fine-tuning.

<table><tbody><tr><td>Model</td><td>CIFAR-10</td><td>CIFAR-100</td></tr><tr><td>Baseline</td><td>59.54 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.16</td><td>37.82 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.69</td></tr><tr><td>Baseline +ES</td><td>64.96 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.19</td><td>39.08 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.65</td></tr><tr><td>Random</td><td>69.51 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.36</td><td>36.56 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.44</td></tr><tr><td>Weighted</td><td>79.17 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.36</td><td>36.56 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.44</td></tr><tr><td>Reed Soft +ES</td><td>63.47 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.05</td><td>38.44 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.90</td></tr><tr><td>Reed Hard +ES</td><td>65.22 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.06</td><td>39.03 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.55</td></tr><tr><td>S-Model</td><td>58.60 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.33</td><td>37.02 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.34</td></tr><tr><td>S-Model +Conf</td><td>68.93 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.09</td><td>46.72 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.87</td></tr><tr><td>S-Model +Conf +ES</td><td>79.24 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.56</td><td>54.50 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.51</td></tr><tr><td colspan="3">Using 10 clean images per class</td></tr><tr><td>Clean Only</td><td>15.90 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 3.32</td><td>8.06 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.76</td></tr><tr><td>Baseline +FT</td><td>82.82 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.93</td><td>54.23 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.75</td></tr><tr><td>Baseline +ES +FT</td><td>85.19 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.46</td><td>55.22 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.40</td></tr><tr><td>Weighted +FT</td><td>85.98 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.47</td><td>53.99 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.62</td></tr><tr><td>S-Model +Conf +FT</td><td>81.90 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.85</td><td>53.11 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.33</td></tr><tr><td>S-Model +Conf +ES +FT</td><td>85.86 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.63</td><td>55.75 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.26</td></tr><tr><td>Ours</td><td>86.73 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.48</td><td>59.30 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.60</td></tr></tbody></table>

#### Experimental details

For Reed model, we use the best $\beta$ reported in [^33] ($\beta=0.8$ for hard bootstrapping and $\beta=0.95$ for soft bootstrapping). For the S-Model, we explore two versions to initialize the transition weights: 1) a smoothed identity matrix; 2) in background flip experiments we consider initializing the transition matrix with the confusion matrix of a pre-trained baseline model (S-Model +Conf). We find baselines can easily overfit the training noise, and therefore we also study early stopped versions of the baselines to provide a stronger comparison. In contrast, we find early stopping not necessary for our method.

To make our results comparable with the ones reported in MentorNet and to save computation time, we exchange their Wide ResNet-101-10 with a Wide ResNet-28-10 (WRN-28-10) [^43] with dropout 0.3 as our base model in the UniformFlip experiments. We find that test accuracy differences between the two base models are within 0.5% on CIFAR datasets under 0% noise. In the BackgroundFlip experiments, we use a ResNet-32 [^14] as our base model.

We train the models with SGD with momentum, at an initial learning rate 0.1 and a momentum 0.9 with mini-batch size 100. For ResNet-32 models, the learning rate decays $\times 0.1$ at 40K and 60K steps, for a total of 80K steps. For WRN and early stopped versions of ResNet-32 models, the learning rate decays at 40K and 50K steps, for a total of 60K steps. Under regular 0% noise settings, our base ResNet-32 gets 92.5% and 68.1% classification accuracy on CIFAR-10 and 100, and the WRN-28-10 gets 95.5% and 78.2%. For the finetuning stage, we run extra 5K steps of training on the limited clean data.

We report the average test accuracy for 5 different random splits of clean and noisy labels, with 95% confidence interval in Table 1 and 2. The background classes for the 5 trials are \[0, 1, 3, 5, 7\] (CIFAR-10) and \[7, 12, 41, 62, 85\] (CIFAR-100).

Figure 3: Example weights distribution on BackgroundFlip. Left: a hyper-validation batch, with randomly flipped background noises. Right: a hyper-validation batch containing only on a single label class, with flipped background noises, averaged across all non-background classes.

Figure 4: Effect of the number of clean imaged used, on CIFAR-10 with 40% of data flipped to label 3. “ES” denotes early stopping.

Figure 5: Model test accuracy on imbalanced noisy CIFAR experiments across various noise levels using a base ResNet-32 model. “ES” denotes early stopping, and “FT” denotes finetuning.

Figure 6: Confusion matrices on CIFAR-10 UniformFlip (top) and BackgroundFlip (bottom)

Figure 7: Training curve of a ResNet-32 on CIFAR-10 BackgroundFlip under 40% noise ratio. Solid lines denote validation accuracy and dotted lines denote training. Our method is less prone to label noise overfitting.

### 4.3 Results and Discussion

The first result that draws our attention is that “Random” performs surprisingly well on the UniformFlip benchmark, outperforming all historical methods that we compared. Given that its performance is comparable with Baseline on BackgroundFlip and MNIST class imbalance, we hypothesize that random example weights act as a strong regularizer and under which the learning objective on UniformFlip is still consistent.

Regardless of the strong baseline, our method ranks the top on both UniformFlip and BackgroundFlip, showing our method is less affected by the changes in the noise type. On CIFAR-100, our method wins more than 3% compared to the state-of-the-art method.

#### Understanding the reweighting mechanism

It is beneficial to understand how our reweighting algorithm contributes to learning more robust models during training. First, we use a pre-trained model (trained at half of the total iterations without learning rate decay) and measure the example weight distribution of a randomly sampled batch of validation images, which the model has never seen. As shown in the left figure of Figure 3, our model correctly pushes most noisy images to zero weights. Secondly, we conditioned the input mini-batch to be a single non-background class and randomly flip 40% of the images to the background, and we would like to see how well our model can distinguish clean and noisy images. As shown in Figure 3 right, the model is able to reliably detect images that are flipped to the background class.

#### Robustness to overfitting noise

Throughout experimentation, we find baseline models can easily overfit to the noise in the training set. For example, shown in Table 2, applying early stopping (“ES”) helps the classification performance of “S-Model” by over 10% on CIFAR-10. Figure 6 compares the final confusion matrices of the baseline and the proposed algorithm, where a large proportion of noise transition probability is cleared in the final prediction. Figure 7 shows training curves on the BackgroundFlip experiments. After the first learning rate decay, both “Baseline” and “S-Model” quickly degrade their validation performance due to overfitting, while our model remains the same validation accuracy until termination. Note that here “S-Model” knows the oracle noise ratio in each class, and this information is not available in our method.

#### Impact of the noise level

We would like to investigate how strongly our method can perform on a variety of noise levels. Shown in Figure 5, our method only drops 6% accuracy when the noise ratio increased from 0% to 50%; whereas the baseline has dropped more than 40%. At 0% noise, our method only slightly underperforms baseline. This is reasonable since we are optimizing on the validation set, which is strictly a subset of the full training set, and therefore suffers from its own subsample bias.

#### Size of the clean validation set

When the size of the clean validation set grows larger, fine-tuning on the validation set will be a reasonble approach. Here, we make an attempt to explore the tradeoff and understand when fine-tuning becomes beneficial. Figure 4 plots the classification performance when we varied the size of the clean validation on BackgroundFlip. Surprisingly, using 15 validation images for all classes only results in a 2% drop in performance, and the overall classification performance does not grow after having more than 100 validation images. In comparison, we observe a significant drop in performance when only fine-tuning on these 15 validation images for the baselines, and the performance catches up around using 1,000 validation images (100 per class). This phenomenon suggests that in our method the clean validation acts more like a regularizer rather than a data source for parameter fine-tuning, and potentially our method can be complementary with fine-tuning based method when the size of the clean set grows larger.

## 5 Conclusion

In this work, we propose an online meta-learning algorithm for reweighting training examples and training more robust deep learning models. While various types of training set biases exist and manually designed reweighting objectives have their own bias, our automatic reweighting algorithm shows superior performance dealing with class imbalance, noisy labels, and both. Our method can be directly applied to any deep learning architecture and is expected to train end-to-end without any additional hyperparameter search. Validating on every training step is a novel setting and we show that it has links with model regularization, which can be a fruitful future research direction.

## References

## Appendix A Reweighting in an MLP

We show the complete derivation below on calculating the example weights in an MLP network.

$$
\displaystyle\frac{\partial}{\partial\epsilon_{i,t}}\mathbb{E}\left[f^{v}(\theta_{t+1}(\epsilon))\right]\Bigr|_{\epsilon_{i,t}=0}
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle\frac{1}{m}\sum_{j=1}^{m}\frac{\partial}{\partial\epsilon_{i,t}}f_{j}^{v}(\theta_{t+1}(\epsilon))\Bigr|_{\epsilon_{i,t}=0}
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle\frac{1}{m}\sum_{j=1}^{m}\frac{\partial f_{j}^{v}(\theta)}{\partial\theta}\Bigr|_{\theta=\theta_{t}}^{\top}\frac{\partial\theta_{t+1}(\epsilon_{i,t})}{\partial\epsilon_{i,t}}\Bigr|_{\epsilon_{i,t}=0}
$$
 
$$
\displaystyle\propto
$$
 
$$
\displaystyle-\frac{1}{m}\sum_{j=1}^{m}\frac{\partial f_{j}^{v}(\theta)}{\partial\theta}\Bigr|_{\theta=\theta_{t}}^{\top}\frac{\partial f_{i}(\theta)}{\partial\theta}\Bigr|_{\theta=\theta_{t}}
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle-\frac{1}{m}\sum_{j=1}^{m}\sum_{l=1}^{L}\frac{\partial f_{j}^{v}}{\partial\theta_{l}}\Bigr|_{\theta_{l}=\theta_{l,t}}^{\top}\frac{\partial f_{i}}{\partial\theta_{l}}\Bigr|_{\theta_{l}=\theta_{l,t}}
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle-\frac{1}{m}\sum_{j=1}^{m}\sum_{l=1}^{L}\vecc\left(\tilde{z}_{j,l-1}^{v}{g_{j,l}^{v}}^{\top}\right)^{\top}\vecc\left(\tilde{z}_{i,l-1}g_{i,l}^{\top}\right)
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle-\frac{1}{m}\sum_{j=1}^{m}\sum_{l=1}^{L}\sum_{p=1}^{D_{1}}\sum_{q=1}^{D_{2}}\tilde{z}_{j,l-1,p}^{v}g_{j,l,q}^{v}\tilde{z}_{i,l-1,p}g_{i,l,q}
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle-\frac{1}{m}\sum_{j=1}^{m}\sum_{l=1}^{L}\sum_{p=1}^{D_{1}}\tilde{z}_{j,l-1,p}^{v}\tilde{z}_{i,l-1,p}\sum_{q=1}^{D_{2}}g_{j,l,q}^{v}g_{i,l,q}
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle-\frac{1}{m}\sum_{j=1}^{m}\sum_{l=1}^{L}(\tilde{z}^{v}_{j,l-1}{}^{\top}\tilde{z}_{i,l-1})(g^{v}_{j,l}{}^{\top}g_{i,l}).
$$

## Appendix B Convergence of our method

This section provides the proof for Lemma 1.

###### Lemma.

Suppose the validation loss function is Lipschitz-smooth with constant $L$, and the train loss function $f_{i}$ of training data $x_{i}$ have $\sigma$ -bounded gradients. Let the learning rate $\alpha_{t}$ satisfies $\alpha_{t}\leq\frac{2n}{L\sigma^{2}}$, where $n$ is the training batch size. Then, following our algorithm, the validation loss always monotonically decreases for any sequence of training batches, namely,

$$
\displaystyle G(\theta_{t+1})\leq G(\theta_{t}),
$$

where $G(\theta)$ is the total validation loss

$$
\displaystyle G(\theta)=\frac{1}{M}\sum_{i=1}^{M}f^{v}_{i}(\theta_{t+1}(\epsilon)).
$$

Furthermore, in expectation, the equality in Eq. 26 holds only when the gradient of validation loss becomes 0 at some time step $t$, namely $\mathop{\mathbb{E}}_{t}\left[G(\theta_{t+1})\right]=G(\theta_{t})$ if and only if $\nabla G(\theta_{t})=0$, where the expectation is taking over possible training batches at time step $t$.

###### Proof.

Suppose we have a small validation set with $M$ clean data $\{x_{1},x_{2},\cdots,x_{M}\}$, each associating with a validation loss function $f_{i}(\theta)$, where $\theta$ is the parameter of the model. The overall validation loss would be,

$$
\displaystyle G(\theta)=\frac{1}{M}\sum_{i=1}^{M}f_{i}(\theta).
$$

Now, suppose we have another $N-M$ training data, $\{x_{M+1},x_{M+2},\cdots,x_{N}\}$, and we add those validation data into this set to form our large training dataset $T$, which has $N$ data in total. The overall training loss would be,

$$
\displaystyle F(\theta)=\frac{1}{M}\sum_{i=1}^{N}f_{i}(\theta).
$$

For simplicity, since $M\ll N$, we assume that the validation data is a subset of the training data. During training, we take a mini-batch $B$ of training data at each step, and $|B|=n$. Following some similar derivation as Appendix A, we have the following update rules:

$$
\displaystyle\theta_{t+1}=\theta_{t}-\frac{\alpha_{t}}{n}\sum_{i\in B}\max\left\{\nabla G^{\top}\nabla f_{i},0\right\}\nabla f_{i},
$$

where $\alpha_{t}$ is the learning rate at time-step $t$. Since all gradients are taken at $\theta_{t}$, we omit $\theta_{t}$ in our notations.

Since the validation loss $G(\theta)$ is Lipschitz-smooth, we have

$$
\displaystyle G(\theta_{t+1})\leq G(\theta_{t})+\nabla G^{\top}\Delta\theta+\frac{L}{2}\lVert\Delta\theta\rVert^{2}.
$$

Plugging our updating rule (Eq. 30),

$$
\displaystyle G(\theta_{t+1})\leq G(\theta_{t})-I_{1}+I_{2},
$$

where,

$$
\displaystyle\begin{split}I_{1}&=\frac{\alpha_{t}}{n}\sum_{i\in B}\max\{\nabla G^{\top}\nabla f_{i},0\}\nabla G^{\top}\nabla f_{i}\\
&=\frac{\alpha_{t}}{n}\sum_{i\in B}\max\{\nabla G^{\top}\nabla f_{i},0\}^{2},\\
\end{split}
$$

and,

$$
\displaystyle I_{2}
$$
 
$$
\displaystyle=\frac{L}{2}\left\lVert\frac{\alpha_{t}}{n}\sum_{i\in B}\max\{\nabla G^{\top}\nabla f_{i},0\}\nabla f_{i}\right\rVert^{2}
$$
 
$$
\displaystyle\leq\frac{L}{2}\frac{\alpha^{2}_{t}}{n^{2}}\sum_{i\in B}\left\lVert\max\left\{\nabla G^{\top}\nabla f_{i},0\right\}\nabla f_{i}\right\rVert^{2}
$$
 
$$
\displaystyle=\frac{L}{2}\frac{\alpha^{2}_{t}}{n^{2}}\sum_{i\in B}\max\left\{\nabla G^{\top}\nabla f_{i},0\right\}^{2}\left\lVert\nabla f_{i}\right\rVert^{2}
$$
 
$$
\displaystyle\leq\frac{L}{2}\frac{\alpha^{2}_{t}}{n^{2}}\sum_{i\in B}\max\left\{\nabla G^{\top}\nabla f_{i},0\right\}^{2}\sigma^{2}.
$$

The first inequality (Eq. 35) comes from the triangle inequality. The second inequality (Eq. 37) holds since $f_{i}$ has $\sigma$ -bounded gradients. If we denote $\mathcal{T}_{t}=\sum_{i\in B}\max\{\nabla G^{\top}\nabla f_{i},0\}^{2}$, where $t$ stands for the time-step $t$, then

$$
\displaystyle G(\theta_{t+1})\leq G(\theta_{t})-\frac{\alpha_{t}}{n}\mathcal{T}_{t}\left(1-\frac{L\alpha_{t}\sigma^{2}}{2n}\right).
$$

Note that by definition, $\mathcal{T}_{t}$ is non-negative, and since $\alpha_{t}\leq\frac{2n}{L\sigma^{2}}$, if follows that that $G(\theta_{t+1})\leq G(\theta_{t})$ for any $t$.

Next, we prove $\mathop{\mathbb{E}}_{t}\left[\mathcal{T}_{t}\right]=0$ if and only if $\nabla G=0$, and $\mathop{\mathbb{E}}_{t}\left[\mathcal{T}_{t}\right]>0$ if and only if $\nabla G\neq 0$, where the expectation is taken over all possible training batches at time step $t$. It is obvious that when $\nabla G=0$, $\mathop{\mathbb{E}}_{t}\left[\mathcal{T}_{t}\right]=0$. If $\nabla G\neq 0$, from the inequality below, we firstly know that there must exist a validation example $x_{j,0\leq j\leq M}$ such that $\nabla G^{\top}\nabla f_{j}>0$,

$$
\displaystyle 0<\lVert\nabla G\rVert^{2}=\nabla G^{\top}\nabla G=\frac{1}{M}\sum_{i=1}^{M}\nabla G^{\top}\nabla f_{i}.
$$

Secondly, there is a non-zero possibility $p$ to sample a training batch $B$ such that it contains this data $x_{j}$. Also noticing that $\mathcal{T}_{t}$ is a non-negative random variable, we have,

$$
\displaystyle\begin{split}\mathop{\mathbb{E}}_{t}\left[\mathcal{T}_{t}\right]&\geq p\sum_{i\in B}\max\{\nabla G^{\top}\nabla f_{i},0\}^{2}\\
&\geq p\max\{\nabla G^{\top}\nabla f_{j},0\}^{2}\\
&=p\left(\nabla G^{\top}\nabla f_{j}\right)^{2}>0.\end{split}
$$

Therefore, if we take expectation over the training batch on both sides of Eq. 38, we can conclude that,

$$
\displaystyle\mathop{\mathbb{E}}_{t}\left[G(\theta_{t+1})\right]\leq G(\theta_{t}),
$$

where the equality holds if and only if $\nabla G=0$. This finishes our proof for Lemma 1. ∎

## Appendix C Convergence rate of our method

This section provides proof for Theorem 2.

###### Theorem.

Suppose $G$, $f_{i}$ and $\alpha_{t}$ satisfy the aforementioned conditions, then Algorithm 1 achieves $\mathbb{E}\left[\lVert\nabla G(\theta_{t})\rVert^{2}\right]\leq\epsilon$ in $O(1/\epsilon^{2})$ steps. More specifically,

$$
\displaystyle\min\limits_{0<t<T}\mathbb{E}\left[\lVert\nabla G(\theta_{t})\rVert^{2}\right]\leq\frac{C}{\sqrt{T}},
$$

where $C$ is some constant independent of the convergence process.

###### Proof.

From the proof of Lemma 1, we have

$$
\displaystyle\begin{split}&\frac{\alpha_{t}}{n}\left(1-\frac{L\alpha_{t}\sigma^{2}}{2n}\right)\mathop{\mathbb{E}}_{0\sim t}\left[\mathcal{T}_{t}\right]\\
\leq&\mathop{\mathbb{E}}_{0\sim t-1}\left[G(\theta_{t})\right]-\mathop{\mathbb{E}}_{0\sim t}\left[G(\theta_{t+1})\right].\end{split}
$$

If we let $\alpha_{t}$ to be a constant $\alpha<\frac{2n}{L\sigma^{2}}$ (or a decay positive sequence upper bounded by $\alpha$), and let $\kappa=\left(1-\frac{L\alpha\sigma^{2}}{2n}\right)\alpha/n>0$, then we have,

$$
\displaystyle\begin{split}\kappa\sum_{t=0}^{T}\mathop{\mathbb{E}}_{0\sim t}\left[\mathcal{T}_{t}\right]&\leq\mathop{\mathbb{E}}_{0}\left[G(\theta_{0})\right]-\mathop{\mathbb{E}}_{0\sim T}\left[G(\theta_{T+1})\right]\\
&\leq G(\theta_{0})-G(\theta^{*}),\end{split}
$$

where $G(\theta^{*})$ is the global minimum of function $G$. Therefore, it is obvious to see that there exist a time-step $0\leq\tau\leq T$ such that,

$$
\displaystyle\mathop{\mathbb{E}}_{0\sim\tau}\left[\mathcal{T}_{\tau}\right]\leq\frac{G(\theta_{0})-G(\theta^{*})}{\kappa T}.
$$

We next prove that for this time-step $\tau$, the gradient square $\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\lVert\nabla G(\theta_{\tau})\rVert^{2}\right]$ is smaller than $O(1/\sqrt{T})$. Considering such $M$ training batches $B_{1},B_{2},\cdots,B_{M}$ such that $B_{i}$ is guaranteed to contain $x_{i}$. We know that those batches have non-zero sampling probability, denoted as $p_{1},p_{2},\cdots,p_{M}$. We also denote $p=\min\{p_{1},p_{2},\cdots,p_{M}\}$. Now, we have,

$$
\displaystyle M\mathop{\mathbb{E}}_{0\sim\tau}\left[\mathcal{T}_{\tau}\right]
$$
 
$$
\displaystyle=\mathop{\mathbb{E}}_{0\sim\tau-1}\left[M\mathop{\mathbb{E}}_{\tau}\left[\mathcal{T}_{\tau}\right]\right]
$$
 
$$
\displaystyle\geq\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\sum_{k=1}^{M}p_{k}\sum_{i\in B_{k}}\max\{\nabla G^{\top}\nabla f_{i},0\}^{2}\right]
$$
 
$$
\displaystyle\geq p\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\sum_{k=1}^{M}\sum_{i\in B_{k}}\max\{\nabla G^{\top}\nabla f_{i},0\}^{2}\right]
$$
 
$$
\displaystyle\geq p\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\sum_{i=1}^{M}\max\{\nabla G^{\top}\nabla f_{i},0\}^{2}\right]
$$
 
$$
\displaystyle=p\sum_{i=1}^{M}\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\max\{\nabla G^{\top}\nabla f_{i},0\}^{2}\right]
$$
 
$$
\displaystyle\geq p\sum_{i=1}^{M}\left(\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\max\{\nabla G^{\top}\nabla f_{i},0\}\right]\right)^{2}
$$
 
$$
\displaystyle\geq\frac{p}{M}\left(\sum_{i=1}^{M}\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\max\{\nabla G^{\top}\nabla f_{i},0\}\right]\right)^{2}.
$$

The inequality in Eq. 47 comes from the non-negativeness of $\mathcal{T}_{t}$, the inequality in Eq. 51 comes from the property of expectation, and the final inequality in Eq. 52 comes from the Cauchy-Schwartz inequality. Therefore,

$$
\displaystyle\begin{split}&\sum_{i=1}^{M}\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\max\{\nabla G^{\top}\nabla f_{i},0\}\right]\\
\leq&M\sqrt{\frac{(G(\theta_{0})-G(\theta^{*}))}{p\kappa}}\sqrt{\frac{1}{T}},\end{split}
$$

and so,

$$
\displaystyle\begin{split}&\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\lVert\nabla G(\theta_{\tau})\rVert^{2}\right]\\
=&\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\nabla G^{\top}\nabla G\right]\\
=&\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\nabla G^{\top}\left(\frac{\sum_{i=0}^{M}\nabla f_{i}}{M}\right)\right]\\
\leq&\frac{1}{M}\sum_{i=1}^{M}\mathop{\mathbb{E}}_{0\sim\tau-1}\left[\max\{\nabla G^{\top}\nabla f_{i},0\}\right]\\
\leq&\sqrt{\frac{G(\theta_{0})-G(\theta^{*})}{p\kappa}}\sqrt{\frac{1}{T}}.\end{split}
$$

Therefore, we can conclude that conclude that our algorithm can always achieve $\min\limits_{0<t<T}\mathbb{E}\left[\lVert\nabla G(\theta_{t})\rVert^{2}\right]\leq O(\sqrt{1/T})$ in $T$ steps, and this finishes our proof of Theorem 2. ∎

[^1]: Abadi, Martín, Barham, Paul, Chen, Jianmin, Chen, Zhifeng, Davis, Andy, Dean, Jeffrey, Devin, Matthieu, Ghemawat, Sanjay, Irving, Geoffrey, Isard, Michael, Kudlur, Manjunath, Levenberg, Josh, Monga, Rajat, Moore, Sherry, Murray, Derek Gordon, Steiner, Benoit, Tucker, Paul A., Vasudevan, Vijay, Warden, Pete, Wicke, Martin, Yu, Yuan, and Zheng, Xiaoqiang. Tensorflow: A system for large-scale machine learning. In *12th USENIX Symposium on Operating Systems Design and Implementation, OSDI*, 2016.

[^2]: Andrychowicz, Marcin, Denil, Misha, Colmenarejo, Sergio Gomez, Hoffman, Matthew W., Pfau, David, Schaul, Tom, and de Freitas, Nando. Learning to learn by gradient descent by gradient descent. In *Advances in Neural Information Processing Systems, NIPS*, 2016.

[^3]: Angluin, Dana and Laird, Philip. Learning from noisy examples. *Machine Learning*, 2(4):343–370, Apr 1988. ISSN 1573-0565.

[^4]: Azadi, Samaneh, Feng, Jiashi, Jegelka, Stefanie, and Darrell, Trevor. Auxiliary image regularization for deep cnns with noisy labels. In *Proceedings of the 4th International Conference on Learning Representation, ICLR*, 2016.

[^5]: Bengio, Yoshua, Louradour, Jérôme, Collobert, Ronan, and Weston, Jason. Curriculum learning. In *Proceedings of the 26th Annual International Conference on Machine Learning, ICML*, 2009.

[^6]: Chang, Haw-Shiuan, Learned-Miller, Erik G., and McCallum, Andrew. Active bias: Training more accurate neural networks by emphasizing high variance samples. In *Advances in Neural Information Processing Systems, NIPS*, 2017.

[^7]: Chawla, Nitesh V., Bowyer, Kevin W., Hall, Lawrence O., and Kegelmeyer, W. Philip. SMOTE: synthetic minority over-sampling technique. *J. Artif. Intell. Res.*, 16:321–357, 2002.

[^8]: Chen, Xinlei and Gupta, Abhinav. Webly supervised learning of convolutional networks. In *Proceedings of the 2015 IEEE International Conference on Computer Vision, ICCV*, 2015.

[^9]: Cordts, Marius, Omran, Mohamed, Ramos, Sebastian, Rehfeld, Timo, Enzweiler, Markus, Benenson, Rodrigo, Franke, Uwe, Roth, Stefan, and Schiele, Bernt. The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR*, 2016.

[^10]: Dong, Qi, Gong, Shaogang, and Zhu, Xiatian. Class rectification hard mining for imbalanced deep learning. In *Proceedings of the IEEE International Conference on Computer Vision, ICCV*, 2017.

[^11]: Finn, Chelsea, Abbeel, Pieter, and Levine, Sergey. Model-agnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the 34th International Conference on Machine Learning, ICML*, 2017.

[^12]: Freund, Yoav and Schapire, Robert E. A decision-theoretic generalization of on-line learning and an application to boosting. *J. Comput. Syst. Sci.*, 55(1):119–139, 1997.

[^13]: Goldberger, Jacob and Ben-Reuven, Ehud. Training deep neural-networks using a noise adaptation layer. In *Proceedings of the 5th International Conference on Learning Representation, ICLR*, 2017.

[^14]: He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR*, 2016.

[^15]: Hendrycks, Dan, Mazeika, Mantas, Wilson, Duncan, and Gimpel, Kevin. Using trusted data to train deep networks on labels corrupted by severe noise. *CoRR*, abs/1802.05300, 2018.

[^16]: Huang, Chen, Li, Yining, Loy, Chen Change, and Tang, Xiaoou. Learning deep representation for imbalanced classification. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR*, 2016.

[^17]: Jiang, Lu, Meng, Deyu, Zhao, Qian, Shan, Shiguang, and Hauptmann, Alexander G. Self-paced curriculum learning. In *Proceedings of the 29th AAAI Conference on Artificial Intelligence*, 2015.

[^18]: Jiang, Lu, Zhou, Zhengyuan, Leung, Thomas, Li, Li-Jia, and Fei-Fei, Li. Mentornet: Regularizing very deep neural networks on corrupted labels. *CoRR*, abs/1712.05055, 2017.

[^19]: Kahn, Herman and Marshall, Andy W. Methods of reducing sample size in monte carlo computations. *Journal of the Operations Research Society of America*, 1(5):263–278, 1953.

[^20]: Khan, Salman Hameed, Bennamoun, Mohammed, Sohel, Ferdous Ahmed, and Togneri, Roberto. Cost sensitive learning of deep feature representations from imbalanced data. *CoRR*, abs/1508.03422, 2015.

[^21]: Koh, Pang Wei and Liang, Percy. Understanding black-box predictions via influence functions. In *Proceedings of the 34th International Conference on Machine Learning, ICML*, 2017.

[^22]: Kumar, M. Pawan, Packer, Benjamin, and Koller, Daphne. Self-paced learning for latent variable models. In *Advances in Neural Information Processing Systems, NIPS*, 2010.

[^23]: Lake, Brenden M., Ullman, Tomer D., Tenenbaum, Joshua B., and Gershman, Samuel J. Building machines that learn and think like people. *Behav Brain Sci*, 40:e253, Jan 2017.

[^24]: Li, Yuncheng, Yang, Jianchao, Song, Yale, Cao, Liangliang, Luo, Jiebo, and Li, Li-Jia. Learning from noisy labels with distillation. In *Proceedings of the IEEE International Conference on Computer Vision, ICCV*, 2017.

[^25]: Lin, Tsung-Yi, Goyal, Priya, Girshick, Ross B., He, Kaiming, and Dollár, Piotr. Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision, ICCV*, 2017.

[^26]: Lorraine, Jonathan and Duvenaud, David. Stochastic hyperparameter optimization through hypernetworks. *CoRR*, abs/1802.09419, 2018.

[^27]: Ma, Fan, Meng, Deyu, Xie, Qi, Li, Zina, and Dong, Xuanyi. Self-paced co-training. In *Proceedings of the 34th International Conference on Machine Learning, ICML*, 2017.

[^28]: Malisiewicz, Tomasz, Gupta, Abhinav, and Efros, Alexei A. Ensemble of exemplar-svms for object detection and beyond. In *Proceedings of the IEEE International Conference on Computer Vision, ICCV*, 2011.

[^29]: Muñoz-González, Luis, Biggio, Battista, Demontis, Ambra, Paudice, Andrea, Wongrassamee, Vasin, Lupu, Emil C., and Roli, Fabio. Towards poisoning of deep learning algorithms with back-gradient optimization. In *Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, AISec@CCS*, 2017.

[^30]: Natarajan, Nagarajan, Dhillon, Inderjit S., Ravikumar, Pradeep, and Tewari, Ambuj. Learning with noisy labels. In *Advances in Neural Information Processing Systems, NIPS*, 2013.

[^31]: Ravi, Sachin and Larochelle, Hugo. Optimization as a model for few-shot learning. In *Proceedings of the 5th International Conference on Learning Representations, ICLR*, 2017.

[^32]: Reddi, Sashank J., Hefny, Ahmed, Sra, Suvrit, Póczos, Barnabás, and Smola, Alexander J. Stochastic variance reduction for nonconvex optimization. In *Proceedings of the 33rd International Conference on Machine Learning, ICML*, 2016.

[^33]: Reed, Scott E., Lee, Honglak, Anguelov, Dragomir, Szegedy, Christian, Erhan, Dumitru, and Rabinovich, Andrew. Training deep neural networks on noisy labels with bootstrapping. *CoRR*, abs/1412.6596, 2014.

[^34]: Ren, Mengye, Triantafillou, Eleni, Ravi, Sachin, Snell, Jake, Swersky, Kevin, Tenenbaum, Joshua B., Larochelle, Hugo, and Zemel, Richard S. Meta learning for few-shot semi-supervised classification. In *Proceedings of the 6th International Conference on Learning Representations, ICLR*, 2018.

[^35]: Russakovsky, Olga, Deng, Jia, Su, Hao, Krause, Jonathan, Satheesh, Sanjeev, Ma, Sean, Huang, Zhiheng, Karpathy, Andrej, Khosla, Aditya, Bernstein, Michael, Berg, Alexander C., and Fei-Fei, Li. ImageNet Large Scale Visual Recognition Challenge. *International Journal of Computer Vision, IJCV*, 115(3):211–252, 2015.

[^36]: Sukhbaatar, Sainbayar and Fergus, Rob. Learning from noisy labels with deep neural networks. *CoRR*, abs/1406.2080, 2014.

[^37]: Thrun, Sebastian and Pratt, Lorien. *Learning to Learn*. Springer, 1998.

[^38]: Ting, Kai Ming. A comparative study of cost-sensitive boosting algorithms. In *Proceedings of the 17th International Conference on Machine Learning, ICML*, 2000.

[^39]: Vahdat, Arash. Toward robustness against label noise in training deep discriminative neural networks. In *Advances in Neural Information Processing Systems, NIPS*, 2017.

[^40]: Wang, Yixin, Kucukelbir, Alp, and Blei, David M. Robust probabilistic modeling with bayesian data reweighting. In *Proceedings of the 34th International Conference on Machine Learning, ICML*, 2017.

[^41]: Wu, Yuhuai, Ren, Mengye, Liao, Renjie, and Grosse, Roger B. Understanding short-horizon bias in stochastic meta-optimization. In *Proceedings of the 6th International Conference on Learning Representations, ICLR*, 2018.

[^42]: Xiao, Tong, Xia, Tian, Yang, Yi, Huang, Chang, and Wang, Xiaogang. Learning from massive noisy labeled data for image classification. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR*, 2015.

[^43]: Zagoruyko, Sergey and Komodakis, Nikos. Wide residual networks. In *Proceedings of the British Machine Vision Conference, BMVC*, 2016.

[^44]: Zhang, Chiyuan, Bengio, Samy, Hardt, Moritz, Recht, Benjamin, and Vinyals, Oriol. Understanding deep learning requires rethinking generalization. In *Proceedings of the 5th International Conference on Learning Representations, ICLR*, 2017.