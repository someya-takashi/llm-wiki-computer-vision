---
title: "SAM 3: Segment Anything with Concepts"
source: "https://ar5iv.labs.arxiv.org/html/2511.16719"
author:
published:
created: 2026-05-26
description:
tags:
  - "clippings"
---
\]Meta Superintelligence Labs \[\*\]core contributor \[∘\]intern \[†\]project lead \[\]order is random within groups L\[1\]>m#1 R\[1\]>m#1

Nicolas Carion    Laura Gustafson    Yuan-Ting Hu    Shoubhik Debnath    Ronghang Hu    Didac Suris    Chaitanya Ryali    Kalyan Vasudev Alwala    Haitham Khedr    Andrew Huang    Jie Lei    Tengyu Ma    Baishan Guo    Arpit Kalla    Markus Marks    Joseph Greer    Meng Wang    Peize Sun    Roman Rädle    Triantafyllos Afouras    Effrosyni Mavroudi    Katherine Xu    Tsung-Han Wu    Yu Zhou    Liliane Momeni    Rishi Hazra    Shuangrui Ding    Sagar Vaze    Francois Porcher    Feng Li    Siyuan Li    Aishwarya Kamath    Ho Kei Cheng    Piotr Dollár    Nikhila Ravi    Kate Saenko    Pengchuan Zhang    Christoph Feichtenhofer \[

###### Abstract

We present Segment Anything Model (SAM) 3, a unified model that detects, segments, and tracks objects in images and videos based on *concept prompts*, which we define as either short noun phrases (e.g., “yellow school bus”), image exemplars, or a combination of both. Promptable Concept Segmentation (PCS) takes such prompts and returns segmentation masks and unique identities for all matching object instances. To advance PCS, we build a scalable data engine that produces a high-quality dataset with 4M unique concept labels, including hard negatives, across images and videos. Our model consists of an image-level detector and a memory-based video tracker that share a single backbone. Recognition and localization are decoupled with a presence head, which boosts detection accuracy. SAM 3 doubles the accuracy of existing systems in both image and video PCS, and improves previous SAM capabilities on visual segmentation tasks. We open source SAM 3 along with our new Segment Anything with Concepts (SA-Co) benchmark for promptable concept segmentation.

## 1 Introduction

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.16719/assets/x1.png)

Figure 1: SAM 3 improves over SAM 2 on promptable visual segmentation with clicks (left) and introduces the new promptable concept segmentation capability (right). Users can segment all instances of a visual concept specified by a short noun phrase, image exemplars (positive or negative), or a combination of both.

The ability to find and segment anything in a visual scene is foundational for multimodal AI, powering applications in robotics, content creation, augmented reality, data annotation, and broader sciences. The SAM series (sam; sam2) introduced the promptable segmentation task for images and videos, focusing on *Promptable Visual Segmentation* (PVS) with points, boxes or masks to segment a single object per prompt. While these methods achieved a breakthrough, they did not address the general task of finding and segmenting all instances of a concept appearing anywhere in the input (e.g., all “cats” in a video).

To fill this gap, we present SAM 3, a model that achieves a step change in promptable segmentation in images and videos, improving PVS relative to SAM 2 and setting a new standard for *Promptable Concept Segmentation (PCS)*. We formalize the PCS task (§2) as taking text and/or image exemplars as input, and predicting instance and semantic masks for every single object matching the concept, while preserving object identities across video frames (see Fig.˜1). To focus on recognizing atomic visual concepts, we constrain text to simple noun phrases (NPs) such as “red apple” or “striped cat”. While SAM 3 is not designed for long referring expressions or queries requiring reasoning, we show that it can be straightforwardly combined with a Multimodal Large Language Model (MLLM) to handle more complex language prompts. Consistent with previous SAM versions, SAM 3 is fully interactive, allowing users to resolve ambiguities by adding refinement prompts to guide the model towards their intended output.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.16719/assets/x2.png)

Figure 2: Examples of SAM 3 improving segmentation of open-vocabulary concepts compared to OWLv2 ( minderer2024scalingopenvocabularyobjectdetection ), on the SA-Co benchmark. See § LABEL:sec:results\_qualitative for additional SAM 3 outputs.

Our *model* (§3) consists of a detector and a tracker that share a vision encoder (bolya2025PerceptionEncoder). The detector is a DETR-based (carion2020end) model conditioned on text, geometry, and image exemplars. To address the challenge of open-vocabulary concept detection, we introduce a separate presence head to decouple recognition and localization, which is especially effective when training with challenging negative phrases. The tracker inherits the SAM 2 transformer encoder-decoder architecture, supporting video segmentation and interactive refinement. The decoupled design for detection and tracking avoids task conflict, as the detector needs to be identity agnostic, while the tracker’s main objective is to separate identities in the video.

To unlock major performance gains, we build a human- and model-in-the-loop *data engine* (§4) that annotates a large and diverse training dataset. We innovate upon prior data engines in three key ways: (i) media curation: we curate more diverse media domains than past approaches that rely on homogeneous web sources, (ii) label curation: we significantly increase label diversity and difficulty by leveraging an ontology and multimodal LLMs as “AI annotators” to generate noun phrases and hard negatives, (iii) label verification: we double annotation throughput by fine-tuning MLLMs to be effective “AI verifiers” that achieve near-human accuracy.

Starting from noisy media-phrase-mask pseudo-labels, our data engine checks mask quality and exhaustivity using both human and AI verifiers, filtering out correctly labeled examples and identifying challenging error cases. Human annotators then focus on fixing these errors by manually correcting masks. This enables us to annotate high-quality training data with 4M unique phrases and 52M masks, and a synthetic dataset with 38M phrases and 1.4B masks. We additionally create the Segment Anything with Concepts (SA-Co) *benchmark* for PCS (§5) containing 207K unique concepts with exhaustive masks in 120K images and 1.7K videos, $>50\times$ more concepts than existing benchmarks.

Our *experiments* (§6) show that SAM 3 sets a new state-of-the-art in promptable segmentation, e.g., reaching a zero-shot mask AP of 48.8 on LVIS vs. the current best of 38.5, surpassing baselines on our new SA-Co benchmark by at least $2\times$ (see examples in Fig.˜2), and improving upon SAM 2 on visual prompts. Ablations (§A) verify that the choice of backbone, novel presence head, and adding hard negatives all boost results, and establish scaling laws on the PCS task for both our high-quality and synthetic datasets. We open-source the SA-Co benchmark and release the SAM 3 checkpoints and inference code. On an H200 GPU, SAM 3 runs in 30 ms for a single image with 100+ detected objects. In video, the inference latency scales with the number of objects, sustaining near real-time performance for $\sim 5$ concurrent objects. We review related work in §7; next, we dive into the task.

## 2 Promptable Concept Segmentation (PCS)

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.16719/assets/x3.png)

Figure 3: Illustration of supported initial and optional interactive refinement prompts in the PCS task.

We define the Promptable Concept Segmentation task as follows: given an image or short video ($\leq$ 30 secs), detect, segment and track all instances of a visual concept specified by a short text phrase, image exemplars, or a combination of both. We restrict concepts to those defined by simple noun phrases (NPs) consisting of a noun and optional modifiers. Noun-phrase prompts (when provided) are global to all frames of the image/video, while image exemplars can be provided on individual frames as positive or negative bounding boxes to iteratively refine the target masks (see Fig.˜3).

All prompts must be consistent in their category definition, or the model’s behavior is undefined; e.g., “fish” cannot be refined with subsequent exemplar prompts of just the tail; instead the text prompt should be updated. Exemplar prompts are particularly useful when the model initially misses some instances, or when the concept is rare.

Our vocabulary includes any simple noun phrase groundable in a visual scene, which makes the task intrinsically ambiguous. There can be multiple interpretations of phrases arising from polysemy (“mouse” device vs. animal), subjective descriptors (“cozy”, “large”), vague or context-dependent phrases that may not even be groundable (“brand identity”), boundary ambiguity (whether ’mirror’ includes the frame) and factors such as occlusion and blur that obscure the extent of the object. While similar issues appear in large closed-vocabulary corpora (e.g., LVIS (lvis)), they are alleviated by carefully curating the vocabulary and setting a clear definition of all the classes of interest. We address the ambiguity problem by collecting test annotations from three experts, adapting the evaluation protocol to allow multiple valid interpretations (§LABEL:app:metrics), designing the data pipeline/guidelines to minimize ambiguity in annotation, and an ambiguity module in the model (§LABEL:app:image\_implementation\_details).

## 3 Model

SAM 3 is a generalization of SAM 2, supporting the new PCS task (§2) as well as the PVS task. It takes concept prompts (simple noun phrases, image exemplars) or visual prompts (points, boxes, masks) to define the objects to be (individually) segmented spatio-temporally. Image exemplars and visual prompts can be iteratively added on individual frames to refine the target masks—false positive and false negative objects can be removed or added respectively using image exemplars and an individual mask(let) can be refined using PVS in the style of SAM 2. Our architecture is broadly based on the SAM and (M)DETR (carion2020end; kamath2021mdetr) series. Fig.˜4 shows the SAM 3 architecture, consisting of a dual encoder-decoder transformer—a detector for image-level capabilities—which is used in combination with a tracker and memory for video. The detector and tracker ingest vision-language inputs from an aligned Perception Encoder (PE) backbone (bolya2025PerceptionEncoder). We present an overview below, see §LABEL:app:sec:model for details.

#### Detector Architecture

The architecture of the detector follows the general DETR paradigm. The image and text prompt are first encoded by PE and image exemplars, if present, are encoded by an exemplar encoder. We refer to the image exemplar tokens and text tokens jointly as “prompt tokens”. The fusion encoder then accepts the unconditioned embeddings from the image encoder and conditions them by cross-attending to the prompt tokens. The fusion is followed by a DETR-like decoder, where learned object queries cross-attend to the conditioned image embeddings from the fusion encoder.

Each decoder layer predicts a classification logit for each object query (in our case, a binary label of whether the object corresponds to the prompt), and a delta from the bounding box predicted by the previous level, following deformabledetr. We use box-region-positional bias (plaindetr) to help focalize the attention on each object, but unlike recent DETR models, we stick to vanilla attention. During training, we adopt dual supervision from DAC-DETR (NEURIPS2023\_edd0d433), and the Align loss (aligndetr). The mask head is adapted from MaskFormer (cheng2021maskformer). In addition, we also have a semantic segmentation head, which predicts a binary label for every pixel in the image, indicating whether or not it corresponds to the prompt. See §LABEL:app:sec:model for details.

#### Presence Token

It can be difficult for each of the proposal queries to both recognize (what) and localize (where) an object in the image/frame. For the recognition component, contextual cues from the entire image are important. However, forcing proposal queries to understand the global context can be counterproductive, as it conflicts with the inherently local nature of the localization objective. We decouple the recognition and localization steps by introducing a learned global presence token. This token is solely responsible for predicting whether the target concept in the form of a noun phrase (NP) is present in the image/frame, i.e. $p(\text{NP is present in input})$. Each proposal query $q_{i}$ only needs to solve the localization problem $p(q_{i}\penalty 10000\ \text{is a match}\penalty 10000\ |\penalty 10000\ \text{NP is present in input})$. The final score for each proposal query is the product of its own score and the presence score.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.16719/assets/x4.png)

Figure 4: SAM 3 architecture overview. See LABEL:fig:arch\_detailed for a more detailed diagram.

#### Image Exemplars and Interactivity

SAM 3 supports image exemplars, given as a pair—a bounding box and an associated binary label (positive or negative)—which can be used in isolation or to supplement the text prompt. The model then detects all the instances that match the prompt. For example, given a positive bounding box on a dog, the model will detect *all* dogs in the image. This is different from the PVS task in SAM 1 and 2, where a visual prompt yields only a single object instance. Each image exemplar is encoded separately by the exemplar encoder using an embedding for the position, an embedding for the label, and ROI-pooled visual features, then concatenated and processed by a small transformer. The resulting prompt is concatenated to the text prompt to comprise the prompt tokens. Image exemplars can be interactively provided based on errors in current detections to refine the output.

#### Tracker and Video Architecture

Given a video and a prompt $P$, we use the detector and a tracker (see Fig.˜4) to detect and track objects corresponding to the prompt throughout the video. On each frame, the detector finds new objects $\mathcal{O}_{t}$ and the tracker propagates masklets $\mathcal{M}_{t-1}$ (spatial-temporal masks) from frames at the previous time $t-1$ to their new locations $\mathcal{\hat{M}}_{t}$ on the current frame at time $t$. We use a matching function to associate propagated masklets $\mathcal{\hat{M}}_{t}$ with new object masks emerging in the current frame $\mathcal{O}_{t}$,

$$
\mathcal{\hat{M}}_{t}=\mathrm{propagate}\left(\mathcal{M}_{t-1}\right),\penalty 10000\ \penalty 10000\ \penalty 10000\ \penalty 10000\ \penalty 10000\ \mathcal{O}_{t}=\mathrm{detect}\left(I_{t},P\right),\penalty 10000\ \penalty 10000\ \penalty 10000\ \penalty 10000\ \penalty 10000\ \mathcal{M}_{t}=\mathrm{match\_and\_update}\left(\mathcal{\hat{M}}_{t},\mathcal{O}_{t}\right).\vskip-2.0pt
$$

#### Tracking an Object with SAM 2 Style Propagation

A masklet is initialized for every object detected on the first frame. Then, on each subsequent frame, the tracker module predicts the new masklet locations $\mathcal{\hat{M}}_{t}$ of those already-tracked objects based on their previous locations $\mathcal{M}_{t-1}$ through a single-frame propagation step similar to the video object segmentation task in SAM 2. The tracker shares the same image/frame encoder (PE backbone) as the detector. After training the detector, we freeze PE and train the tracker as in SAM 2, including a prompt encoder, mask decoder, memory encoder, and a memory bank that encodes the object’s appearance using features from the past frames and conditioning frames (frames where the object is first detected or user-prompted). The memory encoder is a transformer with self-attention across visual features on the current frame and cross-attention from the visual features to the spatial memory features in the memory bank. We describe details of our video approach in §LABEL:sec:dense\_tracking\_heuristics.

During inference, we only retain frames where the object is confidently present in the memory bank. The mask decoder is a two-way transformer between the encoder hidden states and the output tokens. To handle ambiguity, we predict three output masks for every tracked object on each frame along with their confidence, and select the most confident output as the predicted mask on the current frame.

#### Matching and Updating Based on Detections

After obtaining the tracked masks $\mathcal{\hat{M}}_{t}$, we match them with the current frame detections $\mathcal{O}_{t}$ through a simple IoU based *matching function* (§LABEL:sec:dense\_tracking\_heuristics) and add them to $\mathcal{M}_{t}$ on the current frame. We further spawn new masklets for all newly detected objects that are not matched. The merging might suffer from ambiguities, especially in crowded scenes. We address this with two temporal disambiguation strategies outlined next.

First, we use temporal information in the form of a masklet detection score (§LABEL:sec:dense\_tracking\_heuristics) to measure how consistently a masklet is matched to a detection within a temporal window (based on the number of past frames where it was matched to a detection). If a masklet’s detection score falls below a threshold, we suppress it. Second, we use the detector outputs to resolve specific failure modes of the tracker due to occlusions or distractors. We periodically re-prompt the tracker with high-confidence detection masks $\mathcal{O}_{t}$, replacing the tracker’s own predictions $\mathcal{\hat{M}}_{t}$. This ensures that the memory bank has recent and reliable references (other than the tracker’s own predictions).

#### Instance Refinement with Visual Prompts

After obtaining the initial set of masks (or masklets), SAM 3 allows refining individual masks(lets) using positive and negative clicks. Specifically, given the user clicks, we apply the prompt encoder to encode them, and feed the encoded prompt into the mask decoder to predict an adjusted mask. In videos the mask is then propagated across the entire video to obtain a refined masklet.

#### Training Stages

We train SAM 3 in four stages that progressively add data and capabilities: 1) Perception Encoder (PE) pre-training, 2) detector pre-training, 3) detector fine-tuning, and 4) tracker training with a frozen backbone. See §LABEL:sec:app\_training\_stages for details.

## 4 Data Engine

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.16719/assets/x5.png)

Figure 5: Overview of the final SAM 3 data engine. See § LABEL:sec:training\_data\_details for details of collected data.

Achieving a step change in PCS with SAM 3 requires training on a large, diverse set of concepts and visual domains, beyond existing datasets (see LABEL:fig:sac\_traindata\_stats). We build an efficient data engine that iteratively generates annotated data via a feedback loop with SAM 3, human annotators, and *AI annotators*, actively mining media-phrase pairs on which the current version of SAM 3 fails to produce high-quality training data to further improve the model. By delegating certain tasks to AI annotators—models that match or surpass human accuracy—we more than double the throughput compared to a human-only annotation pipeline. We develop the data engine in four phases, with each phase increasing the use of AI models to steer human effort to the most challenging failure cases, alongside expanding visual domain coverage. Phases 1-3 focus only on images, with Phase 4 expanding to videos. We describe the key steps here; details and metrics are in §LABEL:app:sec:data-engine.

#### Data Engine Components ()

Media inputs (image or video) are mined from a large pool with the help of a curated ontology. An AI model proposes noun phrases (NPs) describing visual concepts, followed by another model (e.g., SAM 3) that generates candidate instance masks for each proposed NP. The proposed masks are verified by a two-step process: first, in Mask Verification (MV) annotators accept or reject masks based on their quality and relevance to the NP. Second, in Exhaustivity Verification (EV) annotators check if all instances of the NP have been masked in the input. Any media-NP pairs that did not pass the exhaustivity check are sent to a manual correction stage, where humans add, remove or edit masks (using SAM 1 in a browser based tool), or use “group” masks for small, hard to separate objects. Annotators may reject ungroundable or ambiguous phrases.

#### Phase 1: Human Verification

We first randomly sample images and NP proposal with a simple captioner and parser. The initial mask proposal model is SAM 2 prompted with the output of an off-the-shelf open-vocabulary detector, and initial verifiers are human. In this phase, we collected 4.3M image-NP pairs as the initial SA-Co/HQ dataset. We train SAM 3 on this data and use it as the mask proposal model for the next phase.

#### Phase 2: Human + AI Verification

In this next phase, we use human accept/reject labels from the MV and EV tasks collected in Phase 1 to fine-tune Llama 3.2 (llama3) to create AI verifiers that automatically perform the MV and EV tasks. These models receive image-phrase-mask triplets and output multiple-choice ratings of mask quality or exhaustivity. This new auto-verification process allows our human effort to be focused on the most challenging cases. We continue to re-train SAM 3 on newly collected data and update it 6 times. As SAM 3 and AI verifiers improve, a higher proportion of labels are auto-generated, further accelerating data collection. The introduction of AI verifiers for MV and EV roughly doubles the data engine’s throughput vs. human annotators. We refer to §LABEL:sec:efficient\_domain\_exp for detailed analysis of how AI verifiers improve the data engine’s throughput. We further upgrade the NP proposal step to a Llama-based pipeline that also proposes hard negative NPs adversarial to SAM 3. Phase 2 adds 122M image-NP pairs to SA-Co/HQ.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.16719/assets/x6.png)

Figure 6: Example video (top) and images (bottom) from SA-Co with annotated phrases and instance masks/IDs.

#### Phase 3: Scaling and Domain Expansion

In the third phase, we use AI models to mine increasingly challenging cases and broaden domain coverage in SA-Co/HQ to 15 datasets (LABEL:tab:images\_hq). A *domain* is a unique distribution of text and visual data. In new domains, the MV AI verifier performs well zero-shot, but the EV AI verifier needs to be improved with modest domain-specific human supervision. We also expand concept coverage to long-tail, fine-grained concepts by extracting NPs from the image alt-text where available and by mining concepts from a 22.4M node *SA-Co ontology* (§LABEL:sec:d2\_ontology) based on Wikidata (17 top-level categories, 72 sub-categories). We iterate SAM 3 training 7 times and AI verifiers 3 times, and add 19.5M image-NP pairs to SA-Co/HQ.

#### Phase 4: Video Annotation

This phase extends the data engine to video. We use a mature image SAM 3 to collect targeted quality annotations that capture video-specific challenges. The data mining pipeline applies scene/motion filters, content balancing, ranking, and targeted searches. Video frames are sampled (randomly or by object density) and sent to the image annotation flow (from phase 3). *Masklets* (spatio-temporal masks) are produced with SAM 3 (now extended to video) and post-processed via deduplication and removal of trivial masks. Because video annotation is more difficult, we concentrate humans on likely failures by favoring clips with many crowded objects and tracking failures. The collected video data SA-Co/VIDEO consists of 52.5K videos and 467K masklets. See §LABEL:appendix:vde for details.

## 5 Segment Anything with Concepts (SA-Co) Dataset

#### Training Data

We collect three *image datasets* for the PCS task: (i) SA-Co/HQ, the high-quality image data collected from the data engine in phases 1-4, (ii) SA-Co/SYN, a synthetic dataset of images labeled by a mature data engine (phase 3) without human involvement, and (iii) SA-Co/EXT, 15 external datasets that have instance mask annotations, enriched with hard negatives using our ontology pipeline. Notably in the SA-Co/HQ dataset we annotate 5.2M images and 4M unique NPs, making it the largest high-quality open-vocab segmentation dataset. We also annotate a *video dataset*, SA-Co/VIDEO, containing 52.5K videos and 24.8K unique NPs, forming 134K video-NP pairs. The videos on average have 84.1 frames at 6 fps. See §LABEL:sec:training\_data\_details for details including full statistics, comparison with existing datasets and the distribution of concepts.

#### SA-Co Benchmark

The SA-Co evaluation benchmark has 207K unique phrases, 121K images and videos, and over 3M media-phrase pairs with hard negative labels to test open-vocabulary recognition. It has 4 splits: SA-Co/Gold has seven domains and each image-NP pair is annotated by three different annotators (used to measure human performance); SA-Co/Silver has ten domains and only one human annotation per image-NP pair; SA-Co/Bronze and SA-Co/Bio are nine existing datasets either with existing mask annotations or masks generated by using boxes as prompts to SAM 2. The SA-Co/VEval benchmark has three domains and one annotator per video-NP pair. See LABEL:table:benchmark\_stats for dataset statistics and Fig.˜6 for example annotations.

#### Metrics

We aim to measure the usefulness of the model in downstream applications. Detection metrics such as average precision (AP) do not account for calibration, which means that models can be difficult to use in practice. To remedy this, we only evaluate predictions with confidence above 0.5, effectively introducing a threshold that mimics downstream usages and enforces good calibration. The PCS task can be naturally split into two sub-tasks, *localization* and *classification*. We evaluate localization using *positive micro F1* ($\mathrm{pmF}_{1}$) on positive media-phrase pairs with at least one ground-truth mask. Classification is measured with *image-level Matthews Correlation Coefficient* (IL\_MCC) which ranges in $[-1,1]$ and evaluates binary prediction at the image level (“is the object present?”) without regard for mask quality. Our main metric, *classification-gated F1* (cgF <sub>1</sub>), combines these as follows: $\mathrm{cgF\textsubscript{1}}=100*\mathrm{pmF\textsubscript{1}}*\mathrm{IL\_MCC}$. Full definitions are in §LABEL:app:metrics.

#### Handling Ambiguity

We collect 3 annotations per NP on SA-Co/Gold. We measure *oracle* accuracy comparing each prediction to all ground truths and selecting the best score. See §LABEL:app:metrics.

## 6 Experiments

We evaluate SAM 3 across image and video segmentation, few-shot adaptation to detection and counting benchmarks, and segmentation with complex language queries with SAM 3 + MLLM. We also show a subset of ablations, with more in §A. References, more results and details are in §LABEL:app:sec:experiments.

<table><tbody><tr><td></td><td colspan="6">Instance Segmentation</td><td colspan="8">Box Detection</td><td colspan="3">Semantic Segmentation</td></tr><tr><td></td><td colspan="2">LVIS</td><td colspan="4">SA-Co</td><td colspan="2">LVIS</td><td colspan="2">COCO</td><td colspan="4">SA-Co</td><td>ADE-847</td><td>PC-59</td><td>Cityscapes</td></tr><tr><td>Model</td><td>cgF <sub>1</sub></td><td>AP</td><td>Gold</td><td>Silver</td><td>Bronze</td><td>Bio</td><td>cgF <sub>1</sub></td><td>AP</td><td>AP</td><td>AP <sub>o</sub></td><td>Gold</td><td>Silver</td><td>Bronze</td><td>Bio</td><td>mIoU</td><td>mIoU</td><td>mIoU</td></tr><tr><td></td><td></td><td></td><td>cgF <sub>1</sub></td><td>cgF <sub>1</sub></td><td>cgF <sub>1</sub></td><td>pmF <sub>1</sub></td><td></td><td></td><td></td><td></td><td>cgF <sub>1</sub></td><td>cgF <sub>1</sub></td><td>cgF <sub>1</sub></td><td>pmF <sub>1</sub></td><td></td><td></td><td></td></tr><tr><td>Human</td><td>–</td><td>–</td><td>72.8</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>74.0</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>OWLv2</td><td>20.1</td><td>–</td><td>17.3</td><td>7.6</td><td>3.9</td><td>0.64</td><td>19.9</td><td>35.2</td><td>38.2</td><td>42.4</td><td>16.9</td><td>7.1</td><td>4.1</td><td>0.95</td><td>–</td><td>–</td><td>–</td></tr><tr><td>OWLv2 <sup>⋆</sup></td><td>29.3</td><td>43.4</td><td>24.6</td><td>11.5</td><td>11.7</td><td>0.04</td><td>30.2</td><td>45.5</td><td>46.1</td><td>23.9</td><td>24.5</td><td>11.0</td><td>12.0</td><td>0.08</td><td>–</td><td>–</td><td>–</td></tr><tr><td>gDino-T</td><td>14.7</td><td>–</td><td>3.3</td><td>2.7</td><td>7.0</td><td>0.34</td><td>15.1</td><td>20.5</td><td>45.7</td><td>35.3</td><td>3.4</td><td>2.5</td><td>7.6</td><td>0.35</td><td>–</td><td>–</td><td>–</td></tr><tr><td>LLMDet-L</td><td>35.1</td><td>36.3</td><td>6.5</td><td>7.1</td><td>12.5</td><td>0.15</td><td>39.3</td><td>42.0</td><td>55.6</td><td>49.8</td><td>6.8</td><td>6.7</td><td>14.0</td><td>0.17</td><td>–</td><td>–</td><td>–</td></tr><tr><td>APE-D <sup>⋆</sup></td><td>–</td><td> 53.0 <sup>†</sup></td><td>16.4</td><td>7.3</td><td>12.4</td><td>0.00</td><td>–</td><td>  59.6 <sup>†</sup></td><td>  58.3 <sup>†</sup></td><td>–</td><td>17.3</td><td>7.7</td><td>14.3</td><td>0.00</td><td> 9.2 <sup>†</sup></td><td> 58.5 <sup>†</sup></td><td> 44.2 <sup>†</sup></td></tr><tr><td>DINO-X</td><td>–</td><td> 38.5 <sup>†</sup></td><td>  21.3 <sup>δ</sup></td><td>–</td><td>–</td><td>–</td><td>–</td><td>  52.4 <sup>†</sup></td><td>  56.0 <sup>†</sup></td><td>–</td><td>  22.5 <sup>δ</sup></td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Gemini 2.5</td><td>13.4</td><td>–</td><td>13.0</td><td>8.3</td><td>7.3</td><td>10.7</td><td>16.1</td><td>–</td><td>–</td><td>–</td><td>14.4</td><td>9.4</td><td>8.2</td><td>12.4</td><td>–</td><td>–</td><td>–</td></tr><tr><td>SAM 3</td><td>37.2</td><td>48.5</td><td>54.1</td><td>49.6</td><td>42.6</td><td>55.4</td><td>40.6</td><td>53.6</td><td>56.4</td><td>55.7</td><td>55.7</td><td>50.0</td><td>47.1</td><td>56.3</td><td>13.8</td><td>60.8</td><td>65.2</td></tr></tbody></table>

Table 1: Evaluation on image concept segmentation with text. AP <sub>o</sub> corresponds to COCO-O accuracy, $\star$: partially trained on LVIS, ${\dagger}$: from original papers, $\delta$: from DINO-X API. Gray numbers indicate usage of respective closed set training data (LVIS/COCO). See §LABEL:app:image\_grounding for more baselines and results and §LABEL:app:human\_perf for details of human performance.

#### Image PCS with Text

We evaluate instance segmentation, box detection, and semantic segmentation on external and our benchmarks. SAM 3 is prompted with a single NP at a time, and predicts instance masks, bounding boxes, or semantic masks. As baselines, we evaluate OWLv2, GroundingDino (gDino), and LLMDet on box detection, and prompt SAM 1 with their boxes to evaluate segmentation. We also compare to APE, DINO-X, and Gemini 2.5 Flash, a generalist LLM. Tab. 1 shows that zero-shot, SAM 3 sets a new state-of-the-art on closed-vocabulary COCO, COCO-O and on LVIS boxes, and is significantly better on LVIS masks. On open-vocabulary SA-Co/Gold SAM 3 achieves more than double the cgF <sub>1</sub> score of the strongest baseline OWLv2 <sup>⋆</sup>, and 74% of the estimated human performance. The improvements are even higher on the other SA-Co splits. Open vocabulary semantic segmentation results on ADE-847, PascalConcept-59, and Cityscapes show that SAM 3 outperforms APE, a strong specialist baseline. See §LABEL:app:image\_grounding for details.

<table><tbody><tr><td></td><td colspan="2">ODinW13</td><td colspan="2">RF-100VL</td></tr><tr><td>Model</td><td>AP <sub>0</sub></td><td>AP <sub>10</sub></td><td>AP <sub>0</sub></td><td>AP <sub>10</sub></td></tr><tr><td>Gemini2.5-Pro</td><td>33.7</td><td>–</td><td>11.6</td><td>9.8</td></tr><tr><td>gDino-T</td><td>49.7</td><td>–</td><td>15.7</td><td>33.7</td></tr><tr><td>gDino1.5-Pro</td><td>58.7</td><td>67.9</td><td>–</td><td>–</td></tr><tr><td>SAM 3</td><td>61.0</td><td>71.8</td><td>15.2</td><td>36.5</td></tr></tbody></table>

Table 2: Zero-shot and 10-shot transfer on in-the-wild datasets.

<table><tbody><tr><td></td><td colspan="4">COCO</td><td colspan="4">LVIS</td><td colspan="4">ODinW13</td></tr><tr><td></td><td>AP</td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td><td>AP</td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td><td>AP</td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td><td><math><semantics><msup><mtext>AP</mtext> <mo>+</mo></msup> <annotation>\,\textrm{AP}^{+}</annotation></semantics></math></td></tr><tr><td>Model</td><td>T</td><td>T</td><td>I</td><td>T+I</td><td>T</td><td>T</td><td>I</td><td>T+I</td><td>T</td><td>T</td><td>I</td><td>T+I</td></tr><tr><td>T-Rex2</td><td>52.2</td><td>–</td><td>58.5</td><td>–</td><td>45.8</td><td>–</td><td>65.8</td><td>–</td><td>50.3</td><td>–</td><td>61.8</td><td>–</td></tr><tr><td>SAM 3</td><td>56.4</td><td>58.8</td><td>76.8</td><td>78.1</td><td>52.4</td><td>54.7</td><td>76.0</td><td>78.4</td><td>61.1</td><td>63.1</td><td>82.2</td><td>81.8</td></tr></tbody></table>

Table 3: Prompting with 1 exemplar on COCO, LVIS and ODinW13. Evaluation per prompt type: T (text-only), I (image-only), and T+I (combined text and image). AP <sup>+</sup> is evaluated only on positives examples.

#### Few-Shot Adaptation

We evaluate zero- and few-shot transfer of SAM 3 on ODinW13 and RF100-VL, with their original labels as prompts. We do not perform any prompt tuning. We fine-tune SAM 3 without mask loss, and report average bbox mAP in Tab. 3. SAM 3 achieves state-of-the-art 10-shot performance, surpassing in-context prompting in Gemini and object detection experts (gDino); more details in §LABEL:app:fewshots. RF-100VL contains domains with specialized prompts that are out of SAM 3’s current scope, but SAM 3 adapts through fine-tuning more efficiently than baselines.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.16719/assets/x7.png)

Figure 7: cgF 1 vs. # of interactive box prompts for SAM 3 compared to the ideal PVS baseline, averaged over SA-Co/Gold phrases.

#### PCS with 1 Exemplar

We first evaluate image exemplars using a single input box sampled at random from the ground truth. This can be done only on “positive” data, where each prompted object appears in the image. We report the corresponding $\bm{\textrm{AP}^{+}}$ in Tab.˜3 across three settings: text prompt (T), exemplar image (I), and both text and image (T+I); SAM 3 outperforms prior state-of-the-art T-Rex2 by a healthy margin on COCO (+18.3), LVIS (+10.3), and ODinW (+20.5). See §LABEL:app:visual\_exemplar for more details and results on SA-Co/Gold.

#### PCS with K Exemplars

Next, we evaluate SAM 3 in an interactive setting, simulating collaboration with a human annotator. Starting with a text prompt, we iteratively add one exemplar prompt at a time: missed ground truths are candidate positive prompts, false positive detections are candidate negative prompts. Results (Fig.˜7) are compared to a perfect PVS baseline, where we simulate the user manually fixing errors using ideal box-to-mask corrections. SAM 3’s PCS improves cgF <sub>1</sub> more quickly, as it generalizes from exemplars (e.g., detecting or suppressing similar objects), while PVS only corrects individual instances. After 3 clicks, interactive PCS outperforms text-only by +21.6 cgF <sub>1</sub> points and PVS refinement by +2.0. Performance plateaus after 4 clicks, as exemplars cannot fix poor-quality masks. Simulating a hybrid switch to PVS at this point yields gains, showing complementary.

<table><tbody><tr><td></td><td colspan="2">CountBench</td><td colspan="2">PixMo-Count</td></tr><tr><td>Model</td><td>MAE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Acc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>MAE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Acc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>DINO-X</td><td>0.62</td><td>82.9</td><td>0.21</td><td>85.0</td></tr><tr><td>Qwen2-VL-72B</td><td>0.28</td><td>86.7</td><td>0.61</td><td>63.7</td></tr><tr><td>Molmo-72B</td><td>0.27</td><td>92.4</td><td>0.17</td><td>88.8</td></tr><tr><td>Gemini 2.5 Pro</td><td>0.24</td><td>92.4</td><td>0.38</td><td>78.2</td></tr><tr><td>SAM 3</td><td>0.12</td><td>93.8</td><td>0.21</td><td>86.2</td></tr></tbody></table>

Table 3: Accuracy on counting benchmarks. Gray indicates usage of training sets.

#### Object Counting

We evaluate on object counting benchmarks CountBench and PixMo-Count to compare with several MLLMs using Accuracy (%) and Mean Absolute Error (MAE) from previous technical reports and our own evaluations. See Tab.˜3 for results and §LABEL:app:counting for more evaluation details. Compared to MLLMs, SAM 3 not only achieves good object counting accuracy, but also provides object segmentation that most MLLMs cannot provide.

#### Video PCS with Text

We evaluate video segmentation with text prompts on both our SA-Co/VEval benchmark and existing public benchmarks. For SA-Co/VEval, we report cgF <sub>1</sub> and pHOTA metrics (defined in §LABEL:app:video\_grounding\_details) across its subsets (SA-V, YT-Temporal-1B, SmartGlasses). For public benchmarks, we use their official metrics. Baselines include GLEE, an open-vocabulary image and video segmentation model, “LLMDet + SAM 3 Tracker” (replacing our detector with LLMDet), and “SAM 3 Detector + T-by-D” (replacing our tracker with an association module based on the tracking-by-detection paradigm). In Tab.˜4, SAM 3 largely outperforms these baselines, especially on benchmarks with a very large number of noun phrases. On SA-Co/VEval it reaches over 80% of human pHOTA. See §LABEL:app:video\_grounding\_details for more details.

<table><tbody><tr><td></td><td colspan="6">SA-Co/VEval benchmark test split</td><td colspan="4">Public benchmarks</td></tr><tr><td></td><td colspan="2">SA-V</td><td colspan="2">YT-Temporal-1B</td><td colspan="2">SmartGlasses</td><td>LVVIS</td><td>BURST</td><td>YTVIS21</td><td>OVIS</td></tr><tr><td></td><td colspan="2">(2.0K NPs)</td><td colspan="2">(1.7K NPs)</td><td colspan="2">(2.4K NPs)</td><td>(1.2K NPs)</td><td>(482 NPs)</td><td>(40 NPs)</td><td>(25 NPs)</td></tr><tr><td>Model</td><td>cgF <sub>1</sub></td><td>pHOTA</td><td>cgF <sub>1</sub></td><td>pHOTA</td><td>cgF <sub>1</sub></td><td>pHOTA</td><td>test mAP</td><td>test HOTA</td><td>val mAP</td><td>val mAP</td></tr><tr><td>Human</td><td>53.1</td><td>70.5</td><td>71.2</td><td>78.4</td><td>58.5</td><td>72.3</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>GLEE <sup>†</sup> (all NPs at once)</td><td>0.1</td><td>8.7</td><td>1.6</td><td>16.7</td><td>0.0</td><td>4.7</td><td>20.8</td><td>28.4</td><td>62.2</td><td>38.7</td></tr><tr><td>GLEE <sup>†</sup> (one NP at a time)</td><td>0.1</td><td>11.8</td><td>2.2</td><td>18.9</td><td>0.1</td><td>5.6</td><td>9.3</td><td>20.2</td><td>56.5</td><td>32.4</td></tr><tr><td>LLMDet <sup>†</sup> + SAM 3 Tracker</td><td>2.3</td><td>30.1</td><td>8.0</td><td>37.9</td><td>0.3</td><td>18.6</td><td>15.2</td><td>33.3</td><td>31.3</td><td>20.4</td></tr><tr><td>SAM 3 Detector + T-by-D</td><td>25.7</td><td>55.7</td><td>47.6</td><td>68.2</td><td>29.7</td><td>60.0</td><td>35.9</td><td>39.7</td><td>56.5</td><td>55.1</td></tr><tr><td>SAM 3</td><td>30.3</td><td>58.0</td><td>50.8</td><td>69.9</td><td>36.4</td><td>63.6</td><td>36.3</td><td>44.5</td><td>57.4</td><td>60.5</td></tr></tbody></table>

Table 4: Video PCS from a text prompt (open-vocabulary video instance segmentation) on SA-Co/VEval and public benchmarks (see LABEL:table:vis-results-supp for more results and analyses). SAM 3 shows strong performance, especially on benchmarks with a large number of NPs. †: GLEE and LLMDet do not perform well zero-shot on SA-Co/VEval.

#### PVS

We evaluate SAM 3 on a range of visual prompting tasks, including Video Object Segmentation (VOS) and interactive image segmentation. Tab.˜6 compares SAM 3 to recent state-of-the-art methods on the VOS task. SAM 3 achieves significant improvements over SAM 2 on most benchmarks, particularly on the challenging MOSEv2 dataset, where SAM 3 outperforms prior work by 6.5 points. For the interactive image segmentation task, we evaluate SAM 3 on the 37 datasets benchmark introduced in sam2. As shown in Tab.˜6, SAM 3 outperforms SAM 2 on average $\mathrm{mIoU}$. See also §LABEL:app:pvs\_details and LABEL:fig:supp\_sam3\_pvs for interactive video segmentation.

<table><tbody><tr><td></td><td colspan="5"><math><semantics><mrow><mi>𝒥</mi> <mo>&</mo> <mi>ℱ</mi></mrow> <annotation>\mathcal{J}\&\mathcal{F}</annotation></semantics></math></td><td><math><semantics><mi>𝒢</mi> <annotation>\mathcal{G}</annotation></semantics></math></td><td><math><semantics><mrow><mi>𝒥</mi> <mo>&</mo> <mover><mi>ℱ</mi> <mo>˙</mo></mover></mrow> <annotation>\mathcal{J}\&\dot{\mathcal{F}}</annotation></semantics></math></td></tr><tr><td></td><td>MOSEv1</td><td>DAVIS17</td><td>LVOSv2</td><td>SA-V</td><td>SA-V</td><td>YTVOS19</td><td>MOSEv2</td></tr><tr><td>Model</td><td>val</td><td>val</td><td>val</td><td>val</td><td>test</td><td>val</td><td>val</td></tr><tr><td>SAMURAI</td><td>72.6</td><td>89.9</td><td>84.2</td><td>79.8</td><td>80.0</td><td>88.3</td><td>51.1</td></tr><tr><td>SAM2Long</td><td>75.2</td><td>91.4</td><td>85.9</td><td>81.1</td><td>81.2</td><td>88.7</td><td>51.5</td></tr><tr><td>SeC</td><td>75.3</td><td>91.3</td><td>86.5</td><td>82.7</td><td>81.7</td><td>88.6</td><td>53.8</td></tr><tr><td>SAM 2.1 L</td><td>77.9</td><td>90.7</td><td>79.6</td><td>77.9</td><td>78.4</td><td>89.3</td><td>  47.9 <sup>†</sup></td></tr><tr><td>SAM 3</td><td>78.4</td><td>92.2</td><td>88.5</td><td>83.5</td><td>84.4</td><td>89.7</td><td>60.3</td></tr></tbody></table>

Table 5: SAM 3 improves over SAM 2 in VOS. †: Zero-shot.

<table><tbody><tr><td></td><td colspan="3">Avg. mIoU</td><td></td></tr><tr><td>Model</td><td>1-click</td><td>3-clicks</td><td>5-clicks</td><td>FPS</td></tr><tr><td>SAM 1 H</td><td>58.5</td><td>77.0</td><td>82.1</td><td>41.0</td></tr><tr><td>SAM 2.1 L</td><td>66.4</td><td>80.3</td><td>84.3</td><td>93.0</td></tr><tr><td>SAM 3</td><td>66.1</td><td>81.3</td><td>85.1</td><td>43.5</td></tr></tbody></table>

Table 6: Interactive image segmentation on the SA-37 benchmark.

<table><tbody><tr><td></td><td></td><td colspan="4">ReasonSeg (gIoU)</td><td colspan="4">Omnilabel (AP)</td></tr><tr><td></td><td></td><td>val</td><td colspan="3">test</td><td colspan="4">val 2023</td></tr><tr><td>Model</td><td>MLLM</td><td>All</td><td>All</td><td>Short</td><td>Long</td><td>descr</td><td>descr-S</td><td>descr-M</td><td>descr-L</td></tr><tr><td>X-SAM</td><td>Phi-3-3.8B</td><td>56.6</td><td>57.8</td><td>47.7</td><td>56.0</td><td>  12.0*</td><td>  17.1*</td><td>  11.4*</td><td>  8.8*</td></tr><tr><td>SegZero</td><td>Qwen2.5-VL 7B</td><td>62.6</td><td>57.5</td><td>–</td><td>–</td><td>  13.5*</td><td>  20.7*</td><td>  12.4*</td><td>  9.1*</td></tr><tr><td>RSVP</td><td>GPT-4o</td><td>64.7</td><td>55.4</td><td>61.9</td><td>60.3</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td colspan="2">Overall state-of-the-art <sup>†</sup></td><td>65.0</td><td>61.3</td><td>55.4</td><td>63.2</td><td>36.5</td><td>54.4</td><td>33.2</td><td>25.5</td></tr><tr><td>SAM 3 Agent</td><td>Qwen2.5-VL 7B</td><td>62.2</td><td>63.0</td><td>59.4</td><td>64.1</td><td>36.7</td><td>52.6</td><td>34.3</td><td>26.6</td></tr><tr><td>SAM 3 Agent</td><td>Llama4 Maverick</td><td>68.5</td><td>67.1</td><td>66.8</td><td>67.2</td><td>32.8</td><td>43.7</td><td>30.9</td><td>27.5</td></tr><tr><td>SAM 3 Agent</td><td>Qwen2.5-VL 72B</td><td>74.6</td><td>70.8</td><td>70.3</td><td>71.0</td><td>42.0</td><td>56.0</td><td>40.4</td><td>33.2</td></tr><tr><td>SAM 3 Agent</td><td>Gemini 2.5 Pro</td><td>77.0</td><td>74.0</td><td>75.8</td><td>73.4</td><td>45.3</td><td>53.8</td><td>45.1</td><td>37.7</td></tr></tbody></table>

Table 7: SAM 3 Agent results. Gray indicates fine-tuned results on ReasonSeg (train), \* indicates reproduced results, underline indicates the main metric. †: LISA-13B-LLaVA1.5 for ReasonSeg; REAL for OmniLabel.

#### SAM 3 Agent

We experiment with an MLLM that uses SAM 3 as a tool to segment more complex text queries (see LABEL:fig:agent\_qualitative\_good). The MLLM proposes noun phrase queries to prompt SAM 3 and analyzes the returned masks, iterating until the masks are satisfactory. Tab.˜7 shows that this “SAM 3 Agent” evaluated zero-shot on ReasonSeg and OmniLabel surpasses prior work without training on any referring expression segmentation or reasoning segmentation data. SAM 3 Agent also outperforms previous zero-shot results on RefCOCO+ and RefCOCOg. SAM 3 can be combined with various MLLMs, with the same set of the system prompts for all those MLLMs, showing SAM 3’s robustness. See §LABEL:app:agent for more details.

|  | cgF <sub>1</sub> | IL\_MCC | pmF <sub>1</sub> |
| --- | --- | --- | --- |
| $\times$ | 50.7 | 0.77 | 65.4 |
| ✓ | 52.2 | 0.82 | 63.4 |
|  |  |  |  |
|  |  |  |  |

(a)

| #/img | cgF <sub>1</sub> | IL\_MCC | pmF <sub>1</sub> |
| --- | --- | --- | --- |
| 0 | 28.3 | 0.44 | 62.4 |
| 5 | 39.4 | 0.62 | 62.9 |
| 15 | 41.8 | 0.67 | 62.4 |
| 30 | 43.0 | 0.68 | 62.8 |

(b)

| EXT | SYN | HQ | cgF <sub>1</sub> | IL\_MCC | pmF <sub>1</sub> |
| --- | --- | --- | --- | --- | --- |
| ✓ | $\times$ | $\times$ | 23.7 | 0.46 | 50.4 |
| ✓ | ✓ | $\times$ | 32.8 | 0.57 | 56.9 |
| ✓ | $\times$ | ✓ | 45.5 | 0.71 | 64.0 |
| ✓ | ✓ | ✓ | 47.4 | 0.74 | 63.8 |

(c)

| Model | cgF <sub>1</sub> | IL\_MCC | pmF <sub>1</sub> |
| --- | --- | --- | --- |
| Human | 72.8 | 0.94 | 77.0 |
| SAM 3 | 54.0 | 0.82 | 65.9 |
| \+ EV AI | 61.2 | 0.86 | 70.8 |
| \+ MV AI | 62.3 | 0.87 | 71.1 |

(d)

Table 8: Selected model and data ablations on SA-Co/Gold. Numbers across tables are not directly comparable.

#### Selected Ablations

In Tab.˜8 we report a subset of the more extensive ablations from §A. Note that the ablated models are from different, shorter training runs than the model evaluated above. The presence head boosts cgF <sub>1</sub> by +1.5 (LABEL:subtab:prescence), improving image-level recognition measured by IL\_MCC by +0.05. LABEL:subtab:hns shows that adding hard negatives significantly improves the model performance, most notably the image-level IL\_MCC from 0.44 to 0.68. LABEL:subtab:data shows that synthetic (SYN) training data improves over the external (EXT) by +8.8 cgF <sub>1</sub> and our high-quality (HQ) annotations add +14.6 cgF <sub>1</sub> on top of this baseline. We present detailed data scaling laws of both types of data in §LABEL:sec:img\_data\_ablation, showing their effectiveness on both in-domain and out-of-domain test sets. In LABEL:subtab:verify, we show how AI verifiers can improve pseudo-labels. Replacing the presence score from SAM 3 with that score from the exhaustivity verification (EV) AI verifier boosts cgF <sub>1</sub> by +7.2. Using the mask verification (MV) AI verifier to remove bad masks adds another 1.1 points. Overall, AI verifiers close half of the gap between SAM 3’s and human performance.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.16719/assets/x9.png)

Figure 9: Domain adaptation via synthetic data. Synthetic (SYN) data generated by SAM 3 + AI verifiers (teacher system) achieves similar scaling behavior as human-annotated (HQ) data.

#### Domain adaptation ablation

With domain-specific synthetic data generated by SAM 3 + AI verifiers, we show that one can significantly improve performance on a new domain without any human annotation. We hold out one of the SA-Co domains, “Food&drink”, from training SAM 3 and AI verifiers. We then use three variants of training data for the *novel* “Food&drink” domain: high-quality AI+human annotations as in SA-Co/HQ (referred to as SA-Co/HQ-Food), synthetic annotations as in SA-Co/SYN, using AI but no humans (SA-Co/SYN-Food), and pseudo-labels generated before the AI verification step, i.e. skipping both AI verifiers and humans (PL-Food). Fig. 9 plots performance on the “Food&drink” test set of the SA-Co/Gold benchmark as each type of training data is scaled up. We mix the domain specific data and high-quality general domain data at a 1:1 ratio. PL-Food provides some improvement compared to the baseline SAM 3 (zero-shot), but is far below the other variants due to its lower quality. HQ-Food and SYN-Food show similar scaling behavior, with SYN-Food slightly lower but eventually catching up, without incurring any human annotation cost. This points to a scalable way to improve performance on new data distributions. More details are in §LABEL:sec:auto\_domain\_adapt.

## 7 Related Work

Promptable and Interactive Visual Segmentation. SAM (sam) introduces “promptable” image segmentation with interactive refinement. While the original task definition included text prompts, they were not fully developed. SAM 2 (sam2) extended the promptable visual segmentation task to video, allowing refinement points on any frame. SAM 3 inherits geometry-based segmentation while extending to include text and image exemplar prompts to segment all instances of a concept in images and videos.

Open-Vocabulary Detection and Segmentation in Images exhaustively labels every instance of an open-vocabulary object category with a coarse bounding box (detection) or a fine-grained pixel mask (segmentation). Recent open-vocabulary (OV) detection (gu2021open; minderer2022simple) and segmentation (ding2022open; liang2023open) methods leverage large-scale vision-language encoders such as CLIP (radford2021learning) to handle categories described by arbitrary text, even those never seen during training. While DETR (carion2020end) is limited to a closed set of categories seen during training, MDETR (kamath2021mdetr) evolves the approach to condition on raw text queries. Image exemplars used as prompts to specify the desired object category (e.g., DINOv (Li2023VisualIP), T-Rex2 (trex2)) present a practical alternative to text, but fall short in conveying the abstract concept of objects as effectively as text prompts. We introduce a new benchmark for OV segmentation with $>100\times$ more unique concepts than prior work.

Visual Grounding localizes a language expression referring to a region of the image with a box or mask. (plummer2020revisiting) introduces phrase detection as both deciding whether the phrase is relevant to an image and localizing it. GLIP (li2022grounded) and GroundingDino (Liu2023GroundingDM) formulate object detection as phrase grounding, unifying both tasks during training. MQ-GLIP (xu2023multi) adds image exemplars to text as queries. Building on this trend toward models supporting multiple tasks and modalities, GLEE (glee) allows text phrases, referring expressions, and visual prompts for category and instance grounding in both images and videos. Unlike SAM 3, GLEE does not support exemplars or interactive refinement. LISA (lai2024lisa) allows segmentation that requires reasoning, while OMG-LLaVa (zhang2024omg) and GLaMM (rasheed2024glamm) generate natural language responses interleaved with corresponding segmentation masks, with GLaMM accepting both textual and optional image prompts as input. Some general-purpose MLLMs can output boxes and masks (Gemini2.5 (comanici2025gemini)) or points (Molmo (molmo)). SAM 3 can be used as a “vision tool” in combination with an MLLM (§6).

Multi-Object Tracking and Segmentation methods identify object instances in video and track them, associating each with a unique ID. In tracking-by-detection methods, detection is performed independently on each frame to produce boxes and confidence scores, followed by association of boxes using motion-based and appearance-based matching as in SORT (bewley2016simple; wojke2017simple), Tracktor (bergmann2019tracking), ByteTrack (zhang2022bytetrack), SAM2MOT (jiang2025sam2mot), or OC-SORT (cao2023observation). An alternative is an end-to-end trainable architecture that jointly detects and associates objects, e.g., TrackFormer (meinhardt2022trackformer), TransTrack (sun2020transtrack), or MOTR (zeng2022motr). TrackFormer uses a DETR-like encoder-decoder that initializes new tracks from static object queries and auto-regressively follows existing tracks with identity-preserving track queries. A challenge with joint models is the conflict between detection and tracking (feichtenhofer2017detect; yu2023motrv3), where one needs to focus on semantics while the other on disentangling identities, even if their spatial locations overlap over time. SAM 3 is a strong image detector tightly integrated into a tracker to segment concepts in videos.

## 8 Conclusion

We present Segment Anything with Concepts, enabling open-vocabulary text and image exemplars as prompts in interactive segmentation. Our principal contributions are: (i) introducing the PCS task and SA-Co benchmark, (ii) an architecture that decouples recognition, localization and tracking and extends SAM 2 to solve concept segmentation while retaining visual segmentation capabilities, (iii) a high-quality, efficient data engine that leverages the complimentary strengths of human and AI annotators. SAM 3 achieves state-of-the-art results, doubling performance over prior systems for PCS on SA-Co in images and videos. That said, our model has several limitations. For example, it struggles to generalize to out-of-domain terms, which could be mitigated by automatic domain expansion but requires extra training. We discuss this and other limitations of our model in §LABEL:app:sec:limitations. We believe SAM 3 and the SA-Co benchmark will be important milestones and pave the way for future research and applications in computer vision.

## 9 Acknowledgements

We would like to thank the following people for their contributions to the SAM 3 project: Alex He, Alexander Kirillov, Alyssa Newcomb, Ana Paula Kirschner Mofarrej, Andrea Madotto, Andrew Westbury, Ashley Gabriel, Azita Shokpour, Ben Samples, Bernie Huang, Carleigh Wood, Ching-Feng Yeh, Christian Puhrsch, Claudette Ward, Daniel Bolya, Daniel Li, Facundo Figueroa, Fazila Vhora, George Orlin, Hanzi Mao, Helen Klein, Hu Xu, Ida Cheng, Jake Kinney, Jiale Zhi, Jo Sampaio, Joel Schlosser, Justin Johnson, Kai Brown, Karen Bergan, Karla Martucci, Kenny Lehmann, Maddie Mintz, Mallika Malhotra, Matt Ward, Michelle Chan, Michelle Restrepo, Miranda Hartley, Muhammad Maaz, Nisha Deo, Peter Park, Phillip Thomas, Raghu Nayani, Rene Martinez Doehner, Robbie Adkins, Ross Girshik, Sasha Mitts, Shashank Jain, Spencer Whitehead, Ty Toledano, Valentin Gabeur, Vincent Cho, Vivian Lee, William Ngan, Xuehai He, Yael Yungster, Ziqi Pang, Ziyi Dou, Zoe Quake. We also thank the IDEA team for granting us DINO-X and T-Rex2 access to benchmark them on the SA-Co/Gold dataset.

## Appendix

appendix.Asubsection.A.1subsection.A.2subsection.A.3subsection.A.4subsection.A.5subsection.A.6appendix.Bappendix.Csubsection.C.1subsection.C.2subsection.C.3subsection.C.4subsubsection.C.4.1subsubsection.C.4.2appendix.Dsubsection.D.1subsection.D.2subsection.D.3subsection.D.4subsection.D.5subsection.D.6appendix.Esubsection.E.1subsection.E.2subsection.E.3subsection.E.4subsection.E.5appendix.Fsubsection.F.1subsection.F.2subsection.F.3subsection.F.4subsection.F.5subsection.F.6subsubsection.F.6.1appendix.Gsubsection.G.1subsubsection.G.1.1subsubsection.G.1.2subsubsection.G.1.3subsubsection.G.1.4subsection.G.2subsection.G.3appendix.Hsubsection.H.1subsection.H.2

## Appendix A Ablations

### A.1 Model Ablations

#### Presence Token

We first ablate the impact of the presence token and the approach to its training. The presence token is included in the decoder (discussed further in §LABEL:app:image\_implementation\_details), together with the object queries, and predicts a concept presence score. The presence score receives gradients only on the PCS task during joint training and is always supervised with the presence (or absence) of the concept in the image using a binary cross-entropy loss. Using a presence token to decouple presence and localization brings significant gains in performance, particularly on IL\_MCC, see LABEL:subtab:prescence.

When used with a presence score, we found that it is better for the box/mask object scores to not receive gradients when a concept is an image-level negative, see Setting (a) in LABEL:table:presence\_token\_training. Note that this is in contrast to the approach in typical DETR variants, where all individual object scores are supervised negatively to reflect the absence of the concept in the image, see Setting (b) in LABEL:table:presence\_token\_training. We find that (b) works worse than (a) when used with the presence score. When a concept is present in the image, individual object queries always receive classification supervision based on Hungarian matching. Setting (a) is consistent with our recognition-localization decoupled design, where the presence score is responsible for recognition (existence in the image) and the object scores are responsible for localization (i.e., rank the best match to the positive ground-truth highest among all the proposals).

During inference, we use the product of the global presence score and the object score as the total object score. In Setting (c), we explored directly supervising the total object scores (instead of the typical object scores) as positive or negative (as determined by matching); this setting can slightly improve the overall cgF <sub>1</sub>, but is less flexible as the presence and object scores are jointly calibrated, e.g. such a model is less amenable to conditioning on a concept known to be present in the image. Finally, Setting (d) in LABEL:table:presence\_token\_training investigates detaching the presence score from the computation graph while supervising the total scores, but this does not improve over (c).