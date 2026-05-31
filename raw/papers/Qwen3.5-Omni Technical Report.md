---
title: "Qwen3.5-Omni Technical Report"
source: "https://ar5iv.labs.arxiv.org/html/2604.15804"
author:
published:
created: 2026-05-30
description: "In this work, we present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. Representing a significant evolution over its predecessor, Qwen3.5-Omni scales to hundreds of billions of parameters and supp…"
tags:
  - "clippings"
---
Qwen Team

###### Abstract

In this work, we present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. Representing a significant evolution over its predecessor, Qwen3.5-Omni scales to hundreds of billions of parameters and supports a 256k context length. By leveraging a massive dataset comprising heterogeneous text-vision pairs and over 100 million hours of audio-visual content, the model demonstrates robust omni-modality capabilities. Qwen3.5-Omni-Plus achieves SOTA results across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks, surpassing Gemini-3.1 Pro in key audio tasks and matching it in comprehensive audio-visual understanding. Architecturally, Qwen3.5-Omni employs a Hybrid Attention Mixture-of-Experts (MoE) framework for both Thinker and Talker, enabling efficient long-sequence inference. The model facilitates sophisticated interaction, supporting over 10 hours of audio understanding and 400 seconds of 720P video (at 1 FPS). To address the inherent instability and unnaturalness in streaming speech synthesis—often caused by encoding efficiency discrepancies between text and speech tokenizers—we introduce ARIA (Adaptive Rate Interleave Alignment). ARIA dynamically aligns text and speech units, significantly enhancing the stability and prosody of conversational speech with minimal latency impact. Furthermore, Qwen3.5-Omni expands linguistic boundaries, supporting multilingual understanding and speech generation across 10 languages with human-like emotional nuance. Beyond preset voices, the model enables zero-shot voice customization via user-provided samples. Finally, Qwen3.5-Omni exhibits superior audio-visual grounding capabilities, generating script-level structured captions with precise temporal synchronization and automated scene segmentation. Remarkably, we observed the emergence of a new capability in omnimodal models: directly performing coding based on audio-visual instructions, which we call Audio-Visual Vibe Coding. Qwen3.5-Omni is publicly accessible via API <sup>1</sup>.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2604.15804/assets/figures/image.png)

Figure 1: Qwen3.5-Omni is a unified end-to-end model capable of processing multiple modalities, such as text, audio, image and video, and generating real-time text or speech response. Based on these features, Qwen3.5-Omni supports a wide range of tasks, including but not limited to voice dialogue, video dialogue, and audio-visual tool use.

## 1 Introduction

Human interaction with the world is inherently omnimodal and agentic, involving the integration of visual, auditory, and linguistic information, and the production of responses through text, speech, and goal-directed tool-mediated actions, facilitating information exchange with other organisms and demonstrating intelligence. Building on the rapid advances in the understanding and reasoning capabilities of large models across text (gpt3; gpt4; gemini; claude; claude2; claude3; qwen; qwen2; qwen3; llama2; llama3), vision (blip2; llava; minigpt-4; qwenvl; qwen2.5vl), and audio (qwenaudio; qwen2-audio), natively omnimodal systems that jointly process and generate across all modalities have drawn substantial attention (gpt4o; gemini2.5; qwen2.5omni; qwen3omni). However, existing models predominantly operate within passive perception-response paradigms and exhibit limited capacity for scalable agentic behavior, real-time interaction, autonomous tool utilization, and cross-modal reasoning, which are essential prerequisites for practical deployment.

In this report, we present Qwen3.5-Omni, Qwen’s latest generation of fully omnimodal LLM, supporting the understanding of text, images, audio, and audio-visual content. Natively pretrained in an omnimodal manner on massive amounts of text, visual data, and more than 100 million hours of audio-visual data, Qwen3.5-Omni is designed as a native omni agent model: it not only perceives and reasons across all modalities, but also acts, autonomously invoking WebSearch, executing complex FunctionCall, generating speech outputs, and engaging in real-time streaming interaction. The model series includes Plus and Flash variants, all of which are instruct models with 256k-token long-context input.

Qwen3.5-Omni builds on the Thinker–Talker architecture introduced in Qwen2.5-Omni (qwen2.5omni) and introduces five key technical upgrades over Qwen3-Omni (qwen3omni): (1) both the Thinker and Talker adopt Hybrid-Attention Mixture-of-Experts (MoE) designs, enabling highly efficient inference; (2) supporting long-context modeling up to 256k tokens, supporting more than 10 hours of audio and over 400 seconds of 720P audio-visual content at 1 FPS; (3) on the speech generation side, a multi-codebook codec representation enables single-frame, immediate synthesis; (4) the Talker introduces ARIA, a technique that dynamically aligns text and speech units during streaming decoding, significantly improving naturalness and robustness; and (5) multilingual training is substantially expanded, covering 113 languages and dialects for speech recognition and 36 for speech synthesis.

Enabled by these technical advances, Qwen3.5-Omni delivers three major new capabilities over Qwen3-Omni: (1) controllable audio-visual captioning, capable of generating controllable, detailed, and structured captions as well as screenplay-level fine-grained descriptions, including automatic segmentation, timestamp annotation, and detailed descriptions of characters and their relationship to audio; (2) comprehensive real-time interaction, encompassing semantic interruption through native turn-taking intent recognition, end-to-end voice control over volume, speed, and emotion, and voice cloning from user-provided samples; and (3) native omnimodal agentic behavior, including autonomous WebSearch, complex FunctionCall invocation, and Audio-Visual Vibe Coding, an emergent capability wherein the model directly generates executable code from audio-visual instructions, enabling the model to respond to real-time queries without external orchestration.

Critically, Qwen3.5-Omni maintains state-of-the-art performance on text and visual modalities without degradation relative to same-size single-model Qwen counterparts. Across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks, covering audio-visual benchmarks, audio benchmarks, ASR benchmarks, language-specific speech-to-text translation tasks, and language-specific ASR tasks, Qwen3.5-Omni-Plus achieves SOTA results, surpassing Gemini-3.1 Pro across general audio understanding, reasoning, recognition, translation, and dialogue, while its overall audio-visual understanding reaches the level of Gemini-3.1 Pro.

## 2 Architecture

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2604.15804/assets/figures/model.jpg)

Figure 2: The overview of Qwen3.5-Omni. Qwen3.5-Omni adopts the Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receives high-level representations directly from Thinker. To achieve ultra–low-latency streaming, Talker autoregressively predicts a multi-codebook sequence. At each decoding step, an MTP module outputs the residual codebooks for the current frame, after which the Code2Wav renderer incrementally synthesizes the corresponding waveform, enabling frame-by-frame streaming generation.

### 2.1 Overview

As shown in Figure 2, Qwen3.5-Omni continues to adopt the Thinker-Talker architecture (qwen2.5omni). Compared with Qwen3-Omni (qwen3omni), Qwen3.5-Omni introduces several key improvements in scalability, alignment, and real-time interaction:

- The overall backbone adopts a Hybrid Mixture-of-Experts (MoE) design, improving scalability while better balancing capacity and efficiency across multimodal understanding and generation.
- The Thinker receives visual and audio signals through the Vision Encoder and AuT, respectively. Audio and video inputs are interleaved for unified multimodal modeling, with explicit timestamps inserted to improve temporal perception, especially for long video or audio-video contexts. This design enables the Thinker to handle extended inputs, supporting up to 256k tokens, 10 hours of audio, or 400 seconds of 720P video at 1 FPS.
- The Talker is responsible for contextual speech generation by conditioning on multimodal inputs together with the textual outputs from the Thinker. Qwen3.5-Omni adopts the RVQ-based speech representation introduced in Qwen3-Omni (qwen3omni), which substantially improves inference efficiency.
- To support real-time interaction, Qwen3.5-Omni adopts both chunk-wise streaming input processing in the Thinker and a streaming Talker design, enabling low-latency end-to-end multimodal conversation.
- Different from the dual-track Talker input design in Qwen3-Omni (qwen3omni), the Talker in Qwen3.5-Omni adopts ARIA to dynamically align text and speech units before interleaving them. This design mitigates the instability caused by mismatched tokenization rates between text and speech, thereby reducing issues such as skipped words, incorrect pronunciations, and ambiguous rendering of numbers.

In the following sections, we first introduce with the AuT encoder, including its training methodology. Then, describe how Thinker processes various inputs. We then detail Talker’s multi-codebook streaming speech generation. Finally, we highlight a series of improvements on both the understanding and generation modules aimed at achieving ultra–low-latency, end-to-end streaming audio inference.

### 2.2 Audio Transformer (AuT)

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2604.15804/assets/figures/3.5aut.png)

Figure 3: The overview of AuT. Consuming 40 million hours of supervised data especially more multilingual data, AuT encoder in Qwen3.5-Omni obtain stronger general purpose audio representation in 6.25Hz.

We use transformer based audio encoder trained from scratch in attention-encoder-decoder model AuT, as is shown in Figure 3. The training of Qwen3.5-Omni encoder consumed 40 million hours of audio-text pair data generated by Qwen3-ASR. The filter bank features of the audio are downsampled 16 times using 4 Conv2D blocks and then fed into self-attention layers to obtain audio tokens in 6.25Hz token rate. Comparing to the training process of Qwen3-Omni encoder, the encoder of Qwen3.5-Omni adapts more multilingual data of more than 20 languages, and the proportion of Chinese, English and multilingual data comes to 3.5: 3.5: 3. The dynamic attention window size training mechanism is adopted for guaranting balance performance of inference under real-time prefill caching and for the offline audio understanding tasks.

### 2.3 Perceivation

##### Text, Audio, Image and Video (w/o Audio).

The Thinker converts text, audio, image, and silent video inputs into a unified sequence of representations. For text, we use the Qwen3.5 tokenizer (qwen35blog), which adopts byte-level byte-pair encoding with a vocabulary size of 250k (up from 150k), improving encoding and decoding efficiency by 10–60% across most languages. For audio inputs, including audio extracted from video, we resample the waveform to 16 kHz and convert it into a 128-channel mel-spectrogram using a 25 ms window and a 10 ms hop size. We use AuT as the audio encoder, trained from scratch on 40 million hours of audio data, where each output frame corresponds to approximately 160 ms of the original signal. For visual inputs, we adopt the vision encoder from Qwen3.5 (qwen35blog) to process both images and videos. Trained on a mixture of image and video data, this encoder provides strong capabilities in both image understanding and video comprehension. To preserve video information as much as possible while maintaining alignment with the audio stream, we sample video frames at a dynamic frame rate.

##### Audio-visual Timestamp.

Following Qwen3-Omni (qwen3omni), we apply TM-RoPE to endow the model with temporal awareness for audio-video synchronization. However, we find that directly encoding absolute time through temporal position IDs can lead to excessively sparse indices for visual patches from long video with audio inputs, which weakens long-range temporal modeling. In addition, such a design often requires large-scale and uniformly distributed training samples across different frame rates, increasing data construction cost. To address these issues, we prepend each video or audio-video temporal patch with an explicit timestamp represented as a formatted text string in seconds, allowing the model to learn timecode representations more naturally. For audio sequences, we further insert timestamps at random intervals to improve temporal alignment across modalities. Although this strategy slightly increases the context length, it enables more precise and robust temporal perception, especially when extrapolating long-context multimodal inputs.

In the context of multimodal audio-visual streams, the audio component is encoded with a temporal ID for every 160 ms. The video is treated as a sequence of frames with monotonically increasing temporal IDs that are dynamically adjusted based on their actual timestamps to ensure a consistent temporal resolution of 160 ms per ID. The height and width IDs for video frames are assigned in the same manner as for still images. To prevent positional conflicts when processing multiple modalities, the position numbering is made contiguous, with each subsequent modality commencing from one plus the maximum position ID of the preceding modality. This refined approach to positional encoding enables the model to effectively integrate and jointly model information from diverse modalities. Qwen3.5-Omni aligns these representations using their temporal IDs, which are explicitly anchored to absolute time. This design choice affords the model the flexibility to support streaming inputs of arbitrary duration.

### 2.4 Speech Generation

Talker operates directly on the RVQ tokens produced by Qwen3.5-Omni-Audio-Tokenizer. To model the residual codebooks, it employs a multi-token prediction (MTP) module, which enables fine-grained modeling and control of acoustic details. Coupled with a causal ConvNet for waveform reconstruction, Talker delivers high-fidelity speech synthesis with low inference latency and modest computational overhead.

In multi-turn spoken dialogue, Talker is conditioned on the rich contextual information provided by the Thinker component, including historical text tokens, multimodal representations, and the streamed text of the current turn. Such conditioning allows Talker to dynamically modulate acoustic attributes—such as prosody, loudness, and emotion—in accordance with the evolving conversational context.

Architecturally, our approach differs from Qwen3-Omni (qwen3omni) in two key respects. First, we introduce a dedicated system prompt for Talker that specifies target voice characteristics, thereby enabling both zero-shot voice cloning and controllable speech generation. Compared with conventional speaker embeddings, this prompt can encode richer multimodal cues, including textual descriptions and codec sequences, providing substantially finer-grained control over acoustic realization. Second, we propose ARIA (Adaptive Rate Interleave Alignment), which unifies the conventional dual-channel generation paradigm into a single-channel formulation. Rather than relying on MFA-derived alignments or fixed interleaving rates, ARIA enforces an adaptive rate constraint: for any prefix of the generated sequence, the cumulative speech-to-text token ratio must not exceed the corresponding item-level global ratio. Despite its simplicity, this design affords flexible text-speech alignment across languages, including those with relatively low encoding efficiency, and naturally supports arbitrary text-token prefixes followed by coherent speech-token continuation.

### 2.5 Designs for Streaming and Concurrency

In streaming audio-visual interaction scenarios, the first-packet latency is a critical factor affecting user experience, and the model’s concurrency capability is key to reducing service costs and improving response speed. This section discusses how Qwen3.5-Omni enhances concurrency and reduces first-packet latency through algorithmic and architectural optimizations. Table 1 provides an overview of the relevant architecture of the Qwen3.5-Omni and its associated latency.

Table 1: Architecture of Qwen3.5-Omni and end-to-end first-packet latency under audio/video settings (ms).

<table><tbody><tr><td>Module</td><td>Architecture</td><td>Streaming</td></tr><tr><td>Audio Encoder</td><td>AuT</td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td></tr><tr><td>Vision Encoder</td><td>SigLIP2</td><td>–</td></tr><tr><td>Thinker</td><td>Hybrid MoE Transformer</td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td></tr><tr><td>Talker</td><td>Hybrid MoE Transformer</td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td></tr><tr><td>MTP</td><td>Dense Transformer</td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td></tr><tr><td>Code2wav</td><td>ConvNet</td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td></tr><tr><td colspan="2">First-Packet Latency (Audio Input)</td><td>Plus: 435ms  Flash: 235ms</td></tr><tr><td colspan="2">First-Packet Latency (Video Input)</td><td>Plus: 651ms  Flash: 426ms</td></tr></tbody></table>

##### Chunked Prefilling and Hybrid MoE Architecture.

In Qwen3.5-Omni, we retain the chunked-prefilling mechanism as implemented in Qwen3-Omni and Qwen2.5-Omni, whose audio and vision encoders are capable of outputting chunks along the temporal dimension. This approach significantly reduces the Time-To-First-Token (TTFT) for both the Thinker and the Talker. Architecturally, both the Thinker and the Talker in Qwen3.5-Omni are built upon the Hybrid MoE architecture introduced in Qwen3.5. Beyond the general efficiency advantage of Hybrid MoE, this architecture includes the Gated Delta Net (GDN) module, which is particularly effective for accelerating the modeling of long audio-video sequences. As a result, it significantly reduces KV-cache I/O overhead in long-context inference, improving generation throughput and enabling higher serving concurrency.

##### Streaming Generation with ARIA.

For streaming speech generation and high-concurrency serving, Qwen3.5-Omni largely inherits the efficient design of Qwen3-Omni: Talker predicts RVQ codec tokens with a lightweight MTP module, and the generated multi-codebook tokens are converted to waveform by a causal and streaming ConvNet codec decoder. These components remain computationally lightweight, batch-friendly, and well-suited for low-latency deployment. Built on this shared foundation, the previously introduced ARIA further reformulates the dual-channel generation pattern in Qwen3-Omni into a unified interleaved single-stream formulation over text and speech tokens. By organizing text and speech generation under a monotonic interleaving constraint, ARIA reduces the synchronization overhead between separate generation tracks, enables more efficient token scheduling during decoding, and better matches the naturally incremental regime of streaming interaction.

In Table 2, we report the theoretical first-packet latency of Qwen3.5-Omni under different concurrency levels for audio and video input, evaluated on internal vLLM with torch.compile and CUDA Graph acceleration enabled for the MTP module and codec decoder. Here, Thinker TTFT (Time-To-First-Token) denotes the time from receiving the input stream to the first text token generated by Thinker, while Talker TTFC (Time-To-First-Chunk) measures the time until Talker produces the first audio chunk. TPOP (Time-Per-Output-Token) represents the per-output-token latency during steady-state decoding, where Talker TPOP includes the combined latency of the Talker backbone and the MTP module. TPS (Tokens Per Second) denotes generation throughput. Since ARIA organizes text and speech generation in a unified interleaved stream, Overall Latency cannot be obtained by simply summing several row values, but instead reflects the end-to-end critical path to the first playable audio packet. We also note that, due to the substantial scale difference between Qwen3.5-Omni-Flash and Qwen3.5-Omni-Plus, the two variants adopt different deployment-time resource allocation and parallelization strategies; therefore, their latency and throughput numbers are not intended for strict horizontal comparison. As shown in the table, Qwen3.5-Omni maintains stable latency and decoding efficiency as concurrency increases, while the low Generation RTF provides sufficient margin for smooth streaming audio generation.

Table 2: Theoretical first-packet latency of Qwen3.5-Omni under different concurrency levels. A/V denotes audio/video input.

<table><tbody><tr><td></td><td colspan="3">Qwen3.5-Omni-Flash</td><td colspan="3">Qwen3.5-Omni-Plus</td></tr><tr><td></td><td>1 Conc.</td><td>4 Conc.</td><td>8 Conc.</td><td>1 Conc.</td><td>4 Conc.</td><td>8 Conc.</td></tr><tr><td>Thinker TTFT</td><td>80/255ms</td><td>86/446ms</td><td>103/765ms</td><td>162/377ms</td><td>183/907ms</td><td>260/1243ms</td></tr><tr><td>Talker TTFC</td><td>56/61ms</td><td>68/108ms</td><td>81/116ms</td><td>54/56ms</td><td>72/88ms</td><td>95/116ms</td></tr><tr><td>Thinker TPOP</td><td>5.6/5.9ms</td><td>8.2/9.2ms</td><td>9.6/15.8ms</td><td>17.4/18.5ms</td><td>25.6/26.9ms</td><td>33.3/40.2ms</td></tr><tr><td>Talker TPOP</td><td>14.2/14.2ms</td><td>16.9/17.0ms</td><td>20.5/20.6ms</td><td>14.9/14.9ms</td><td>21.0/21.3ms</td><td>25.8/27.1ms</td></tr><tr><td>Codec Decode</td><td colspan="6">3~5ms</td></tr><tr><td>Overall Latency</td><td>235/426ms</td><td>298/891ms</td><td>352/1625ms</td><td>435/651ms</td><td>619/1515ms</td><td>955/1980ms</td></tr><tr><td>Thinker TPS</td><td>177/171</td><td>556/457</td><td>942/598</td><td>57/54</td><td>156/149</td><td>266/240</td></tr><tr><td>Talker TPS</td><td>70/70</td><td>237/235</td><td>389/388</td><td>67/67</td><td>191/189</td><td>320/296</td></tr><tr><td>Generation RTF</td><td>0.178</td><td>0.211</td><td>0.257</td><td>0.187</td><td>0.267</td><td>0.334</td></tr></tbody></table>

## 3 Pretraining

Table 3: Supported languages and dialects in Qwen3.5-Omni-Plus.

| Modality | \# Varieties | Supported languages and dialects |
| --- | --- | --- |
| Text | 201 | See Qwen3.5 for the complete list of supported languages. |
| Speech Input | 113 | 74 languages: Afrikaans, Arabic, Asturian, Azerbaijani, Basque, Belarusian, Bengali, Bosnian, Bulgarian, Cantonese, Catalan, Cebuano, Chinese, Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Interlingua, Italian, Japanese, Javanese, Kannada, Kazakh, Korean, Kyrgyz, Lingala, Latvian, Lithuanian, Macedonian, Malay, Malayalam, Maltese, Maori, Marathi, Mongolian, Norwegian Bokmål, Norwegian Nynorsk, Oriya, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tajiki, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Uyghur, and Vietnamese. 39 Chinese dialects: Northeastern Mandarin, Guizhou dialect, Guangdong Cantonese, Henan dialect, Hong Kong Cantonese, Shanghainese, Shaanxi dialect, Tianjin dialect, Taiwanese Mandarin, Yunnan dialect, Anhui dialect, Fujian dialect, Gansu dialect, Guangdong Mandarin, Hubei dialect, Hunan dialect, Jiangxi dialect, Shandong dialect, Shanxi dialect, Sichuanese, Guangxi dialect, Hainan dialect, Chongqing dialect, Changsha dialect, Hangzhou dialect, Hefei dialect, Yinchuan dialect, Zhengzhou dialect, Shenyang dialect, Wenzhou dialect, Wuhan dialect, Kunming dialect, Taiyuan dialect, Nanchang dialect, Jinan dialect, Lanzhou dialect, Nanjing dialect, Hakka, and Southern Min. |
| Speech Output | 36 | 29 languages: Chinese, English, German, Italian, Portuguese, Spanish, Japanese, Korean, French, Russian, Thai, Indonesian, Arabic, Vietnamese, Turkish, Finnish, Polish, Hindi, Dutch, Czech, Urdu, Tagalog, Swedish, Danish, Hebrew, Icelandic, Malay, Norwegian, and Persian. 7 Chinese dialects: Sichuanese, Beijing dialect, Tianjin dialect, Nanjing dialect, Shaanxi dialect, Cantonese, and Southern Min. |

Qwen3.5-Omni is pre-trained on a diverse dataset that encompasses multiple languages and dialects as shown in Table 3 and modalities, including image-text, video-text, audio-text, video-audio, video-audio-text, and pure text corpora. Following Qwen3-Omni (qwen3omni), we employ a wider range of natural language prompts to enhance both the generalization ability and instruction-following capabilities. To achieve robust performance across all modalities, our training strategy incorporates both unimodal and cross-modal data from the early pretraining stage.

In Qwen3-Omni (qwen3omni), we employ TMRoPE to endow the model with temporal awareness. However, we identify two key limitations of this approach: (1) By directly tying temporal position IDs to absolute time, it produces excessively large and sparse temporal position IDs for long audio-video or video inputs, which undermines the model’s ability to capture long-range temporal contexts. (2) Effective learning under this scheme typically requires large-scale and uniformly distributed sampling across different frame rates (fps), significantly increasing the cost of training data construction. To address these issues, we prepend each video or audio-video temporal patch with a timestamp represented as a formatted text string in seconds, enabling the model to better learn and interpret timecode representations. In addition, for audio sequences, we insert timestamps at random intervals to better align training across different modalities. Although this approach introduces a modest increase in context length, it allows the model to perceive temporal information more effectively and precisely.

The pre-training of Qwen3.5-Omni is structured into three distinct stages. In the first stage, we lock the LLM parameters and focus on training the vision and audio encoders, utilizing a vast corpus of audio-text and image-text pairs to enhance semantic understanding within the LLM. In the second stage, we unfreeze all parameters and train with a wider range of multimodal data for more comprehensive learning with a sequence length of 32,768. In the final stage, we use data with a sequence length of 262,144 to enhance the model’s ability to understand complex long-sequence data:

1. Encoder Alignment Stage (S1): During the initial pretraining phase, the LLM component of Qwen3.5-Omni is initialized with parameters from Qwen3.5, while the vision encoder is adopted from Qwen3.5, and the audio encoder is initialized with AuT. The two encoders are trained separately on the fixed LLM, with both initially focusing on training their respective adapters before training the encoders.
2. General Stage (S2): The second phase of pretraining utilizes a large-scale dataset containing approximately 4 trillion tokens, with the following distribution across modalities: text (0.92 trillion), audio (1.99 trillion), image (0.95 trillion), video (0.14 trillion), and video-audio 0.29 trillion). During this stage, the introduction of more diverse multimodal data and tasks enhances the model’s understanding and interaction capabilities in auditory, visual, textual, and audio-visual information.
3. Long Context Stage (S3): In the final pre-training phase, we increased the maximum token length from 32,768 to 262,144 and also raised the proportion of long audio and long video in the training data. Experimental results indicate that these adjustments lead to significant improvements in the model’s ability to understand long sequence data.

## 4 Post-training

### 4.1 Thinker

The post-training phase employs a three-stage strategy for the Thinker, designed to preserve the model’s capabilities across all modalities without degradation, ensure high response quality under audio queries, and optimize the overall interaction experience. The training corpus, structured in the ChatML (chatml) format, encompasses pure text, visual, audio, and mixed-modality conversational data. Specifically, the process consists of the following stages:

- Stage 1: Specialist Distillation To establish a strong foundation for omnimodal capabilities, we first train a suite of domain-specialized teacher models via independent Supervised Fine-Tuning (SFT) and reinforcement learning (RL). All teacher models are fine-tuned from the pre-trained Qwen-3.5 base checkpoint. Beyond text-related tasks, including agentic, coding, and foundational reasoning tasks, we also train specialized teacher models for vision and audio. These teacher models are used to generate domain-specific data, enabling the specialized capabilities learned in each domain to be distilled into a single unified model.
- Stage 2: On-Policy Distillation Through the specialist distillation described above, the model already achieves strong performance in domains such as multimodal understanding and reasoning, as well as text-based dialogue, reasoning, coding, and agentic tasks. Nevertheless, a substantial gap remains between the quality of responses conditioned on audio queries and that of responses conditioned on text queries, particularly in speech dialogue. To reduce this gap, we introduce a second-stage training procedure based on on-policy distillation (OPD), with the goal of distilling the model’s stronger response capabilities under text inputs into the audio-input setting. Concretely, for each audio-text paired query, we first obtain a response generated under the text condition, which typically exhibits higher quality in terms of fluency, reasoning, and task completion. We then use this response as the distillation target for the corresponding audio-conditioned query. By training on such on-policy targets, the model gradually aligns its audio-conditioned outputs with its text-conditioned behavior, thereby improving response quality under audio inputs and promoting modality-consistent generation.
- Stage 3: Interaction-Aligned Reinforcement Learning Although the previous two stages substantially improve the model’s domain capabilities and cross-modal response quality, they are not sufficient to fully optimize the model for real-world interactive use. In multi-turn conversations, we observe several interaction-specific issues, including unintended language code-switching, persona inconsistency, and degraded instruction-following over extended contexts. To mitigate these issues, we introduce Interaction-Aligned RL, a third-stage reinforcement learning procedure aimed at optimizing the model for interaction quality. We construct multi-turn interaction trajectories and design reward signals around these user experience objectives, enabling the model to learn behaviors that are more stable, consistent, and aligned in prolonged interactions. By explicitly optimizing for interaction quality, this stage improves the model’s overall usability in practical conversational scenarios.

### 4.2 Talker

We employ a four-stage training pipeline for Talker, enabling Qwen3.5-Omni to generate natural and contextually appropriate spoken responses jointly with text. All training data is organized in the ChatML format to maintain consistency with Thinker and to facilitate voice cloning.

1. General Stage: In the initial pre-training stage, we train Qwen3.5-Omni on more than 20 million hours of multilingual speech data paired with multimodal context. In particular, the introduction of more diverse tasks, such as instruction-following speech generation, substantially enhance contextual reasoning and paralinguistic alignment, going beyond a simple monotonic mapping from multimodal representations to speech.
2. Long-Context Stage: We perform data quality stratification through a dedicated curation pipeline and conduct continual pre-training (CPT) on high-quality subsets. Augmented by Qwen3-Omni-Captioner, this stage mitigates hallucinations introduced by noisy data in the initial pre-training phase and substantially improves the naturalness and quality of generated speech. Meanwhile, we extend the maximum context length to 64k tokens, allowing the model to better handle long and complex user inputs and to produce more contextually grounded speech responses.
3. Reinforcement Learning Stage: We further align model behavior with human preferences through Direct Preference Optimization (DPO) (rafailov2024direct). Concretely, we construct multilingual preference pairs based on human annotations and optimize the model with DPO. In addition, we incorporate rule-based rewards and adopt GSPO (gspo) to further improve overall capability and training stability across diverse tasks.
4. Speaker Fine-tuning Stage: Finally, we perform lightweight speaker fine-tuning on top of the base model, enabling Qwen3.5-Omni to faithfully capture target speaker characteristics while further improving the naturalness, expressiveness, and controllability of its speech outputs.

## 5 Evaluation

A comprehensive evaluation was performed on two variants of models, including Qwen3.5-Omni-Flash and Qwen3.5-Omni-Plus. The evaluation results are divided into two main categories: understanding (X $\to$ Text) and speech generation (X $\to$ Speech).

### 5.1 Evaluation of X→\\toText

In this section, we evaluate Qwen3.5-Omni’s ability to comprehend various multimodal inputs (text, audio, vision, and audio-visual video) and generate textual responses.

##### Text→\\toText

Our evaluation of Qwen3.5-Omni on text $\to$ text primarily focuses on general knowledge tasks, instruction following, long context tasks, STEM tasks, reasoning tasks and general agent ability. Specifically, we utilize MMLU-Pro (mmlupro), MMLU-Redux (mmluredux), SuperGPQA (supergpqa) and C-Eval (ceval) for general knowledge tasks, IFEval (ifeval) and IFBench (ifbench) for instruction following, AA-LCR (aalcr) and LongBench v2 (longbenchv2) for long context tasks, GPQA (gpqa) for STEM tasks, LiveCodeBench v6 (livecodebench), HMMT Nov 25 (hmmtnov25) and IMOAnswerBench (imoanswerbench) for reasoning tasks, BFCL-V4 (bfcl) and TAU2Bench (barres2025tau2) for general agent ability.

##### Audio→\\toText

To evaluate audio-to-text capabilities, we employ benchmarks across four domains: audio understanding, end-to-end speech dialogue, speech-to-text translation (S2TT), and automatic speech recognition (ASR). For audio understanding, we utilize MMAU (sakshi2024mmaumassivemultitaskaudio), MMAR (mmar), MMSU (mmsu), RUL-MuchoMusic (zang2025you), and SongFormBench (hao2025songformer) to assess comprehension of sound effects, speech, and music. Dialogue performance is evaluated via VoiceBench (chen2024voicebench), URO-Bench-pro (yan2025uro), SpeechRole (jiang2025speechrole), and WildSpeech-Bench (zhang2025wildspeechbenchbenchmarkingendtoendspeechllms). For S2TT, we focus on the translation of the top 59 languages in Fleurs (Conneau2022FLEURSFL) into English and Chinese. Finally, ASR performance is measured using Fleurs (Conneau2022FLEURSFL), Common Voice (DBLP:conf/lrec/ArdilaBDKMHMSTW20), LibriSpeech (Librispeech), WenetSpeech (DBLP:conf/icassp/ZhangLGSYXXBCZW22), KeSpeech (DBLP:conf/nips/Tang0XSLZWTXZYL21), Opencpop-test (DBLP:conf/interspeech/WangWZWLXZXB22), and MIR-1K (vocal) (DBLP:journals/taslp/HsuJ10), covering multilingual speech, Chinese dialects, and singing voice transcription.

##### Vision→\\toText

The evaluation of the model’s vision-to-text capabilities encompasses a suite of benchmarks targeting diverse and challenging tasks. To assess performance in specialized domain of mathematical and STEM reasoning, we utilize MMMU (yue2023mmmu), MMMU-Pro (mmmupro), MathVista (mathvista), MathVision (mathvision), DynaMath (dynamath), ZEROBench (zerobench). For the general visual question answering, the model is evaluated on RealWorldQA (mme-realworld), MMStar (chen2024we), HallusionBench (hallusionbench), and SimpleVQA (simplevqa). The model’s proficiency in document understanding is measured using the CharXiv (wang2024charxiv), CC-OCR (ccocr), AI2D (kembhavi2016diagram), MMLongBench-Doc (mmlongbench), and OCRBench (Liu\_2024\_OCRBench). Furthermore, the model’s spatial intelligence is specifically tested on ERQA (erqa), CountBench (countbench), RefCOCO (refcoco), ODInW13 (odinw), and EmbSpatialBench (embspatial). To evaluate performance on dynamic visual data, we report results on six video understanding benchmarks: Video-MME (fu2024video), MLVU (mlvu), MVBench (li2024mvbench), LVBench (lvbench), MMVU (mmvu) and MME-VideoOCR (videoocr). Specifically, we evaluate the model’s performance on medical VQA across three established benchmarks: SLAKE (slake), PMC-VQA (pmc), and MedXpertQA-MM (medxpertqa). This assessment is designed to demonstrate the model’s comprehensive clinical reasoning capabilities and its potential utility as a reliable healthcare AI assistant.

##### Audio-Visual Video→\\toText

We evaluate our model’s audio-visual understanding capabilities from multiple perspectives. For text-query evaluation, we use DailyOmni (dailyomni), WorldSense (worldsense), AVUT (avut), AV-SpeakerBench (avspeakerbench), and VideoMME (videomme). To assess the model’s ability in real-world audio-visual interactive scenarios, we use Qualcomm IVD (qivd) as the benchmark for audio-query-based evaluation. Beyond understanding, we also evaluate the model’s captioning capability on OmniCloze (omnicloze) and its tool-use ability on OmniGAIA (omnigaia).

#### 5.1.1 Performance of Text→\\toText

We compare Qwen3.5-Omni-Plus and Qwen3.5-Omni-Flash with Qwen3.5-Plus-Instruct. As shown in Table 4, Qwen3.5-Omni-Plus demonstrates text capabilities that are on par with its text-only counterpart across multiple dimensions, including knowledge, instruction following, long-context understanding, STEM, reasoning, and general agent tasks, highlighting its strong language ability. In particular, Qwen3.5-Omni ’s instruction-following performance is slightly better than the baseline. We believe that OPD and interaction-aligned RL have a positive effect on improving the instruction-following capabilities of an omni-model LLM.

Table 4: Text $\to$ Text performance of Qwen3.5-Omni and Qwen3.5-Plus-Instruct. The highest scores are shown in bold.

<table><tbody><tr><td>Datasets</td><td>Qwen3.5-Plus-Instruct</td><td>Qwen3.5-Omni-Flash</td><td>Qwen3.5-Omni-Plus</td></tr><tr><td colspan="4">Knowledge</td></tr><tr><td>MMLU-Pro</td><td>86.8</td><td>79.9</td><td>85.9</td></tr><tr><td>MMLU-Redux</td><td>94.3</td><td>90.0</td><td>94.2</td></tr><tr><td>SuperGPQA</td><td>67.4</td><td>54.9</td><td>66.4</td></tr><tr><td>C-Eval</td><td>92.3</td><td>86.0</td><td>92.0</td></tr><tr><td colspan="4">Instruction Following</td></tr><tr><td>IFEval</td><td>89.7</td><td>85.2</td><td>89.7</td></tr><tr><td>IFBench</td><td>51.1</td><td>38.4</td><td>52.6</td></tr><tr><td colspan="4">Long Context</td></tr><tr><td>AA-LCR</td><td>62.0</td><td>46.0</td><td>57.0</td></tr><tr><td>LongBench v2</td><td>60.2</td><td>46.4</td><td>59.6</td></tr><tr><td colspan="4">STEM</td></tr><tr><td>GPQA</td><td>85.9</td><td>76.4</td><td>83.9</td></tr><tr><td colspan="4">Reasoning</td></tr><tr><td>LiveCodeBench v6</td><td>67.1</td><td>56.6</td><td>65.6</td></tr><tr><td>HMMT Nov 25</td><td>86.2</td><td>59.0</td><td>84.4</td></tr><tr><td>IMOAnswerBench</td><td>68.3</td><td>51.5</td><td>65.5</td></tr><tr><td colspan="4">General Agent</td></tr><tr><td>BFCL-V4</td><td>66.1</td><td>55.3</td><td>63.3</td></tr><tr><td>TAU2Bench</td><td>82.7</td><td>78.0</td><td>81.0</td></tr></tbody></table>

#### 5.1.2 Performance of Audio→\\toText

Table 5: Audio benchmark comparison across Gemini-3.1 Pro, Qwen3.5-Omni-Flash, and Qwen3.5-Omni-Plus. For most benchmarks, higher is better. For ASR benchmarks, lower WER is better. Best results are shown in bold.

<table><tbody><tr><td>Datasets</td><td>Gemini-3.1 Pro</td><td>Qwen3.5-Omni-Flash</td><td>Qwen3.5-Omni-Plus</td></tr><tr><td colspan="4">Audio Understanding (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td></tr><tr><td>MMAU</td><td>81.1</td><td>80.4</td><td>82.2</td></tr><tr><td>MMAR</td><td>83.7</td><td>74.0</td><td>80.0</td></tr><tr><td>MMSU</td><td>81.3</td><td>72.2</td><td>82.8</td></tr><tr><td>RUL-MuchoMusic</td><td>59.6</td><td>60.5</td><td>72.4</td></tr><tr><td>SongFormBench-HarmonixSet <math><semantics><msub><mtext>(acc—hr.5f—hr3f)</mtext></msub> <annotation>{}_{\text{(acc|hr.5f|hr3f)}}</annotation></semantics></math> <sup>a</sup></td><td>75.6 — 46.8 — 77.9</td><td>80.6 — 67.8 — 83.4</td><td>81.1 — 72.9 — 85.3</td></tr><tr><td>SongFormBench-CN <math><semantics><msub><mtext>(acc—hr.5f—hr3f)</mtext></msub> <annotation>{}_{\text{(acc|hr.5f|hr3f)}}</annotation></semantics></math> <sup>a</sup></td><td>78.1 — 43.2 — 71.9</td><td>86.7 — 66.4 — 84.6</td><td>87.1 — 65.7 — 84.2</td></tr><tr><td colspan="4">Dialogue (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td></tr><tr><td>VoiceBench</td><td>88.9</td><td>87.8</td><td>93.1</td></tr><tr><td>URO-Bench-pro <math><semantics><msub><mtext>(U—R—O)</mtext></msub> <annotation>{}_{\text{(U|R|O)}}</annotation></semantics></math> <sup>b</sup></td><td>69.1 — 84.0 — 99.2</td><td>64.1 — 83.8 — 98.7</td><td>66.3 — 86.3 — 99.8</td></tr><tr><td>SpeechRole</td><td>124.2</td><td>119.8</td><td>123.5</td></tr><tr><td>WildSpeech-Bench</td><td>76.3</td><td>72.2</td><td>75.4</td></tr><tr><td colspan="4">S2TT (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td></tr><tr><td>Fleurs <math><semantics><msub><mrow><mtext>xx</mtext> <mo>↔</mo> <mtext>zh (top59)</mtext></mrow></msub> <annotation>{}_{\text{xx}\leftrightarrow\text{zh (top59)}}</annotation></semantics></math> <sup>c</sup></td><td>29.5</td><td>26.9</td><td>30.2</td></tr><tr><td>Fleurs <math><semantics><msub><mrow><mtext>xx</mtext> <mo>↔</mo> <mtext>en (top59)</mtext></mrow></msub> <annotation>{}_{\text{xx}\leftrightarrow\text{en (top59)}}</annotation></semantics></math> <sup>c</sup></td><td>34.6</td><td>32.0</td><td>35.4</td></tr><tr><td>Fleurs <math><semantics><msub><mrow><mtext>xx</mtext> <mo>↔</mo> <mtext>zh/en (top59)</mtext></mrow></msub> <annotation>{}_{\text{xx}\leftrightarrow\text{zh/en (top59)}}</annotation></semantics></math> <sup>c</sup></td><td>32.1</td><td>29.4</td><td>32.8</td></tr><tr><td colspan="4">ASR (WER <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math>)</td></tr><tr><td>Fleurs <math><semantics><msub><mtext>(top60)</mtext></msub> <annotation>{}_{\text{(top60)}}</annotation></semantics></math></td><td>7.32</td><td>10.75</td><td>6.55</td></tr><tr><td>CV15 <math><semantics><msub><mtext>(zh—yue—zh-tw)</mtext></msub> <annotation>{}_{\text{(zh|yue|zh-tw)}}</annotation></semantics></math></td><td>8.59 — 13.40 — 6.78</td><td>4.25 — 3.45 — 2.68</td><td>3.46 — 1.95 — 2.27</td></tr><tr><td>CV15 <math><semantics><msub><mtext>(en)</mtext></msub> <annotation>{}_{\text{(en)}}</annotation></semantics></math></td><td>8.73</td><td>5.90</td><td>4.83</td></tr><tr><td>Librispeech <math><semantics><msub><mtext>(clean—other)</mtext></msub> <annotation>{}_{\text{(clean|other)}}</annotation></semantics></math></td><td>3.36 — 4.41</td><td>1.30 — 2.43</td><td>1.11 — 2.23</td></tr><tr><td>Weneetspeech <math><semantics><msub><mtext>(net—meeting)</mtext></msub> <annotation>{}_{\text{(net|meeting)}}</annotation></semantics></math></td><td>11.53 — 14.21</td><td>4.41 — 5.51</td><td>4.30 — 5.84</td></tr><tr><td>Kespeech</td><td>23.67</td><td>4.47</td><td>3.46</td></tr><tr><td>MIR-1K <math><semantics><msub><mtext>(vocal-only)</mtext></msub> <annotation>{}_{\text{(vocal-only)}}</annotation></semantics></math> <sup>d</sup></td><td>8.76</td><td>4.94</td><td>4.56</td></tr><tr><td>Opencpop</td><td>6.83</td><td>1.11</td><td>1.49</td></tr></tbody></table>

- SongFormBench: We use a unified prompt defining an SRT-like output timestamp format and a closed vocabulary for evaluation. The vocabulary follows the SongForm-HX-8Class specified in the official codebase.
- URO-Bench-Pro: We use the pro track of URO-Bench and denote the three evaluation dimensions as follows: U for Understanding, R for Reasoning, and O for Oral Conversation. We use GenStyle-en, GenStyle-zh, Multilingual tasks for oral dimension.
- Fleurs: The top59 languages are English, Chinese, Cantonese, Korean, Japanese, Vietnamese, Thai, Malay, German, Russian, Italian, French, Spanish, Portuguese, Dutch, Indonesian, Turkish, Arabic, Polish, Hindi, Urdu, Filipino, Persian, Czech, Greek, Swedish, Hebrew, Danish, Finnish, Norwegian, Icelandic, Bengali, Punjabi, Javanese, Marathi, Swahili, Ukrainian, Gujarati, Kannada, Azerbaijani, Malayalam, Cebuano, Romanian, Hungarian, Bulgarian, Belarusian, Catalan, Tamil, Croatian, Bosnian, Slovak, Galician, Kyrgyz, Macedonian, Slovenian, Latvian, Estonian, and Asturian; compared with the top60 list, Afrikaans is excluded because the Fleurs S2TT test set does not cover this language.
- MIR-1K: Transcription is converted into Simplified Chinese.

In Table 5, we compare Qwen3.5-Omni with Gemini-3.1 Pro in terms of audio-to-text performance. Compared to Gemini-3.1 Pro, Qwen3.5-Omni exhibits superior performance on MMAU, MMSU, RUL-MuchoMusic, and SongFormBench, while achieving comparable results on MMAR, demonstrating its strong comprehension capabilities across multiple audio domains. Regarding end-to-end speech dialogue, Qwen3.5-Omni significantly outperforms Gemini-3.1 Pro on VoiceBench and matches its performance on other benchmarks, further validating Qwen3.5-Omni ’s robust capabilities in end-to-end voice interaction. For S2TT and ASR, Qwen3.5-Omni consistently outperforms Gemini-3.1 Pro, underscoring its superior translation and speech recognition performance across diverse languages, dialects, and domains.

#### 5.1.3 Performance of Vision →\\to Text

To comprehensively evaluate vision-to-text capabilities, we compare Qwen3.5-Omni-Flash and Qwen3.5-Omni-Plus with Qwen3.5-Plus-Instruct. As shown in Table 6, Qwen3.5-Omni-Plus achieves performance comparable to that of Qwen3.5-Plus-Instruct, while demonstrating stronger results on video understanding tasks involving both short and long videos. These findings highlight the strong dynamic visual perception ability of our model in real-world scenarios and suggest the effectiveness of joint video-audio training paradigms. Furthermore, we posit that audio-visual streams constitute the most naturalistic representation of real-world phenomena, wherein visual and auditory modalities are intrinsically coupled rather than independently processed.

Table 6: Vision $\to$ Text performance of Qwen3.5-Omni and Qwen3.5-Plus-Instruct. The highest scores are shown in bold.

<table><tbody><tr><td>Datasets</td><td>Qwen3.5-Plus-Instruct</td><td>Qwen3.5-Omni-Flash</td><td>Qwen3.5-Omni-Plus</td></tr><tr><td colspan="4">STEM and Puzzle</td></tr><tr><td>MMMU</td><td>81.0</td><td>76.9</td><td>80.1</td></tr><tr><td>MMMU-Pro</td><td>73.8</td><td>68.2</td><td>73.9</td></tr><tr><td>MathVision</td><td>73.6</td><td>65.4</td><td>73.0</td></tr><tr><td>Mathvista (mini)</td><td>86.9</td><td>82.9</td><td>86.1</td></tr><tr><td>DynaMath</td><td>84.2</td><td>79.3</td><td>83.8</td></tr><tr><td>ZEROBench</td><td>6</td><td>1</td><td>5</td></tr><tr><td>ZEROBench_sub</td><td>31.1</td><td>26.0</td><td>34.4</td></tr><tr><td colspan="4">General VQA</td></tr><tr><td>RealWorldQA</td><td>79.1</td><td>77.5</td><td>84.1</td></tr><tr><td>MMStar</td><td>80.3</td><td>75.7</td><td>79.4</td></tr><tr><td>MMBench <sub>EN-DEV-v1.1</sub></td><td>93.8</td><td>88.8</td><td>92.8</td></tr><tr><td>SimpleVQA</td><td>66.1</td><td>54.4</td><td>65.3</td></tr><tr><td colspan="4">Text Recognition and Document Understanding</td></tr><tr><td>CharXiv (RQ)</td><td>74.2</td><td>64.4</td><td>72.5</td></tr><tr><td>CC-OCR</td><td>83.0</td><td>80.8</td><td>83.4</td></tr><tr><td>AI2D_TEST</td><td>92.1</td><td>89.0</td><td>91.2</td></tr><tr><td>MMLongBench-Doc</td><td>59.7</td><td>53.6</td><td>57.5</td></tr><tr><td>OCRBench</td><td>91.4</td><td>89.1</td><td>91.3</td></tr><tr><td colspan="4">Spatial Intelligence</td></tr><tr><td>ERQA</td><td>53.8</td><td>50.0</td><td>54.8</td></tr><tr><td>CountBench</td><td>95.1</td><td>88.2</td><td>95.1</td></tr><tr><td>RefCOCO(avg)</td><td>95.2</td><td>92.6</td><td>95.0</td></tr><tr><td>ODInW13</td><td>50.3</td><td>46.8</td><td>49.5</td></tr><tr><td>EmbSpatialBench</td><td>83.4</td><td>82.7</td><td>85.4</td></tr><tr><td colspan="4">Video Understanding</td></tr><tr><td>VideoMME <sub>(w/o sub.)</sub></td><td>81.0</td><td>77.0</td><td>81.9</td></tr><tr><td>MLVU <sub>(M-Avg)</sub></td><td>85.1</td><td>81.9</td><td>86.8</td></tr><tr><td>MVBench</td><td>76.7</td><td>70.8</td><td>79.0</td></tr><tr><td>LVBench</td><td>68.6</td><td>65.7</td><td>71.2</td></tr><tr><td>MMVU</td><td>67.1</td><td>62.7</td><td>67.5</td></tr><tr><td>MME-VideoOCR</td><td>74.2</td><td>70.5</td><td>77.0</td></tr><tr><td colspan="4">Medical VQA</td></tr><tr><td>SLAKE</td><td>82.8</td><td>73.1</td><td>84.7</td></tr><tr><td>PMC-VQA</td><td>62.4</td><td>58.7</td><td>62.7</td></tr><tr><td>MedXpertQA-MM</td><td>55.3</td><td>44.8</td><td>54.7</td></tr></tbody></table>

#### 5.1.4 Performance of Audio-Visual Video→\\toText

We compare Qwen3.5-Omni and Gemini-3.1 Pro across a diverse range of audio-visual tasks, as shown in Table 7. For general understanding, Qwen3.5-Omni achieves state-of-the-art performance on DailyOmni and obtains comparable results on AVUT. Our model also surpasses Gemini-3.1-Pro by a substantial margin on Qualcomm IVD, demonstrating its effectiveness in real-world audio-visual interactive scenarios. Moreover, our model shows strong performance on captioning tasks. It can provide detailed audio, visual, and audio-visual captions. In this version, we also enhance the model’s tool-use capability, achieving 57.2% on OmniGAIA.

Table 7: Audio-Visual $\to$ Text performance of Qwen3.5-Omni and Gemini-3.1-Pro. The highest scores are shown in bold.

   Datasets        Gemini-3.1 Pro        Qwen3.5-Omni-Flash        Qwen3.5-Omni-Plus    Text Query QA    DailyOmni    82.7    81.8    84.6    WorldSense    65.5    57.9    62.8    AVUT    85.6    81.4    85.0    AV-SpeakerBench    75.1    65.2    71.3    VideoMME <sub>w/ audio</sub> <sup>a</sup>    89.0    79.3    83.7    Audio Query QA    Qualcomm IVD    66.2    66.3    68.5    Caption    Omni-Cloze    57.2    63.0    64.8    Agent (Tool Use)    OmniGAIA <sup>b</sup>    68.9    33.9    57.2 a VideoMME is evaluated with use\_audio\_in\_video=True. b OmniGAIA is evaluated without a thinking prompt and without <answer> formatting. All results are evaluated using DeepSeek-V3.2-Thinking as the judge.

### 5.2 Evaluation of X→\\toSpeech

In this section, we evaluate the speech generation capability of Qwen3.5-Omni. Our evaluation mainly focuses on speech generation conditioned on text and prompt speech, following a zero-shot text-to-speech (TTS) setting. We study the model from four perspectives:

- Zero-Shot Speech Generation: We evaluate content consistency, measured by WER, on SEED (seedtts).
- Multilingual Speech Generation: We evaluate both content consistency and speaker similarity in zero-shot multilingual speech generation on the TTS multilingual test set (minimaxspeech) and an internal multilingual test set built on FLEURS (Conneau2022FLEURSFL).
- Cross-Lingual Speech Generation: We evaluate content consistency in zero-shot cross-lingual speech generation on CV3-Eval (cosyvoice3).
- Custom-Voice Speech Generation: We evaluate the stability of our speaker fine-tuned model on the TTS multilingual test set (minimaxspeech) and our internal multilingual test set.

#### 5.2.1 Evaluation of Zero-Shot Speech Generation

We compare Qwen3.5-Omni with state-of-the-art zero-shot TTS systems. As shown in Table 8, Qwen3.5-Omni achieves highly competitive performance on the SEED-TTS benchmark, demonstrating strong content fidelity in zero-shot speech generation. These results reflect the effectiveness of our pretraining and continual pretraining pipeline in building robust speech generation and context modeling capabilities. Moreover, after RLHF optimization, Qwen3.5-Omni further improves generation stability and naturalness, achieving the best performance on the test-en split with a WER of 1.26.

Table 8: Zero-shot speech generation on the SEED-TTS test set. Performance is measured by Word Error Rate (WER, $\downarrow$), where lower is better. The best results are highlighted in bold.

<table><tbody><tr><td>Datasets</td><td>Model</td><td>Performance</td></tr><tr><td colspan="3">Content Consistency</td></tr><tr><td rowspan="11">SEED test-zh — test-en</td><td>Seed-TTS <sub>ICL</sub> <cite>(seedtts)</cite></td><td>1.11 — 2.24</td></tr><tr><td>Seed-TTS <sub>RL</sub> <cite>(seedtts)</cite></td><td>1.00 — 1.94</td></tr><tr><td>MaskGCT <cite>(maskgct)</cite></td><td>2.27 — 2.62</td></tr><tr><td>E2 TTS <cite>(e2tts)</cite></td><td>1.97 — 2.19</td></tr><tr><td>F5-TTS <cite>(f5tts)</cite></td><td>1.56 — 1.83</td></tr><tr><td>Spark TTS <cite>(sparktts)</cite></td><td>1.20 — 1.98</td></tr><tr><td>CosyVoice 2 <cite>(cosyvoice2)</cite></td><td>1.45 — 2.57</td></tr><tr><td>CosyVoice 3 <cite>(cosyvoice3)</cite></td><td>0.71 — 1.45</td></tr><tr><td>MiniMax-Speech <cite>(minimaxspeech)</cite></td><td>0.83 — 1.65</td></tr><tr><td>MiMo-Audio-7B-Instruct <cite>(mimoaudio)</cite></td><td>1.96 — 5.37</td></tr><tr><td>Qwen2.5-Omni-7B <cite>(qwen2.5omni)</cite></td><td>1.42 — 2.33</td></tr><tr><td></td><td>Qwen3-Omni-30B-A3B <cite>(qwen3omni)</cite></td><td>1.07 — 1.39</td></tr><tr><td></td><td>Qwen3.5-Omni-Plus</td><td>0.99 — 1.26</td></tr></tbody></table>

#### 5.2.2 Evaluation of Multilingual Speech Generation

Qwen3.5-Omni supports speech generation in 29 languages. We compare its multilingual speech generation performance with two strong commercial systems, MiniMax-Speech and ElevenLabs. For the internal multilingual test set, we use GPT-4o-transcribe-2025-03-20 for automatic speech recognition.

As shown in Table 9 and Table 10, Qwen3.5-Omni achieves the lowest WER in 22 out of 29 evaluated languages on the multilingual test sets, outperforming the comparison systems by a clear margin in most cases. On the remaining languages, Qwen3.5-Omni remains competitive with state-of-the-art systems. In addition to content consistency, Qwen3.5-Omni also shows strong voice cloning fidelity. It obtains the highest speaker similarity scores in the majority of evaluated languages and consistently outperforms both MiniMax-Speech and ElevenLabs overall. These results suggest that Qwen3.5-Omni effectively preserves speaker characteristics, such as timbre and prosodic style, while maintaining robust multilingual speech generation quality.

Furthermore, in Table 10, we report results on our internal multilingual test set, covering an additional 9 languages. Qwen3.5-Omni continues to achieve strong performance across all evaluated languages, indicating that its multilingual speech generation ability generalizes well beyond the public benchmark languages.

Table 9: Multilingual speech generation on the TTS multilingual test set. Performance is measured by Word Error Rate (WER, $\downarrow$) for content consistency and cosine similarity (SIM, $\uparrow$) for speaker similarity. The best results are highlighted in bold.

<table><tbody><tr><td rowspan="2">Language</td><td colspan="3">Content Consistency</td><td colspan="3">Speaker Similarity</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>MiniMax</td><td>ElevenLabs</td><td>Qwen3.5-Omni-Plus</td><td>MiniMax</td><td>ElevenLabs</td></tr><tr><td>Chinese</td><td>0.695</td><td>2.252</td><td>16.026</td><td>0.800</td><td>0.780</td><td>0.677</td></tr><tr><td>English</td><td>0.631</td><td>2.164</td><td>0.756</td><td>0.833</td><td>0.756</td><td>0.613</td></tr><tr><td>German</td><td>0.447</td><td>1.906</td><td>0.572</td><td>0.757</td><td>0.733</td><td>0.614</td></tr><tr><td>Italian</td><td>0.503</td><td>1.543</td><td>1.743</td><td>0.785</td><td>0.699</td><td>0.679</td></tr><tr><td>Portuguese</td><td>1.221</td><td>1.877</td><td>1.331</td><td>0.792</td><td>0.805</td><td>0.711</td></tr><tr><td>Spanish</td><td>0.862</td><td>1.029</td><td>1.084</td><td>0.797</td><td>0.762</td><td>0.615</td></tr><tr><td>Japanese</td><td>3.479</td><td>3.519</td><td>10.046</td><td>0.788</td><td>0.776</td><td>0.738</td></tr><tr><td>Korean</td><td>1.458</td><td>1.747</td><td>1.865</td><td>0.747</td><td>0.776</td><td>0.700</td></tr><tr><td>French</td><td>2.430</td><td>4.099</td><td>5.216</td><td>0.730</td><td>0.628</td><td>0.535</td></tr><tr><td>Russian</td><td>3.182</td><td>4.281</td><td>3.878</td><td>0.790</td><td>0.761</td><td>0.676</td></tr><tr><td>Thai</td><td>2.170</td><td>2.701</td><td>73.936</td><td>0.788</td><td>0.800</td><td>0.588</td></tr><tr><td>Indonesian</td><td>0.823</td><td>1.237</td><td>1.059</td><td>0.780</td><td>0.729</td><td>0.660</td></tr><tr><td>Arabic</td><td>2.602</td><td>1.665</td><td>1.666</td><td>0.745</td><td>0.736</td><td>0.706</td></tr><tr><td>Vietnamese</td><td>1.143</td><td>0.880</td><td>73.415</td><td>0.767</td><td>0.743</td><td>0.369</td></tr><tr><td>Turkish</td><td>0.938</td><td>1.520</td><td>0.699</td><td>0.747</td><td>0.779</td><td>0.596</td></tr><tr><td>Finnish</td><td>2.784</td><td>4.666</td><td>2.964</td><td>0.859</td><td>0.835</td><td>0.759</td></tr><tr><td>Polish</td><td>1.427</td><td>1.415</td><td>0.766</td><td>0.839</td><td>0.802</td><td>0.729</td></tr><tr><td>Hindi</td><td>6.444</td><td>6.962</td><td>5.827</td><td>0.797</td><td>0.818</td><td>0.730</td></tr><tr><td>Dutch</td><td>1.238</td><td>1.143</td><td>0.803</td><td>0.762</td><td>0.738</td><td>0.680</td></tr><tr><td>Czech</td><td>2.929</td><td>3.875</td><td>2.108</td><td>0.802</td><td>0.796</td><td>0.685</td></tr></tbody></table>

Table 10: Multilingual speech generation on the internal multilingual test set. Performance is measured by Word Error Rate (WER, $\downarrow$) for content consistency and cosine similarity (SIM, $\uparrow$) for speaker similarity. The best results are highlighted in bold.

<table><tbody><tr><td rowspan="2">Language</td><td colspan="2">Content Consistency</td><td colspan="2">Speaker Similarity</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>Ground Truth</td><td>Qwen3.5-Omni-Plus</td><td>Ground Truth</td></tr><tr><td>Urdu</td><td>14.819</td><td>17.822</td><td>0.775</td><td>-</td></tr><tr><td>Tagalog</td><td>5.193</td><td>6.885</td><td>0.870</td><td>-</td></tr><tr><td>Swedish</td><td>3.760</td><td>4.813</td><td>0.822</td><td>-</td></tr><tr><td>Danish</td><td>3.636</td><td>6.403</td><td>0.775</td><td>-</td></tr><tr><td>Hebrew</td><td>7.860</td><td>16.178</td><td>0.760</td><td>-</td></tr><tr><td>Icelandic</td><td>10.244</td><td>11.451</td><td>0.764</td><td>-</td></tr><tr><td>Malay</td><td>3.142</td><td>4.628</td><td>0.794</td><td>-</td></tr><tr><td>Norwegian</td><td>3.613</td><td>4.442</td><td>0.825</td><td>-</td></tr><tr><td>Persian</td><td>11.113</td><td>14.469</td><td>0.800</td><td>-</td></tr></tbody></table>

#### 5.2.3 Evaluation of Cross-Lingual Speech Generation

Beyond multilingual voice cloning, Qwen3.5-Omni also supports cross-lingual voice cloning, where the model is required to preserve speaker identity while generating speech in a different target language. We evaluate this capability on the Cross-Lingual benchmark and compare against the CosyVoice series as well as Qwen3-Omni-30B-A3B.

In Table 11, we report the mixed error rate (WER for English and CER for the other languages) across different source–target language pairs. Overall, Qwen3.5-Omni achieves the best performance in 10 out of 12 evaluated directions and sets a new state of the art on most English-, Japanese-, and Korean-targeted pairs. In particular, for zh-to-ko, Qwen3.5-Omni reduces the error rate from 14.4 to 4.03 compared with CosyVoice3, corresponding to an approximately 72% relative reduction. Qwen3.5-Omni also performs strongly on commonly used language pairs such as zh-to-en and en-to-zh, indicating better content consistency under cross-lingual generation. These results demonstrate that Qwen3.5-Omni generalizes effectively across language boundaries while preserving target linguistic accuracy.

Table 11: Cross-lingual speech generation on the Cross-Lingual benchmark. Performance is measured by mixed error rate (WER for English and CER for the other languages, $\downarrow$). The best results are highlighted in bold.

| Language | Qwen3.5-Omni-Plus | Qwen3-Omni-30B-A3B | CosyVoice3 | CosyVoice2 |
| --- | --- | --- | --- | --- |
| English-to-Chinese | 4.86 | 5.37 | 5.09 | 13.5 |
| Japanese-to-Chinese | 3.55 | 3.32 | 3.05 | 48.1 |
| Korean-to-Chinese | 0.84 | 0.99 | 1.06 | 7.70 |
| Chinese-to-English | 2.18 | 2.76 | 2.98 | 6.47 |
| Japanese-to-English | 2.18 | 3.31 | 4.20 | 17.1 |
| Korean-to-English | 2.51 | 3.34 | 4.19 | 11.2 |
| Chinese-to-Japanese | 5.92 | 8.29 | 7.08 | 13.1 |
| English-to-Japanese | 5.12 | 7.53 | 6.80 | 14.9 |
| Korean-to-Japanese | 2.16 | 4.24 | 3.93 | 5.86 |
| Chinese-to-Korean | 4.03 | 5.13 | 14.4 | 24.8 |
| English-to-Korean | 3.72 | 4.96 | 5.87 | 21.9 |
| Japanese-to-Korean | 5.12 | 6.23 | 7.92 | 21.5 |

#### 5.2.4 Evaluation of Custom-Voice Speech Generation

We evaluate the custom-voice speech generation capability of Qwen3.5-Omni in multilingual settings. We compare Qwen3.5-Omni with several strong commercial systems accessed through their official APIs in March 2026, including ElevenLabs Multilingual v2 (9YHcvj6GT2YYXdXww), Gemini-2.5 Pro-Preview-TTS (Achernar), GPT-Audio-2025-08-28 (Alloy), and MiniMax-Speech-2.8-HD (English\_expressive\_narrator).

Table 12: Custom-voice multilingual speech generation on the multilingual test set. Performance is measured by Word Error Rate (WER, $\downarrow$). The best results are highlighted in bold.

| Language | Qwen3.5-Omni-Plus | ElevenLabs | Gemini-2.5 Pro | GPT-Audio | MiniMax |
| --- | --- | --- | --- | --- | --- |
| Chinese | 0.785 | 3.801 | 1.890 | 0.829 | 0.786 |
| English | 0.839 | 1.126 | 0.953 | 1.050 | 1.429 |
| German | 0.182 | 0.500 | 0.509 | 0.558 | 1.581 |
| Italian | 0.458 | 0.513 | 0.991 | 0.769 | 1.063 |
| Portuguese | 1.581 | 1.109 | 2.050 | 1.506 | 1.240 |
| Spanish | 0.768 | 0.520 | 0.891 | 0.936 | 0.691 |
| Japanese | 3.306 | 11.685 | 4.420 | 4.317 | 4.254 |
| Korean | 1.309 | 3.981 | 4.110 | 3.999 | 3.635 |
| French | 2.724 | 2.574 | 3.284 | 2.809 | 3.439 |
| Russian | 4.723 | 4.324 | 3.858 | 4.346 | 3.529 |
| Thai | 1.653 | 114.813 | 2.539 | 4.430 | 1.811 |
| Indonesian | 1.596 | 6.094 | 1.498 | 2.362 | 1.585 |
| Arabic | 3.183 | 5.400 | 5.525 | 5.326 | 3.309 |
| Vietnamese | 1.320 | 82.849 | 1.699 | 1.854 | 1.058 |
| Turkish | 1.309 | 0.551 | 2.237 | 1.389 | 0.652 |
| Finnish | 4.039 | 2.522 | 5.331 | 3.270 | 2.939 |
| Polish | 1.462 | 0.733 | 1.622 | 1.737 | 0.833 |
| Hindi | 6.776 | 6.388 | 6.596 | 7.191 | 6.146 |
| Dutch | 1.135 | 1.005 | 0.973 | 1.561 | 1.406 |
| Czech | 3.769 | 1.916 | 3.380 | 2.859 | 1.766 |
| Urdu | 14.916 | 12.970 | 14.141 | 13.362 | 24.151 |
| Tagalog | 5.090 | 5.473 | 6.784 | 5.352 | 5.674 |
| Swedish | 3.588 | 3.132 | 3.196 | 2.898 | 2.833 |
| Danish | 7.183 | 2.604 | 3.876 | 3.846 | 4.951 |
| Hebrew | 7.680 | 102.018 | 4.459 | 5.328 | 8.161 |
| Icelandic | 10.322 | 25.110 | 6.348 | 9.648 | 33.431 |
| Malay | 3.738 | 6.448 | 3.731 | 3.406 | 3.955 |
| Norwegian | 5.576 | 7.351 | 4.304 | 3.400 | 9.492 |
| Persian | 12.140 | 20.564 | 12.620 | 13.202 | 12.722 |

As shown in Table 12, although Qwen3.5-Omni is fine-tuned only on monolingual data, it demonstrates strong cross-lingual generalization in custom-voice speech generation. The model is able to transfer the target speaker characteristics to all 29 evaluated languages while maintaining stable generation quality. Overall, Qwen3.5-Omni achieves the best WER in 10 languages and remains competitive in many others. In particular, it shows clear advantages in several challenging languages, including Japanese (3.306) and Korean (1.309), indicating strong intelligibility under cross-lingual voice transfer. These results suggest that Qwen3.5-Omni can generate custom-voice speech with robust linguistic fidelity across a wide range of languages.

## 6 Conclusion

In this work, we present Qwen3.5-Omni, a fully omnimodal large language model that unifies understanding, reasoning, generation, and action across text, images, audio, and audio-visual inputs. Built on the Thinker–Talker framework, Qwen3.5-Omni introduces efficient Hybrid-Attention MoE architectures, 256k long-context modeling, improved streaming speech generation with multi-codebook codec prediction and ARIA, and substantially expanded multilingual speech support. These advances enable three key capabilities: controllable audio-visual captioning, comprehensive real-time interaction, and native omnimodal agentic behavior through autonomous tool use and audio-visual code generation. Empirically, Qwen3.5-Omni achieves state-of-the-art or highly competitive performance across a broad range of audio and audio-visual benchmarks, while maintaining the strong text and vision capabilities of same-scale Qwen models. These results suggest that scaling native omnimodal training can produce unified systems that not only perceive and reason across modalities, but also interact and act in real time. We hope Qwen3.5-Omni provides a strong foundation for future research on general-purpose omnimodal agents.

## 7 Authors

Core Contributors <sup>2</sup>

Bing Han  
Baosong Yang  
Bin Zhang  
Bo Zheng  
Dayiheng Liu  
Fan Zhou  
Hongkun Hao  
Hangrui Hu  
Jin Xu <sup>∗</sup>  
Jianxin Yang  
Jingren Zhou  
Keqin Chen  
Le Yu  
Mingkun Yang  
Peng Wang  
Pei Zhang  
Qize Yang  
Rui Men  
Ruiyang Xu  
Shuai Bai  
Sibo Song  
Ting He  
Xize Cheng  
Xuejing Liu  
Xingzhang Ren  
Xian Shi  
Xiong Wang  
Xinyu Zhang  
Xinfa Zhu  
Yunfei Chu  
Yuanjun Lv  
Yuchong Sun  
Yongqi Wang  
Yuxuan Wang  
Yang Zhang  
Zhifang Guo  
Zishan Guo  
Ziyang Ma

Contributors <sup>†</sup>

Andong Chen  
Anfeng Li  
An Yang  
Bei Chen  
Bin Lin  
Bingshen Mu  
Bohan Wang  
Buxiao Wu  
Bowen Xu  
Beichen Zhang  
Cheng Chen  
Chang Gao  
Chengen Huang  
Chenyang Le  
Chenhao Li  
Chenglong Liu  
Chenxu Lv  
Chen Qiang  
Chenfei Wu  
Chenhan Yuan  
Chengruidong Zhang  
Chujie Zheng  
Daren Chen  
Dake Guo  
Fei Huang  
Gaoji Liu  
Guangdong Zhou  
Hao Ge  
Huiqiang Jiang  
Haoran Lian  
Hongjian Tu  
Hao Yu  
Hang Zhang  
Hao Zhou  
Haiquan Zhao  
Humen Zhong  
Jiawei Chen  
Jian Guan  
Jiayi Leng  
Jiahao Li  
Junrong Lin  
Jiawei Liu  
Jialong Tang  
Jun Tang  
Jianhong Tu  
Jianqiang Wan  
Jinxi Wei  
Jianwei Zhang  
Jing Zhou  
Kai Dang  
Kangxiang Xia  
Kun Yan  
Kexin Yang  
Lianghao Deng  
Lulu Hu  
Linhan Ma  
Lingchen Meng  
Lei Xie  
Laiwen Zheng  
Miao Hong  
Mei Li  
Mingcheng Li  
Mingze Li  
Minsheng Li  
Minghao Wu  
Mingfeng Xue  
Na Ni  
Peng Liu  
Peng Wang  
Pengfei Wang  
Peiyang Zhang  
Qidong Huang  
Qingfeng Lan  
Qintong Li  
Que Shen  
Qiuyue Wang  
Qin Zhu  
Ruisheng Cao  
Rongyao Fang  
Rui Hu  
Ruibin Yuan  
Song Chen  
Su Hao  
Shen Li  
Shixuan Liu  
Shurui Li  
Siqi Zhang  
Tianyi Tang  
Tingyu Xia  
Wei Ding  
Wenbin Ge  
Weizhou Shen  
Wei Wang  
Wentao Yao  
Xi Chen  
Xiaotong Chen  
Xionghui Chen  
Xiaodong Deng  
Xudong Guo  
Xin Le  
Xiao Li  
Xie Chen  
Xinyao Niu  
Xuancheng Ren  
Xuechun Wang  
Xuwu Wang  
Xingzhe Wu  
Xipin Wei  
Xiao Xu  
Xian Yang  
Yuxuan Cai  
Yizhong Cao  
Yilei Chen  
Yuxiang Chen  
Yiming Dong  
Yang Fan  
Yanpeng Li  
Yucheng Li  
Yang Liu  
Yantao Liu  
Yuqiong Liu  
Yuxuan Liu  
Yuyan Luo  
Yubo Ma  
Yang Su  
Yuezhang Wang  
Yuhao Wang  
Yi Wu  
Yunbao Wu  
Yu Xi  
Yi Zhang  
Yichang Zhang  
Yinger Zhang  
Yuxiang Zheng  
Zeyu Cui  
Ziwei Ji  
Ziyue Jiang  
Zhaohai Li  
Zheng Li  
Zhi Li  
Zihan Qiu  
Zekun Wang  
Zhihai Wang  
Zhenghao Xing  
Zhibo Yang  
Zhuorui Ye  
Zhenru Zhang  
Zhipeng Zhou  
Zhengyang Zhuge

## 8 Appendix

### 8.1 Detailed Multilingual Evaluation Results

##### Multilingual ASR.

As presented in Table 13, Qwen3.5-Omni demonstrates superior speech recognition capabilities compared to state-of-the-art competitors on the FLEURS test set. Qwen3.5-Omni-Plus achieves the lowest average WER of 6.6%, outperforming both Gemini-3.1-Pro (7.3%) and GPT-4o-Transcribe (10.4%). It secures the best performance in the majority of languages, with particularly significant margins in complex tonal and low-resource languages such as Cantonese (2.2% vs. 6.3% for Gemini-3.1-Pro), Thai, and Vietnamese. Meanwhile, Qwen3.5-Omni-Flash offers a highly efficient alternative, achieving an average WER of 10.8% that remains competitive against Gemini-3-Flash (10.5%). Notably, Qwen3.5-Omni-Flash exhibits exceptional robustness in challenging scenarios, drastically reducing errors in Cantonese (3.1% vs. 10.8% for Gemini-3-Flash) and maintaining strong performance in Japanese and Korean, thereby highlighting its advantage for high-value Asian language pairs.

##### Multilingual Translation.

As shown in Tables 14 and 15, the Qwen3.5-Omni series demonstrates distinct advantages over state-of-the-art competitors on the FLEURS test set, particularly in Asian languages and specific high-resource pairs. Qwen3.5-Omni-Plus exhibits comprehensive superiority over Gemini-3.1-Pro in the many-to-many directions (en2xx/zh2xx), achieving higher average BLEU scores in both English-to-XX (33.8 vs. 31.8) and Chinese-to-XX (21.4 vs. 19.6). It also leads in key xx2en pairs such as Portuguese (49.4 vs. 47.7) and Indonesian (45.7 vs. 45.1). Although Gemini-3.1-Pro holds a slight edge in overall xx2zh averages, Qwen3.5-Omni-Plus significantly outperforms it in critical Asian languages, including Cantonese (+15.6 BLEU), Korean, and Japanese. Similarly, Qwen3.5-Omni-Flash shows targeted strengths against Gemini-3-Flash. While maintaining competitive general performance, it vastly surpasses Gemini in Cantonese translation across all directions (e.g., 37.5 vs. 22.4 in xx2zh and 37.3 vs. 26.7 in en2xx) and delivers better results in Japanese and Korean xx2zh tasks. These results underscore Qwen3.5-Omni’s robust optimization for complex Asian linguistic structures and key regional languages.

Table 13: Multilingual ASR performance on the FLEURS test set. Results are reported using Word Error Rate (WER, ↓), where lower values indicate better performance. For italicized languages, Character Error Rate (CER, ↓) is reported instead. Compared with competing models, Qwen3.5-Omni-Plus achieves the best results on the majority of languages. The best results are highlighted in bold.

| Language | Qwen3.5-Omni-Plus | Qwen3.5-Omni-Flash | Gemin-3.1-Pro | GPT-4o-Transcribe | Gemini-3-Flash |
| --- | --- | --- | --- | --- | --- |
| Chinese | 2.9 | 2.9 | 3.6 | 2.6 | 4.6 |
| English | 3.2 | 3.7 | 2.7 | 3.2 | 2.9 |
| Cantonese | 2.2 | 3.1 | 6.3 | 5.2 | 10.8 |
| Arabic | 11.7 | 13.6 | 9.2 | 13.0 | 10.1 |
| German | 2.0 | 2.5 | 2.7 | 2.3 | 3.3 |
| French | 2.6 | 3.3 | 3.7 | 3.7 | 3.9 |
| Spanish | 2.2 | 2.4 | 2.5 | 2.3 | 2.7 |
| Portuguese | 2.1 | 2.2 | 2.6 | 2.3 | 2.9 |
| Indonesian | 1.6 | 2.4 | 2.5 | 3.5 | 2.8 |
| Italian | 0.8 | 1.0 | 1.1 | 1.4 | 1.8 |
| Korean | 1.7 | 2.1 | 2.0 | 2.1 | 2.4 |
| Russian | 3.1 | 3.6 | 3.4 | 3.7 | 3.9 |
| Thai | 2.8 | 3.2 | 4.3 | 4.9 | 4.5 |
| Vietnamese | 1.9 | 2.5 | 2.5 | 3.5 | 3.5 |
| Japanese | 1.9 | 2.5 | 2.3 | 3.0 | 3.4 |
| Turkish | 3.1 | 4.4 | 3.8 | 4.2 | 4.4 |
| Hindi | 9.7 | 9.9 | 4.5 | 12.0 | 5.6 |
| Malay | 2.7 | 4.2 | 3.7 | 4.1 | 6.2 |
| Dutch | 2.8 | 3.5 | 3.5 | 3.7 | 4.7 |
| Urdu | 20.8 | 31.9 | 25.2 | 19.7 | 23.0 |
| Norwegian | 3.9 | 5.2 | 5.0 | 5.5 | 6.7 |
| Swedish | 3.1 | 5.0 | 4.6 | 5.2 | 7.7 |
| Danish | 3.5 | 5.3 | 5.7 | 6.5 | 7.9 |
| Hebrew | 12.5 | 16.6 | 15.6 | 19.4 | 20.2 |
| Finnish | 2.4 | 4.5 | 3.4 | 3.8 | 5.2 |
| Polish | 1.9 | 3.1 | 2.7 | 2.8 | 4.9 |
| Icelandic | 3.6 | 8.9 | 4.7 | 10.8 | 6.8 |
| Czech | 2.6 | 4.5 | 3.8 | 4.7 | 8.0 |
| Filipino | 5.1 | 7.1 | 7.6 | 7.3 | 8.5 |
| Persian | 12.0 | 12.1 | 8.9 | 9.9 | 10.0 |
| Greek | 4.7 | 8.1 | 5.4 | 6.5 | 7.6 |
| Afrikaans | 10.6 | 13.7 | 12.7 | 17.9 | 18.6 |
| Asturian | 15.8 | 25.9 | 23.7 | 23.8 | 48.3 |
| Belarusian | 6.7 | 12.2 | 6.7 | 10.2 | 10.9 |
| Bulgarian | 6.2 | 10.7 | 5.3 | 7.0 | 7.9 |
| Bengali | 16.2 | 19.8 | 21.9 | 24.1 | 21.9 |
| Bosnian | 5.4 | 9.5 | 6.0 | 13.9 | 11.2 |
| Catalan | 2.8 | 6.3 | 2.7 | 2.7 | 6.4 |
| Cebuano | 10.5 | 16.6 | 13.0 | 15.1 | 12.8 |
| Estonian | 6.7 | 16.6 | 4.9 | 7.6 | 7.3 |
| Galician | 5.0 | 8.6 | 4.9 | 6.8 | 14.6 |
| Gujarati | 13.9 | 18.4 | 14.8 | 26.9 | 16.3 |
| Croatian | 5.4 | 9.0 | 5.1 | 16.4 | 9.3 |
| Hungarian | 4.9 | 10.6 | 5.5 | 7.4 | 11.3 |
| Javanese | 11.8 | 18.3 | 14.1 | 24.7 | 15.7 |
| Kazakh | 6.3 | 16.6 | 6.2 | 11.5 | 12.7 |
| Kannada | 16.0 | 23.8 | 16.3 | 28.1 | 16.3 |
| Kyrgyz | 10.0 | 19.7 | 8.3 | 20.7 | 16.3 |
| Latvian | 6.7 | 17.8 | 3.7 | 6.3 | 6.8 |
| Macedonian | 4.1 | 7.9 | 4.0 | 6.2 | 7.8 |
| Malayalam | 18.8 | 27.0 | 18.3 | 33.5 | 20.3 |
| Marathi | 16.3 | 23.6 | 15.3 | 26.3 | 16.0 |
| Punjabi | 13.7 | 24.6 | 14.4 | 36.4 | 17.4 |
| Romanian | 3.2 | 6.1 | 3.4 | 4.5 | 5.9 |
| Slovak | 3.3 | 5.5 | 2.8 | 3.6 | 6.5 |
| Slovenian | 6.1 | 14.3 | 6.3 | 8.8 | 10.0 |
| Swahili | 9.4 | 17.5 | 9.9 | 16.3 | 10.7 |
| Tajik | 10.0 | 41.1 | 20.4 | 20.2 | 53.1 |
| Azerbaijani | 7.2 | 13.0 | 5.8 | 10.6 | 13.2 |
| Ukrainian | 3.2 | 5.4 | 3.4 | 4.3 | 5.1 |
| Average | 6.6 | 10.8 | 7.3 | 10.4 | 10.5 |

Table 14: Multilingual translation performance on the FLEURS en2xx and zh2xx test sets. Results are reported using BLEU (↑). Compared with competing models, Qwen3.5-Omni-Plus outperforms them on the majority of language pairs. The best results are highlighted in bold.

<table><tbody><tr><td></td><td colspan="4">en2xx (English → Other Languages)</td><td colspan="4">zh2xx (Chinese → Other Languages)</td></tr><tr><td>Language</td><td>Qwen3.5-Omni-Plus</td><td>Qwen3.5-Omni-Flash</td><td>Gemin-3.1-Pro</td><td>Gemini-3-Flash</td><td>Qwen3.5-Omni-Plus</td><td>Qwen3.5-Omni-Flash</td><td>Gemin-3.1-Pro</td><td>Gemini-3-Flash</td></tr><tr><td>Chinese</td><td>47.8</td><td>46.6</td><td>47.4</td><td>46.3</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>English</td><td>–</td><td>–</td><td>–</td><td>–</td><td>32.2</td><td>31.2</td><td>30.1</td><td>29.5</td></tr><tr><td>Cantonese</td><td>40.1</td><td>37.3</td><td>25.5</td><td>26.7</td><td>36.7</td><td>35.9</td><td>23.7</td><td>24.0</td></tr><tr><td>Arabic</td><td>31.1</td><td>28.2</td><td>27.0</td><td>28.5</td><td>16.1</td><td>13.9</td><td>14.2</td><td>14.4</td></tr><tr><td>German</td><td>43.2</td><td>39.6</td><td>41.8</td><td>40.9</td><td>23.2</td><td>20.8</td><td>22.0</td><td>21.4</td></tr><tr><td>French</td><td>50.9</td><td>48.8</td><td>48.0</td><td>47.8</td><td>30.7</td><td>28.8</td><td>29.4</td><td>29.2</td></tr><tr><td>Spanish</td><td>29.1</td><td>28.9</td><td>28.3</td><td>28.8</td><td>22.2</td><td>20.4</td><td>20.8</td><td>20.6</td></tr><tr><td>Portuguese</td><td>51.2</td><td>48.6</td><td>47.3</td><td>47.2</td><td>28.5</td><td>26.8</td><td>25.7</td><td>25.6</td></tr><tr><td>Indonesian</td><td>45.3</td><td>43.7</td><td>42.1</td><td>41.7</td><td>28.8</td><td>26.9</td><td>25.4</td><td>25.2</td></tr><tr><td>Italian</td><td>32.7</td><td>30.7</td><td>31.9</td><td>30.9</td><td>23.1</td><td>21.1</td><td>21.9</td><td>21.3</td></tr><tr><td>Korean</td><td>33.9</td><td>31.8</td><td>30.8</td><td>31.7</td><td>25.1</td><td>23.4</td><td>21.9</td><td>22.8</td></tr><tr><td>Russian</td><td>33.8</td><td>31.8</td><td>33.1</td><td>33.2</td><td>21.5</td><td>18.9</td><td>20.0</td><td>19.9</td></tr><tr><td>Thai</td><td>65.4</td><td>62.9</td><td>64.8</td><td>64.4</td><td>58.0</td><td>55.5</td><td>57.2</td><td>56.4</td></tr><tr><td>Vietnamese</td><td>43.0</td><td>41.8</td><td>41.1</td><td>40.2</td><td>31.6</td><td>30.5</td><td>28.1</td><td>28.4</td></tr><tr><td>Japanese</td><td>53.2</td><td>50.6</td><td>51.3</td><td>50.8</td><td>45.6</td><td>41.6</td><td>43.0</td><td>42.0</td></tr><tr><td>Turkish</td><td>30.4</td><td>27.6</td><td>29.3</td><td>29.0</td><td>16.8</td><td>14.6</td><td>15.9</td><td>16.0</td></tr><tr><td>Hindi</td><td>33.1</td><td>29.1</td><td>28.2</td><td>29.2</td><td>19.1</td><td>14.3</td><td>17.6</td><td>17.5</td></tr><tr><td>Malay</td><td>39.6</td><td>37.2</td><td>35.9</td><td>36.0</td><td>24.1</td><td>21.7</td><td>21.0</td><td>20.5</td></tr><tr><td>Dutch</td><td>30.1</td><td>28.2</td><td>28.1</td><td>28.8</td><td>21.0</td><td>18.8</td><td>19.3</td><td>19.1</td></tr><tr><td>Urdu</td><td>25.0</td><td>22.1</td><td>23.0</td><td>23.0</td><td>15.5</td><td>8.6</td><td>14.9</td><td>14.7</td></tr><tr><td>Norwegian</td><td>35.3</td><td>32.8</td><td>33.1</td><td>33.7</td><td>20.3</td><td>17.8</td><td>18.4</td><td>18.9</td></tr><tr><td>Swedish</td><td>47.5</td><td>44.1</td><td>45.7</td><td>45.8</td><td>25.4</td><td>23.0</td><td>23.7</td><td>24.1</td></tr><tr><td>Danish</td><td>48.4</td><td>45.2</td><td>45.7</td><td>45.2</td><td>25.7</td><td>22.8</td><td>23.5</td><td>23.2</td></tr><tr><td>Hebrew</td><td>36.4</td><td>29.9</td><td>36.5</td><td>35.4</td><td>18.2</td><td>14.5</td><td>17.7</td><td>17.4</td></tr><tr><td>Finnish</td><td>30.1</td><td>26.0</td><td>32.1</td><td>32.3</td><td>18.1</td><td>15.3</td><td>18.6</td><td>17.7</td></tr><tr><td>Polish</td><td>25.2</td><td>22.5</td><td>24.9</td><td>23.4</td><td>17.5</td><td>15.0</td><td>15.6</td><td>15.5</td></tr><tr><td>Icelandic</td><td>28.5</td><td>27.2</td><td>29.6</td><td>28.2</td><td>16.2</td><td>13.5</td><td>16.0</td><td>15.6</td></tr><tr><td>Czech</td><td>35.9</td><td>32.5</td><td>33.3</td><td>33.7</td><td>20.4</td><td>18.1</td><td>19.4</td><td>19.0</td></tr><tr><td>Filipino</td><td>35.0</td><td>32.0</td><td>32.1</td><td>33.1</td><td>22.3</td><td>19.0</td><td>20.7</td><td>20.7</td></tr><tr><td>Persian</td><td>30.7</td><td>27.3</td><td>25.1</td><td>25.9</td><td>19.5</td><td>16.4</td><td>15.9</td><td>15.9</td></tr><tr><td>Greek</td><td>30.0</td><td>27.8</td><td>30.0</td><td>29.4</td><td>18.4</td><td>15.9</td><td>17.6</td><td>17.5</td></tr><tr><td>Asturian</td><td>32.4</td><td>27.9</td><td>31.5</td><td>30.4</td><td>20.4</td><td>16.2</td><td>18.6</td><td>18.1</td></tr><tr><td>Belarusian</td><td>16.4</td><td>14.7</td><td>16.4</td><td>16.5</td><td>12.6</td><td>10.8</td><td>12.1</td><td>12.2</td></tr><tr><td>Bulgarian</td><td>45.0</td><td>40.7</td><td>41.7</td><td>42.5</td><td>25.6</td><td>23.0</td><td>24.3</td><td>24.0</td></tr><tr><td>Bengali</td><td>18.6</td><td>15.7</td><td>14.3</td><td>15.0</td><td>10.6</td><td>9.2</td><td>9.0</td><td>9.3</td></tr><tr><td>Bosnian</td><td>37.5</td><td>34.0</td><td>36.3</td><td>35.2</td><td>21.4</td><td>18.6</td><td>19.9</td><td>19.4</td></tr><tr><td>Catalan</td><td>43.9</td><td>41.5</td><td>42.7</td><td>42.9</td><td>26.6</td><td>17.2</td><td>25.0</td><td>25.1</td></tr><tr><td>Cebuano</td><td>28.5</td><td>12.7</td><td>28.5</td><td>29.2</td><td>19.0</td><td>5.6</td><td>17.5</td><td>17.7</td></tr><tr><td>Estonian</td><td>30.8</td><td>26.3</td><td>31.4</td><td>30.6</td><td>18.9</td><td>13.6</td><td>17.5</td><td>17.2</td></tr><tr><td>Galician</td><td>37.4</td><td>35.4</td><td>36.6</td><td>35.9</td><td>23.9</td><td>22.0</td><td>22.7</td><td>22.3</td></tr><tr><td>Gujarati</td><td>23.8</td><td>20.9</td><td>21.0</td><td>21.5</td><td>14.3</td><td>10.8</td><td>12.3</td><td>12.3</td></tr><tr><td>Croatian</td><td>33.3</td><td>30.7</td><td>33.4</td><td>32.6</td><td>21.3</td><td>18.5</td><td>19.2</td><td>18.6</td></tr><tr><td>Hungarian</td><td>29.5</td><td>24.9</td><td>28.1</td><td>27.6</td><td>18.8</td><td>15.8</td><td>17.3</td><td>16.7</td></tr><tr><td>Javanese</td><td>26.8</td><td>24.4</td><td>16.4</td><td>22.1</td><td>16.5</td><td>14.8</td><td>9.0</td><td>13.0</td></tr><tr><td>Kazakh</td><td>24.9</td><td>21.1</td><td>20.6</td><td>22.5</td><td>15.0</td><td>12.4</td><td>12.9</td><td>13.1</td></tr><tr><td>Kannada</td><td>20.0</td><td>17.0</td><td>16.1</td><td>17.2</td><td>11.7</td><td>6.9</td><td>9.5</td><td>10.0</td></tr><tr><td>Kyrgyz</td><td>15.3</td><td>12.6</td><td>15.1</td><td>14.7</td><td>10.4</td><td>7.8</td><td>10.0</td><td>9.5</td></tr><tr><td>Latvian</td><td>36.1</td><td>31.0</td><td>35.4</td><td>35.3</td><td>21.9</td><td>17.8</td><td>20.1</td><td>19.5</td></tr><tr><td>Macedonian</td><td>38.1</td><td>34.0</td><td>38.5</td><td>38.2</td><td>22.3</td><td>20.1</td><td>22.0</td><td>21.6</td></tr><tr><td>Malayalam</td><td>19.3</td><td>11.1</td><td>16.1</td><td>16.1</td><td>10.4</td><td>5.2</td><td>9.9</td><td>9.4</td></tr><tr><td>Marathi</td><td>17.7</td><td>11.6</td><td>15.9</td><td>16.2</td><td>11.3</td><td>8.1</td><td>9.4</td><td>10.1</td></tr><tr><td>Punjabi</td><td>26.1</td><td>23.1</td><td>24.1</td><td>24.6</td><td>15.7</td><td>8.9</td><td>14.3</td><td>14.1</td></tr><tr><td>Romanian</td><td>42.0</td><td>39.9</td><td>41.4</td><td>42.0</td><td>25.6</td><td>22.8</td><td>23.6</td><td>23.4</td></tr><tr><td>Slovak</td><td>35.3</td><td>31.4</td><td>34.8</td><td>34.6</td><td>19.6</td><td>16.4</td><td>19.2</td><td>18.5</td></tr><tr><td>Slovenian</td><td>32.8</td><td>28.5</td><td>33.5</td><td>32.9</td><td>20.1</td><td>17.7</td><td>20.9</td><td>20.2</td></tr><tr><td>Swahili</td><td>36.3</td><td>30.9</td><td>32.1</td><td>32.2</td><td>20.4</td><td>9.3</td><td>18.5</td><td>18.3</td></tr><tr><td>Tajik</td><td>23.8</td><td>18.3</td><td>22.1</td><td>22.9</td><td>14.6</td><td>10.9</td><td>14.3</td><td>14.2</td></tr><tr><td>Azerbaijani</td><td>13.7</td><td>9.8</td><td>15.2</td><td>14.7</td><td>11.5</td><td>9.7</td><td>11.3</td><td>10.8</td></tr><tr><td>Ukrainian</td><td>31.7</td><td>29.2</td><td>29.8</td><td>30.1</td><td>19.5</td><td>15.0</td><td>17.0</td><td>17.6</td></tr><tr><td>Average</td><td>33.8</td><td>30.4</td><td>31.8</td><td>31.8</td><td>21.4</td><td>18.1</td><td>19.6</td><td>19.5</td></tr></tbody></table>

Table 15: Multilingual translation performance on the FLEURS xx2en and xx2zh test sets. Results are reported using BLEU (↑). The best results are highlighted in bold.

<table><tbody><tr><td></td><td colspan="4">xx2en (Other Languages → English)</td><td colspan="4">xx2zh (Other Languages → Chinese)</td></tr><tr><td>Language</td><td>Qwen3.5-Omni-Plus</td><td>Qwen3.5-Omni-Flash</td><td>Gemin-3.1-Pro</td><td>Gemini-3-Flash</td><td>Qwen3.5-Omni-Plus</td><td>Qwen3.5-Omni-Flash</td><td>Gemin-3.1-Pro</td><td>Gemini-3-Flash</td></tr><tr><td>Chinese</td><td>32.2</td><td>31.2</td><td>30.1</td><td>29.5</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>English</td><td>–</td><td>–</td><td>–</td><td>–</td><td>47.8</td><td>46.6</td><td>47.4</td><td>46.3</td></tr><tr><td>Cantonese</td><td>30.3</td><td>29.9</td><td>27.7</td><td>26.3</td><td>36.8</td><td>37.5</td><td>21.2</td><td>22.4</td></tr><tr><td>Arabic</td><td>42.9</td><td>40.0</td><td>42.1</td><td>42.3</td><td>40.2</td><td>37.3</td><td>40.5</td><td>40.5</td></tr><tr><td>German</td><td>44.6</td><td>44.1</td><td>43.9</td><td>43.9</td><td>43.3</td><td>42.8</td><td>42.1</td><td>41.8</td></tr><tr><td>French</td><td>43.5</td><td>42.0</td><td>41.6</td><td>41.3</td><td>41.6</td><td>40.3</td><td>41.6</td><td>41.4</td></tr><tr><td>Spanish</td><td>32.3</td><td>31.3</td><td>30.3</td><td>30.4</td><td>38.8</td><td>38.5</td><td>38.4</td><td>38.2</td></tr><tr><td>Portuguese</td><td>49.4</td><td>48.2</td><td>47.7</td><td>47.5</td><td>43.6</td><td>41.8</td><td>42.5</td><td>42.4</td></tr><tr><td>Indonesian</td><td>45.7</td><td>43.1</td><td>45.1</td><td>44.9</td><td>43.5</td><td>41.4</td><td>42.8</td><td>42.9</td></tr><tr><td>Italian</td><td>34.4</td><td>31.9</td><td>30.8</td><td>31.0</td><td>40.8</td><td>39.5</td><td>39.7</td><td>39.4</td></tr><tr><td>Korean</td><td>34.1</td><td>32.4</td><td>32.0</td><td>32.1</td><td>39.9</td><td>37.5</td><td>37.0</td><td>37.2</td></tr><tr><td>Russian</td><td>38.6</td><td>37.2</td><td>36.7</td><td>36.5</td><td>41.7</td><td>39.5</td><td>41.1</td><td>40.4</td></tr><tr><td>Thai</td><td>34.1</td><td>32.4</td><td>34.2</td><td>33.0</td><td>40.2</td><td>37.9</td><td>40.0</td><td>39.5</td></tr><tr><td>Vietnamese</td><td>36.4</td><td>34.9</td><td>36.1</td><td>35.4</td><td>38.7</td><td>36.3</td><td>39.2</td><td>38.9</td></tr><tr><td>Japanese</td><td>30.4</td><td>29.2</td><td>29.5</td><td>29.4</td><td>38.0</td><td>35.7</td><td>35.6</td><td>34.8</td></tr><tr><td>Turkish</td><td>40.3</td><td>39.1</td><td>39.5</td><td>38.9</td><td>41.8</td><td>40.3</td><td>40.6</td><td>41.0</td></tr><tr><td>Hindi</td><td>38.8</td><td>36.2</td><td>39.3</td><td>39.2</td><td>38.9</td><td>36.9</td><td>38.0</td><td>38.4</td></tr><tr><td>Malay</td><td>42.9</td><td>41.1</td><td>44.8</td><td>42.4</td><td>41.0</td><td>39.5</td><td>42.4</td><td>41.5</td></tr><tr><td>Dutch</td><td>33.3</td><td>32.2</td><td>31.2</td><td>30.4</td><td>40.1</td><td>38.5</td><td>39.4</td><td>39.4</td></tr><tr><td>Urdu</td><td>35.5</td><td>31.5</td><td>32.9</td><td>32.6</td><td>37.2</td><td>33.8</td><td>36.9</td><td>36.5</td></tr><tr><td>Norwegian</td><td>43.5</td><td>42.1</td><td>42.6</td><td>41.4</td><td>42.2</td><td>39.6</td><td>40.9</td><td>40.8</td></tr><tr><td>Swedish</td><td>47.2</td><td>45.2</td><td>46.6</td><td>44.7</td><td>42.9</td><td>40.7</td><td>41.5</td><td>41.6</td></tr><tr><td>Danish</td><td>45.4</td><td>44.0</td><td>44.7</td><td>42.7</td><td>43.4</td><td>41.1</td><td>41.7</td><td>40.6</td></tr><tr><td>Hebrew</td><td>39.7</td><td>36.4</td><td>42.5</td><td>39.9</td><td>36.7</td><td>34.1</td><td>40.3</td><td>37.8</td></tr><tr><td>Finnish</td><td>36.9</td><td>35.0</td><td>36.7</td><td>35.5</td><td>40.8</td><td>38.7</td><td>41.0</td><td>40.6</td></tr><tr><td>Polish</td><td>32.1</td><td>30.5</td><td>30.4</td><td>29.7</td><td>38.4</td><td>36.0</td><td>37.9</td><td>37.4</td></tr><tr><td>Icelandic</td><td>31.5</td><td>27.5</td><td>35.1</td><td>35.8</td><td>38.2</td><td>31.8</td><td>37.9</td><td>37.5</td></tr><tr><td>Czech</td><td>42.1</td><td>39.3</td><td>40.1</td><td>39.5</td><td>40.6</td><td>39.4</td><td>40.6</td><td>40.0</td></tr><tr><td>Filipino</td><td>42.7</td><td>40.9</td><td>45.0</td><td>44.3</td><td>41.0</td><td>38.0</td><td>42.6</td><td>41.5</td></tr><tr><td>Persian</td><td>40.2</td><td>36.8</td><td>38.0</td><td>38.1</td><td>40.2</td><td>37.0</td><td>41.1</td><td>41.1</td></tr><tr><td>Greek</td><td>36.0</td><td>32.5</td><td>35.3</td><td>35.4</td><td>38.0</td><td>32.8</td><td>39.0</td><td>38.4</td></tr><tr><td>Asturian</td><td>37.0</td><td>35.1</td><td>37.2</td><td>35.5</td><td>37.7</td><td>34.0</td><td>38.1</td><td>36.7</td></tr><tr><td>Belarusian</td><td>23.1</td><td>19.9</td><td>20.6</td><td>19.9</td><td>33.3</td><td>31.2</td><td>33.7</td><td>33.7</td></tr><tr><td>Bulgarian</td><td>39.6</td><td>36.0</td><td>41.3</td><td>40.0</td><td>40.9</td><td>36.6</td><td>41.0</td><td>40.4</td></tr><tr><td>Bengali</td><td>32.0</td><td>27.6</td><td>34.9</td><td>34.3</td><td>35.8</td><td>32.4</td><td>38.1</td><td>37.5</td></tr><tr><td>Bosnian</td><td>43.1</td><td>40.8</td><td>42.7</td><td>42.4</td><td>41.5</td><td>39.1</td><td>42.3</td><td>41.4</td></tr><tr><td>Catalan</td><td>46.6</td><td>42.3</td><td>46.2</td><td>45.5</td><td>42.2</td><td>38.9</td><td>42.6</td><td>41.5</td></tr><tr><td>Cebuano</td><td>37.3</td><td>26.3</td><td>38.9</td><td>38.2</td><td>34.0</td><td>26.5</td><td>36.7</td><td>36.6</td></tr><tr><td>Estonian</td><td>35.7</td><td>28.3</td><td>40.1</td><td>38.3</td><td>38.0</td><td>32.2</td><td>41.7</td><td>41.2</td></tr><tr><td>Galician</td><td>40.8</td><td>38.6</td><td>39.6</td><td>38.6</td><td>40.9</td><td>39.1</td><td>41.6</td><td>40.9</td></tr><tr><td>Gujarati</td><td>33.4</td><td>28.3</td><td>40.3</td><td>39.7</td><td>35.8</td><td>31.4</td><td>39.7</td><td>39.0</td></tr><tr><td>Croatian</td><td>39.5</td><td>36.3</td><td>38.0</td><td>36.9</td><td>40.0</td><td>37.7</td><td>40.2</td><td>39.6</td></tr><tr><td>Hungarian</td><td>35.5</td><td>29.6</td><td>35.3</td><td>33.3</td><td>39.1</td><td>33.9</td><td>40.0</td><td>37.4</td></tr><tr><td>Javanese</td><td>35.9</td><td>28.4</td><td>38.4</td><td>36.5</td><td>34.9</td><td>28.3</td><td>36.7</td><td>36.3</td></tr><tr><td>Kazakh</td><td>34.4</td><td>26.9</td><td>35.7</td><td>35.5</td><td>37.1</td><td>31.4</td><td>39.3</td><td>38.8</td></tr><tr><td>Kannada</td><td>26.4</td><td>19.8</td><td>33.1</td><td>33.6</td><td>32.3</td><td>26.1</td><td>38.0</td><td>37.8</td></tr><tr><td>Kyrgyz</td><td>22.2</td><td>17.1</td><td>24.8</td><td>23.7</td><td>29.9</td><td>24.8</td><td>33.7</td><td>32.8</td></tr><tr><td>Latvian</td><td>33.7</td><td>25.2</td><td>38.0</td><td>37.9</td><td>37.1</td><td>30.0</td><td>41.4</td><td>40.6</td></tr><tr><td>Macedonian</td><td>43.3</td><td>39.6</td><td>43.1</td><td>41.9</td><td>41.6</td><td>38.0</td><td>42.1</td><td>41.8</td></tr><tr><td>Malayalam</td><td>31.2</td><td>25.7</td><td>34.9</td><td>34.2</td><td>36.1</td><td>31.5</td><td>38.4</td><td>38.0</td></tr><tr><td>Marathi</td><td>33.5</td><td>25.7</td><td>36.7</td><td>35.5</td><td>34.6</td><td>29.4</td><td>38.8</td><td>37.9</td></tr><tr><td>Punjabi</td><td>33.0</td><td>26.9</td><td>38.3</td><td>36.5</td><td>35.1</td><td>29.9</td><td>37.4</td><td>36.4</td></tr><tr><td>Romanian</td><td>43.5</td><td>39.7</td><td>42.3</td><td>41.5</td><td>42.0</td><td>38.7</td><td>42.4</td><td>41.3</td></tr><tr><td>Slovak</td><td>39.7</td><td>38.4</td><td>39.9</td><td>38.8</td><td>39.2</td><td>38.1</td><td>40.2</td><td>39.3</td></tr><tr><td>Slovenian</td><td>31.7</td><td>26.5</td><td>34.7</td><td>33.2</td><td>34.5</td><td>30.1</td><td>38.4</td><td>37.3</td></tr><tr><td>Swahili</td><td>35.0</td><td>27.4</td><td>42.6</td><td>40.5</td><td>33.9</td><td>26.8</td><td>39.2</td><td>37.9</td></tr><tr><td>Tajik</td><td>33.9</td><td>29.0</td><td>34.5</td><td>33.3</td><td>36.7</td><td>32.7</td><td>38.9</td><td>38.1</td></tr><tr><td>Azerbaijani</td><td>25.0</td><td>22.0</td><td>23.7</td><td>23.3</td><td>33.4</td><td>30.5</td><td>33.5</td><td>33.3</td></tr><tr><td>Ukrainian</td><td>42.0</td><td>40.1</td><td>41.7</td><td>41.4</td><td>41.7</td><td>39.7</td><td>41.6</td><td>40.7</td></tr><tr><td>Average</td><td>37.0</td><td>33.5</td><td>37.4</td><td>36.6</td><td>38.9</td><td>35.7</td><td>39.4</td><td>38.9</td></tr></tbody></table>