---
title: "Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"
source: "https://ar5iv.labs.arxiv.org/html/2409.12191"
author:
published:
created: 2026-05-30
description: "We present the Qwen2-VL Series, an advanced upgrade of the previous Qwen-VL models that redefines the conventional predetermined-resolution approach in visual processing.Qwen2-VL introduces the Naive Dynamic Resolutio…"
tags:
  - "clippings"
---
Peng Wang\* Shuai Bai\* Sinan Tan\* Shijie Wang\* Zhihao Fan\* Jinze Bai\* <sup>†</sup>  
Keqin Chen Xuejing Liu Jialin Wang Wenbin Ge Yang Fan Kai Dang Mengfei Du  
Xuancheng Ren Rui Men Dayiheng Liu Chang Zhou Jingren Zhou Junyang Lin <sup>†</sup>  
Qwen Team Alibaba Group

###### Abstract

We present the Qwen2-VL Series, an advanced upgrade of the previous Qwen-VL models that redefines the conventional predetermined-resolution approach in visual processing. Qwen2-VL introduces the Naive Dynamic Resolution mechanism, which enables the model to dynamically process images of varying resolutions into different numbers of visual tokens. This approach allows the model to generate more efficient and accurate visual representations, closely aligning with human perceptual processes. The model also integrates Multimodal Rotary Position Embedding (M-RoPE), facilitating the effective fusion of positional information across text, images, and videos. We employ a unified paradigm for processing both images and videos, enhancing the model’s visual perception capabilities. To explore the potential of large multimodal models, Qwen2-VL investigates the scaling laws for large vision-language models (LVLMs). By scaling both the model size-with versions at 2B, 8B, and 72B parameters-and the amount of training data, the Qwen2-VL Series achieves highly competitive performance. Notably, the Qwen2-VL-72B model achieves results comparable to leading models such as GPT-4o and Claude3.5-Sonnet across various multimodal benchmarks, outperforming other generalist models. Code is available at [https://github.com/QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL).

<sup>†</sup>

## Introduction

In the realm of artificial intelligence, Large Vision-Language Models (LVLMs) represent a significant leap forward, building upon the strong textual processing capabilities of traditional large language models. These advanced models now encompass the ability to interpret and analyze a broader spectrum of data, including images, audio, and video. This expansion of capabilities has transformed LVLMs into indispensable tools for tackling a variety of real-world challenges. Recognized for their unique capacity to condense extensive and intricate knowledge into functional representations, LVLMs are paving the way for more comprehensive cognitive systems. By integrating diverse data forms, LVLMs aim to more closely mimic the nuanced ways in which humans perceive and interact with their environment. This allows these models to provide a more accurate representation of how we engage with and perceive our environment

Recent advancements in large vision-language models (LVLMs) [^48] [^54] [^22] [^120] [^34] [^11] [^53] [^99] [^71] [^93] have led to significant improvements in a short span. These models [^70] [^94] [^95] [^21] [^10] generally follow a common approach of visual encoder $\rightarrow$ cross-modal connector $\rightarrow$ LLM. This setup, combined with next-token prediction as the primary training method and the availability of high-quality datasets [^53] [^118] [^14] [^47], has driven much of the progress. Additional factors like larger model architectures [^1], higher-resolution images [^46] [^51], and advanced techniques such as mixture-of-expert models (MoE) [^99] [^108], model ensembles [^52], and more sophisticated connectors [^107] between visual and textual modalities have also played a key role in enhancing LVLMs’ ability to process complex visual and textual information more effectively.

However, current large vision-language models (LVLMs) are typically constrained by a fixed image input size. Standard LVLMs encode input images to a fixed resolution (e.g., 224×224), often by either downsampling or upsampling the images [^120] [^34], or by employing a scale-then-padding approach [^54] [^53]. While this one-size-fits-all strategy enables processing of images at consistent resolutions, it also limits the model’s ability to capture information at different scales, particularly leading to a significant loss of detailed information in high-resolution images. Consequently, such models fall short of perceiving visual information with the same sensitivity to scale and detail as human vision.

Additionally, most LVLMs rely on a static, frozen CLIP-style [^78] vision encoder, raising concerns about whether the visual representations produced by such pre-trained models are adequate, particularly for complex reasoning tasks and processing intricate details within images. Recent works [^11] [^107] have attempted to address these limitations by fine-tuning the vision transformer (ViT) during the LVLM training process, which has shown to yield improved results. To further enhance the model’s adaptability to varying resolutions, we introduce dynamic resolution training in the LVLM training process. Specifically, we employ a 2D Rotary Position Embedding (RoPE) in the ViT, thus allowing the model to better capture information across different spatial scales.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2_vl_example.jpg)

Figure 1: Qwen2-VL capabilities: Multilingual image text understanding, code/math reasoning, video analysis, live chat, agent potential, and more. See Appendix for details.

When it comes to video content, which is essentially a sequence of frames, many existing models continue to treat it as an independent modality. However, understanding the dynamic nature of reality, as manifested in videos, is crucial for models aiming to grasp the complexities of the real world. Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions. The use of one-dimensional position embeddings in current models significantly limits their ability to model three-dimensional space and temporal dynamics effectively. To bridge this gap, we have developed Multimodal Rotary Position Embedding (M-RoPE), which employs separate components to represent temporal and spatial information. This enables the model to naturally comprehend dynamic content, such as videos or streaming data, improving its ability to understand and interact with the world.

Furthermore, compared to the scaling of large language models (LLMs), current LVLMs are still in the early stages of exploring the impact of scaling in terms of training data and model parameters. The exploration of scaling laws for LVLMs—how increases in model and data size affect performance—remains an open and promising area of research.

In this work, we introduce the newest addition to the large vision-language models of the Qwen family: Qwen2-VL series, which comprises three open-weight models with total parameter counts of 2 billion, 8 billion, and 72 billion. As shown in Figure 1, the key advances in Qwen2-VL include:

- State-of-the-art understanding across various resolutions and aspect ratios: Qwen2-VL achieves leading performance on visual benchmarks, including DocVQA, InfoVQA, RealWorldQA, MTVQA, MathVista, and others.
- Comprehension of extended-duration videos (20 min+): Qwen2-VL is capable of understanding videos over 20 minutes in length, enhancing its ability to perform high-quality video-based question answering, dialogue, content creation, and more.
- Robust agent capabilities for device operation: With advanced reasoning and decision-making abilities, Qwen2-VL can be integrated with devices such as mobile phones, robots, etc., enabling autonomous operation based on visual inputs and text instructions.
- Multilingual support: To serve a global audience, beyond English and Chinese, Qwen2-VL now supports multilingual context understanding within images, including most European languages, Japanese, Korean, Arabic, Vietnamese, and others.

Table 1: Model descriptions of Qwen2-VL.

| Model Name | Vision Encoder | LLM | Model Description |
| --- | --- | --- | --- |
| Qwen2-VL-2B | 675M | 1.5B | The most efficient model, designed to run on-device. It delivers adequate performance for most scenarios with limited resources. |
| Qwen2-VL-7B | 675M | 7.6B | The performance-optimized model in terms of cost, significantly upgraded for text recognition and video understanding capabilities. It delivers significant performance across a broad range of visual tasks. |
| Qwen2-VL-72B | 675M | 72B | The most capable model, further improvements in visual reasoning, instruction-following, decision-making, and agent capabilities. It delivers optimal performance on most complex tasks. |

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2_vl_frame.jpg)

Figure 2: Qwen2-VL is capable of accurately identifying and comprehending the content within images, regardless of their clarity, resolution, or extreme aspect ratios.

## Approach

The Qwen2-VL series consists of models of 3 sizes, which are Qwen2-VL-2B, Qwen2-VL-7B and Qwen2-VL-72B. Table 1 lists the hyper-parameters and important information. Notably, Qwen2-VL employs a 675M parameter ViT across various-sized LLMs, ensuring that the computational load of the ViT remains constant regardless of the scale of the LLM.

### 2.1 Model Architecture

Figure 2 illustrates the comprehensive structure of Qwen2-VL. We have retained the Qwen-VL [^11] framework, which integrates vision encoders and language models. For various scale adaptations, we have implemented a Vision Transformer (ViT) [^26] with approximately 675 million parameters, adept at handling both image and video inputs. In terms of language processing, we have opted for the more powerful Qwen2 [^104] series of language models. To further enhance the model’s ability to effectively perceive and comprehend visual information in videos, we introduced several key upgrades:

##### Naive Dynamic Resolution

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/mrope.png)

Figure 3: A demonstration of M-RoPE. By decomposing rotary embedding into temporal, height, and width components, M-RoPE can explicitly model the positional information of text, images, and video in LLM.

A key architectural improvement in Qwen2-VL is the introduction of naive dynamic resolution support [^25]. Unlike Qwen-VL, Qwen2-VL can now process images of any resolution, dynamically converting them into a variable number of visual tokens.<sup>1</sup> To support this feature, we modified ViT by removing the original absolute position embeddings and introducing 2D-RoPE [^91] [^89] to capture the two-dimensional positional information of images. At the inference stage, images of varying resolutions are packed into a single sequence, with the packed length controlled to limit GPU memory usage. Furthermore, to reduce the visual tokens of each image, a simple MLP layer is employed after the ViT to compress adjacent $2\times 2$ tokens into a single token, with the special ¡—vision\_start—¿ and ¡—vision\_end—¿ tokens placed at the beginning and end of the compressed visual tokens. As a result, an image with a resolution of $224\times 224$, encoded with a ViT using patch\_size=14, will be compressed to 66 tokens before entering LLM.

##### Multimodal Rotary Position Embedding (M-RoPE)

Another key architectural enhancement is the innovation of Multimodal Rotary Position Embedding (M-RoPE). Unlike the traditional 1D-RoPE in LLMs, which is limited to encoding one-dimensional positional information, M-RoPE effectively models the positional information of multimodal inputs. This is achieved by deconstructing the original rotary embedding into three components: temporal, height, and width. For text inputs, these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE [^90]. When processing images, the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components based on the token’s position in the image. For videos, which are treated as sequences of frames, the temporal ID increments for each frame, while the height and width components follow the same ID assignment pattern as images. In scenarios where the model’s input encompasses multiple modalities, position numbering for each modality is initialized by incrementing the maximum position ID of the preceding modality by one. An illustration of M-RoPE is shown in Figure 3. M-RoPE not only enhances the modeling of positional information but also reduces the value of position IDs for images and videos, enabling the model to extrapolate to longer sequences during inference.

##### Unified Image and Video Understanding

Qwen2-VL employs a mixed training regimen incorporating both image and video data, ensuring proficiency in image understanding and video comprehension. To preserve video information as completely as possible, we sampled each video at two frames per second. Additionally, we integrated 3D convolutions [^12] with a depth of two to process video inputs, allowing the model to handle 3D tubes instead of 2D patches, thus enabling it to process more video frames without increasing the sequence length [^8]. For consistency, each image is treated as two identical frames. To balance the computational demands of long video processing with overall training efficiency, we dynamically adjust the resolution of each video frame, limiting the total number of tokens per video to 16384. This training approach strikes a balance between the model’s ability to comprehend long videos and training efficiency.

### 2.2 Training

Following Qwen-VL [^11], we adopt a three-stage training methodology. In the first stage, we focus exclusively on training the Vision Transformer (ViT) component, utilizing a vast corpus of image-text pairs to enhance semantic understanding within the Large Language Model (LLM). In the second stage, we unfreeze all parameters and train with a wider range of data for more comprehensive learning. In the final stage, we lock the ViT parameters and perform exclusive fine-tuning of the LLM using instructional datasets.

The model is pre-trained on a diverse dataset that includes image-text pairs, optical character recognition (OCR) data, interleaved image-text articles, visual question answering datasets, video dialogues, and image knowledge datasets. Our data sources primarily comprise cleaned web pages, open-source datasets, and synthetic data. The cutoff date for our data knowledge is June 2023. This diverse data composition is instrumental in developing a robust multimodal understanding capability.

During the initial pre-training phase, Qwen2-VL is exposed to a corpus of around 600 billion tokens. The LLM component of Qwen2-VL is initialized using the parameters from Qwen2 [^104], while the vision encoder of Qwen2-VL is initialized with the ViT derived from DFN. However, the fixed position embedding in the original DFN’s ViT [^28] is replaced by RoPE-2D. This pre-training phase primarily focuses on learning image-text relationships, textual content recognition within images through OCR, and image classification tasks. Such foundational training is instrumental in enabling the model to develop a robust understanding of core visual-textual correlations and alignments.

The second pre-training phase marks a significant progression, involving an additional 800 billion tokens of image-related data. This stage introduces a higher volume of mixed image-text content, facilitating a more nuanced understanding of the interplay between visual and textual information. The incorporation of visual question answering datasets refines the model’s capacity to respond to image-related queries. Moreover, the inclusion of multitasking datasets is pivotal in developing the model’s ability to navigate diverse tasks concurrently, a skill of paramount importance when dealing with complex, real-world datasets. Concurrently, purely textual data continues to play a crucial role in maintaining and advancing the model’s linguistic proficiency.

Throughout the pre-training stages, Qwen2-VL processes a cumulative total of 1.4 trillion tokens. Specifically, these tokens encompass not only text tokens but also image tokens. During the training process, however, we only provide supervision for the text tokens. This exposure to extensive and diverse linguistic and visual scenarios ensures that the model develops a deep understanding of the intricate relationships between visual and textual information, thereby laying a robust foundation for various multimodal tasks.

During the instruction fine-tuning phase, we employ the ChatML [^72] format to construct instruction-following data. This dataset encompasses not only pure text-based dialogue data but also multimodal conversational data. The multimodal components include image question-answering, document parsing, multi-image comparison, video comprehension, video stream dialogue, and agent-based interactions. Our comprehensive approach to data construction aims to enhance the model’s capability to understand and execute a wide range of instructions across various modalities. By incorporating diverse data types, we seek to develop a more versatile and robust language model capable of handling complex, multimodal tasks in addition to traditional text-based interactions.

#### 2.2.1 Data Format.

In line with Qwen-VL, Qwen2-VL also employs special tokens to distinguish vision and text inputs. Tokens ¡—vision\_start—¿ and ¡—vision\_end—¿ are inserted at the start and end of the image feature sequence to demarcate the image content.

##### Dialogue Data.

In terms of dialogue format, we construct our instruction tuning dataset using the ChatML format, where each interaction’s statement is marked with two special tokens (¡—im\_start—¿ and ¡—im\_end—¿) to facilitate dialogue termination. The sections marked in blue indicate the supervised parts.

<svg id="S2.SS2.SSS1.Px1.p2.pic1" height="128.57" overflow="visible" version="1.1" width="600"><g transform="translate(0,128.57) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 122.67 C 0 125.93 2.64 128.57 5.91 128.57 L 594.09 128.57 C 597.36 128.57 600 125.93 600 122.67 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 104.46 L 598.03 104.46 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 110.37)"><foreignObject width="556.69" height="12.3" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="S2.SS2.SSS1.Px1.p2.pic1.1.1.1.1.1" style="width:402.3pt;"><span id="S2.SS2.SSS1.Px1.p2.pic1.1.1.1.1.1.1">The Dataset Format Example of ChatML</span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="78.87" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1" style="width:402.3pt;"><span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1.1">¡—im_start—¿user</span> <span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1.2">¡—vision_start—¿Picture1.jpg¡—vision_end—¿¡—vision_start—¿Picture2.jpg¡—vision_end—¿What do the two pictures have in common?¡—im_end—¿</span> <span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1.3">¡—im_start—¿assistant</span> <span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1.4">Both pictures are of SpongeBob SquarePants. ¡—im_end—¿</span> <span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1.5">¡—im_start—¿user</span> <span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1.6">What is happening in the video?¡—vision_start—¿video.mp4¡—vision_end—¿¡—im_end—¿</span> <span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1.7">¡—im_start—¿assistant</span> <span id="S2.SS2.SSS1.Px1.p2.pic1.2.2.2.1.1.8">The protagonist in the video is frying an egg.¡—im_end—¿</span></span></foreignObject></g></g></svg>

##### Visual Grounding.

To endow the model with visual grounding capabilities, bounding box coordinates are normalized within \[0, 1000) and represented as ” $(X_{\text{top left}},Y_{\text{top left}}),(X_{\text{bottom right}},Y_{\text{bottom right}})$ ”. Tokens ¡—box\_start—¿ and ¡—box\_end—¿ are utilized to demarcate bounding box text. To accurately link bounding boxes with their textual descriptions, we introduce tokens ¡—object\_ref\_start—¿ and ¡—object\_ref\_end—¿ to indicate the content that the bounding box references, thereby allowing the model to effectively interpret and generate precise descriptions of specific regions.

<svg id="S2.SS2.SSS1.Px2.p2.pic1" height="80.14" overflow="visible" version="1.1" width="600"><g transform="translate(0,80.14) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 74.24 C 0 77.5 2.64 80.14 5.91 80.14 L 594.09 80.14 C 597.36 80.14 600 77.5 600 74.24 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 56.03 L 598.03 56.03 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 61.94)"><foreignObject width="556.69" height="12.3" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="S2.SS2.SSS1.Px2.p2.pic1.1.1.1.1.1" style="width:402.3pt;"><span id="S2.SS2.SSS1.Px2.p2.pic1.1.1.1.1.1.1">Referring Grounding</span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="30.44" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><span id="S2.SS2.SSS1.Px2.p2.pic1.2.2.2.1.1" style="width:402.3pt;"><span id="S2.SS2.SSS1.Px2.p2.pic1.2.2.2.1.1.1">¡—vision_start—¿Picture1.jpg¡—vision_end—¿</span> <span id="S2.SS2.SSS1.Px2.p2.pic1.2.2.2.1.1.2">¡—object_ref_start—¿the eyes on a giraffe¡—object_ref_end—¿¡—box_start—¿(176,106),(232,160) ¡—box_end—¿</span></span></foreignObject></g></g></svg>

##### Visual Agent.

To develop Qwen2-VL as a general-purpose VL-Agent, we treat various agent tasks, such as UI Operations, Robotic Control, Games, and Navigation, as sequential decision-making problems, enabling Qwen2-VL to accomplish tasks through multi-step action execution. For each task, we first define a set of permissible actions and keywords pattern (underline) for function call [^77]. Qwen2-VL then analyzes the observations, performs reasoning and planning, executes the selected actions, and interacts with the environment to acquire new observations. This cycle repeats iteratively until the task is successfully completed. By integrating various tools and leveraging the vision perception capabilities of large vision-language models (LVLMs), Qwen2-VL is able to iteratively execute increasingly complex tasks involving real-world visual interactions.

<svg id="S2.SS2.SSS1.Px3.p2.pic1" height="311.22" overflow="visible" version="1.1" width="600"><g transform="translate(0,311.22) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 305.31 C 0 308.58 2.64 311.22 5.91 311.22 L 594.09 311.22 C 597.36 311.22 600 308.58 600 305.31 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 287.11 L 598.03 287.11 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 293.02)"><foreignObject width="556.69" height="12.3" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="S2.SS2.SSS1.Px3.p2.pic1.1.1.1.1.1" style="width:402.3pt;"><span id="S2.SS2.SSS1.Px3.p2.pic1.1.1.1.1.1.1">Visual Agent</span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="261.52" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1" style="width:402.3pt;"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.1">¡—im_start—¿system</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.2">You are a helpful assistant.</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.3"># Actions</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.4">## You have the following actions.</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.5">### Tap</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.6">Tap: A gentle tap that commands, chooses, or navigates through a smartphone’s user interface. Parameters: [{”name”: ”point”, ”description”: ”The specific spot of interest on the monitor, denoted by the coordinates (x, y) where x and y range from 0 to 1000.”, ”required”: True}]</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.7">### Home</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.8">Home: Go to phone’s home screen. Parameters: []</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.9">### Other Actions …</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.10">## Continuously take action until the task is completed.</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.11"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.11.1">*FUNCTION*:</span> The action to take, should be one of {Actions}.</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.12"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.12.1">*ARGS*:</span> The input of the action.</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.13"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.13.1">*RESULT*:</span> Action results.</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.14"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.14.1">*RETURN*:</span> Reply based on action results. ¡—im_end—¿</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.15">¡—im_start—¿user</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.16">Find a pizza restaurant nearby in Map. ¡—vision_start—¿Screenshot_1.jpg¡—vision_end—¿ ¡—im_end—¿</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.17">¡—im_start—¿assistant</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.18">Several e-mails are displaying on the phone screen. To open Map, I need go back to the home screen and find the corresponding app icon.</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.19"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.19.1">*FUNCTION*</span>: Home <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.19.2">*ARGS*:</span> {}</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.20"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.20.1">*RESULT*:</span> ¡—vision_start—¿Screenshot_2.jpg¡—vision_end—¿</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.21"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.21.1">*RETURN*:</span> I return to the home screen. Next, I need to find the icon of Map and tap on it.</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.22"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.22.1">*FUNCTION*:</span> Tap <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.22.2">*ARGS*:</span> {”point”: (348,291)}</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.23"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.23.1">*RESULT*:</span> ¡—vision_start—¿Screenshot_3.jpg¡—vision_end—¿</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.24"><span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.24.1">*RETURN*:</span> [Thinking for the next action.]</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.25">[Other subsequent actions.] ……</span> <span id="S2.SS2.SSS1.Px3.p2.pic1.2.2.2.1.1.26">I have found the pizza restaurant nearby in Map. ¡—im_end—¿</span></span></foreignObject></g></g></svg>

### 2.3 Multimodal Model Infrastructure

The Qwen2-VL models were trained on Alibaba Cloud’s PAI-Lingjun Intelligent Computing Service [^4] with its scalable computing, auto resuming and straggler detection.

##### Storage.

We use Alibaba Cloud’s ultra-speed CPFS (Cloud Parallel File Storage) [^2] to build a storage system of Qwen2-VL pre-training and post-training. We decoupled the text data and vision data storage. We simply store text data on CPFS and use mmap for efficient access. For vision data, we use Alibaba Cloud’s OSS (Object Storage Service) [^3] for persistent storage. During training, we accessed vision data through OSS’s python-client concurrently and tuned the concurrency and retrying parameters to avoid reaching the QPS (queries per second) limit. We also found that video data decoding is a main bottleneck, especially for long videos. After several attempts with open-source [^29] and in-house software failed, we opted for a caching decoding technique. Checkpointing saves each GPU’s optimizer and model states on CPFS.

##### Parallelism.

We use 3D parallelism which combines data parallelism (DP) [^50], tensor parallelism (TP) [^44] [^83] and pipeline parallelism (PP) [^36] [^67] [^45] to scale Qwen2-VL model training. We also leverage deepspeed’s zero-1 redundancy optimizer [^79] to shard states for memory saving. Sequence parallelism (SP) [^43] with selective checkpointing activation [^17] was leveraged to reduce memory usage. When enabling TP training, we always shard the vision encoder and large language models together but not the vision merger due to its relatively few parameters. We found the TP training would result in different model shared-weights due to the convolution operator’s non-deterministic behavior <sup>2</sup>. We resolved this issue by performing offline reduction of the shared weights, thereby avoiding an additional all-reduce communication step. This approach resulted in only a minimal impact on performance. We leverage 1F1B PP [^67] for Qwen2-VL 72B training. We combine the vision encoder, vision adapter and several LLM’s decoder layers into one stage, and evenly split the remaining decoder layers. Note that the vision and text sequence lengths are dynamic for each data point. We broadcast the dynamic sequence lengths before initiating the 1F1B process and access the shape information using batch indices. We also implemented an interleaved 1F1B PP [^67] but found it is slower than the standard 1F1B setting.

##### Software.

We use PyTorch [^74] [^6] version 2.1.2 with CUDA 11.8 [^69] for training. Additionally, we leverage flash-attention [^24] [^23] [^82] for efficient training in both the vision encoder and the LLM. We also utilize fused operators [^68] such as LayerNorm [^9], RMSNorm [^115], and Adam [^58]. Besides this, we leverage the overlap of communication and computation during matrix multiplication in our training process.

## Experiments

In this section, we first evaluate the model’s performance by conducting a comparative analysis across a variety of visual benchmarks, demonstrating the advantages of our approach. Subsequently, we carry out a detailed examination of specific capabilities, including general visual perception, document understanding, multilingual recognition in images, video comprehension, and agent abilities. Finally, we present an ablation study to investigate several key components of our approach.

Table 2: Performance Comparison of Qwen2-VL Models and State-of-the-art.

| Benchmark | Previous SoTA | Claude-3.5 Sonnet | GPT-4o | Qwen2-VL-72B | Qwen2-VL-7B | Qwen2-VL-2B |
| --- | --- | --- | --- | --- | --- | --- |
| MMMU <sub>val</sub> [^111] | 66.1 [^101] | 68.3 | 69.1 | 64.5 | 54.1 | 41.1 |
| DocVQA <sub>test</sub> [^66] | 94.1 [^20] | 95.2 | 92.8 | 96.5 | 94.5 | 90.1 |
| InfoVQA <sub>test</sub> [^66] | 82.0 [^20] | \- | \- | 84.5 | 76.5 | 65.5 |
| AI2D [^40] | 87.6 [^20] | 80.2(94.7) | 84.6(94.2) | 88.1 | 83.0 | 74.7 |
| ChartQA <sub>test</sub> [^65] | 88.4 [^20] | 90.8 | 85.7 | 88.3 | 83.0 | 73.5 |
| TextVQA <sub>val</sub> [^87] | 84.4 [^20] | \- | \- | 85.5 | 84.3 | 79.7 |
| OCRBench [^57] | 852 [^106] | 788 | 736 | 877 | 866 | 809 |
| MTVQA [^92] | 23.2 [^93] | 25.7 | 27.8 | 30.9 | 25.6 | 18.1 |
| VCR <sub>en easy</sub> [^119] | 84.7 [^20] | 63.9 | 91.6 | 91.9 | 89.7 | 81.5 |
| VCR <sub>zh easy</sub> [^119] | 22.1 [^20] | 1.0 | 14.9 | 65.4 | 59.9 | 46.2 |
| RealWorldQA [^100] | 72.2 [^20] | 60.1 | 75.4 | 77.8 | 70.1 | 62.9 |
| MME <sub>sum</sub> [^30] | 2414.7 [^20] | 1920.0 | 2328.7 | 2482.7 | 2326.8 | 1872.0 |
| MMBench-EN <sub>test</sub> [^56] | 86.5 [^20] | 79.7 | 83.4 | 86.5 | 83.0 | 74.9 |
| MMBench-CN <sub>test</sub> [^56] | 86.3 [^20] | 80.7 | 82.1 | 86.6 | 80.5 | 73.5 |
| MMBench-V1.1 <sub>test</sub> [^56] | 85.5 [^20] | 78.5 | 82.2 | 85.9 | 80.7 | 72.2 |
| MMT-Bench <sub>test</sub> [^109] | 63.4 [^19] | \- | 65.5 | 71.7 | 63.7 | 54.5 |
| MMStar [^15] | 67.1 [^20] | 62.2 | 63.9 | 68.3 | 60.7 | 48.0 |
| MMVet <sub>GPT-4-Turbo</sub> [^110] | 67.5 [^71] | 66.0 | 69.1 | 74.0 | 62.0 | 49.5 |
| HallBench <sub>avg</sub> [^32] | 55.2 [^20] | 49.9 | 55.0 | 58.1 | 50.6 | 41.7 |
| MathVista <sub>testmini</sub> [^61] | 69.0 [^101] | 67.7 | 63.8 | 70.5 | 58.2 | 43.0 |
| MathVision [^96] | 30.3 [^70] | \- | 30.4 | 25.9 | 16.3 | 12.4 |
| MMMU-Pro [^112] | 46.9 [^93] | 51.5 | 51.9 | 46.2 | 43.5 | 37.6 |

Table 3: Performance of Qwen2-VL and GPT-4o on internal multilingual OCR benchmarks.

| Language | Korean | Japanese | French | German | Italian | Russian | Vietnamese | Arabic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4o | 87.8 | 88.3 | 89.7 | 88.3 | 74.1 | 96.8 | 72.0 | 75.9 |
| Qwen2-VL-72B | 94.5 | 93.4 | 94.1 | 91.5 | 89.8 | 97.2 | 73.0 | 70.7 |

Table 4: Performance of Qwen2-VL and other models on video benchmarks.

| Benchmark | Previous SoTA | Gemini 1.5-Pro | GPT-4o | Qwen2-VL-72B | Qwen2-VL-7B | Qwen2-VL-2B |
| --- | --- | --- | --- | --- | --- | --- |
| MVBench [^49] | 69.6 | \- | \- | 73.6 | 67.0 | 63.2 |
| PerceptionTest <sub>test</sub> [^75] | 66.9 | \- | \- | 68.0 | 62.3 | 53.9 |
| EgoSchema <sub>test</sub> [^63] | 62.0 | 63.2 | 72.2 | 77.9 | 66.7 | 54.9 |
| Video-MME <sub>(wo/w subs)</sub> [^31] | 66.3/69.6 | 75.0/81.3 | 71.9/77.2 | 71.2/77.8 | 63.3/69.0 | 55.6/60.4 |

Table 5: Performance Comparison of Qwen2-VL-72B across various agent benchmarks and GPT-4o. SR, GC, TM and EM are short for success rate, goal-condition success, type match and exact match. ALFRED, R2R and REVERIE are performance in valid-unseen.

<table><tbody><tr><td></td><td>Benchmark</td><td>Metric</td><td>Previous SoTA</td><td>GPT-4o</td><td>Qwen2-VL-72B</td></tr><tr><td rowspan="2">General</td><td rowspan="2">FnCall</td><td>TM</td><td>-</td><td>90.2</td><td>93.1</td></tr><tr><td>EM</td><td>-</td><td>50.0</td><td>53.2</td></tr><tr><td rowspan="2">UI Operations</td><td rowspan="2">AITZ <sup><a href="#fn:117">117</a></sup></td><td>TM</td><td>83.0 <sup><a href="#fn:33">33</a></sup></td><td>70.0</td><td>89.6</td></tr><tr><td>EM</td><td>47.7 <sup><a href="#fn:114">114</a></sup></td><td>35.3</td><td>72.1</td></tr><tr><td rowspan="4">Card Games</td><td>Number Line <sup><a href="#fn:113">113</a></sup></td><td>SR</td><td>89.4 <sup><a href="#fn:113">113</a></sup></td><td>91.5</td><td>100.0</td></tr><tr><td>BlackJack <sup><a href="#fn:113">113</a></sup></td><td>SR</td><td>40.2 <sup><a href="#fn:113">113</a></sup></td><td>34.5</td><td>42.6</td></tr><tr><td>EZPoint <sup><a href="#fn:113">113</a></sup></td><td>SR</td><td>50.0 <sup><a href="#fn:113">113</a></sup></td><td>85.5</td><td>100.0</td></tr><tr><td>Point24 <sup><a href="#fn:113">113</a></sup></td><td>SR</td><td>2.6 <sup><a href="#fn:54">54</a></sup></td><td>3.0</td><td>4.5</td></tr><tr><td rowspan="2">Robotic Control</td><td rowspan="2">ALFRED <sup><a href="#fn:84">84</a></sup></td><td>SR</td><td>67.7 <sup><a href="#fn:59">59</a></sup></td><td>-</td><td>67.8</td></tr><tr><td>GC</td><td>75.3 <sup><a href="#fn:59">59</a></sup></td><td>-</td><td>75.8</td></tr><tr><td rowspan="2">Navigation</td><td>R2R <sup><a href="#fn:5">5</a></sup></td><td>SR</td><td>79.0 <sup><a href="#fn:16">16</a></sup></td><td>43.7</td><td>51.7</td></tr><tr><td>REVERIE <sup><a href="#fn:76">76</a></sup></td><td>SR</td><td>61.0 <sup><a href="#fn:86">86</a></sup></td><td>31.6</td><td>31.0</td></tr></tbody></table>

### 3.1 Compare to SOTAs

We evaluate the visual capabilities of our model through various visual benchmarks, video tasks, and agent-based assessments. Qwen2-VL demonstrates highly competitive performance at the same scale, achieving new state-of-the-art (SoTA) results. Overall, our 72B model consistently delivers top-tier performance across most evaluation metrics, frequently surpassing even closed-source models such as GPT-4o [^73] and Claude 3.5-Sonnet [^7]. Notably, it exhibits a significant advantage in document understanding tasks. However, in the MMMU [^111] benchmark, our model still lags behind GPT-4o to some extent, indicating that Qwen2-VL-72B has room for improvement when handling more complex and challenging problem sets.

### 3.2 Quantitative Results

In this section, we present an extensive evaluation of the Qwen2-VL series across an array of datasets, offering a comprehensive understanding of the model’s capabilities in various aspects.

#### 3.2.1 General Visual Question Answering

To rigorously assess our models’ capabilities in general visual question answering tasks, we conduct extensive evaluations across a diverse array of state-of-the-art benchmarks: RealWorldQA [^100], MMStar [^15], MMVet [^110], MMT-Bench [^109], MMBench [^56], MMbench-1.1 [^56], MME [^30], and HallusionBench [^32]. The Qwen2-VL series exhibits exceptional performance across these benchmarks, with the 72B model consistently achieving or surpassing state-of-the-art results, while the 7B and 2B variants also demonstrate robust capabilities. On RealWorldQA, which evaluates real-world spatial comprehension, Qwen2-VL-72B achieves a score of 77.8, surpassing both the previous state-of-the-art (72.2) and formidable baselines such as GPT-4o (75.4), thus demonstrating superior understanding of physical environments. For MMStar, a benchmark designed to assess genuine multimodal capabilities through visually indispensable samples, Qwen2-VL-72B attains 68.3, outperforming the previous best of 67.1 and highlighting its proficiency in integrating visual and textual information. On MMVet, which evaluates the integration of core vision-language capabilities across 16 complex multimodal tasks, Qwen2-VL-72B achieves a remarkable 74.0, significantly outperforming strong competitors including GPT-4V (67.5) and showcasing its versatility in addressing diverse multimodal challenges. In the MMT-Bench evaluation, which assesses advanced reasoning and instruction following across 32 core meta-tasks and 162 subtasks in multimodal understanding, Qwen2-VL-72B achieves 71.7, markedly surpassing the previous best (63.4) and demonstrating its prowess in applying expert knowledge and executing deliberate visual recognition, localization, reasoning, and planning. On MMBench, which evaluates fine-grained abilities across 20 dimensions, Qwen2-VL-72B exhibits strong performance, achieving 86.5 on the English test set, matching the state-of-the-art, and 86.6 on the Chinese test set, establishing a new benchmark. For MME, which measures a wide spectrum of perception and cognition abilities across 14 subtasks, Qwen2-VL-72B achieves a cumulative score of 2482.7, significantly outperforming the previous best (2414.7), underscoring its advanced capabilities in both visual perception and high-level cognition tasks.

These comprehensive results underscore the Qwen2-VL series’ exceptional proficiency in general visual question answering tasks. The models demonstrate advanced capabilities in real-world spatial comprehension, genuine multimodal integration, complex reasoning, instruction following, and a broad range of perception and cognition tasks. The consistent superior performance across diverse benchmarks, particularly the outstanding results of the 72B model, positions the Qwen2-VL series as a leading solution in the field of visual question answering. Our models excel in handling visually indispensable tasks, integrating core vision-language capabilities, and demonstrating expertise across diverse multimodal scenarios, ranging from fundamental perception tasks to complex reasoning and planning. This exhaustive evaluation highlights the Qwen2-VL series’ versatility and effectiveness in addressing the multifaceted challenges posed by state-of-the-art multimodal benchmarks, thereby setting a new standard for large vision-language models.

#### 3.2.2 Document and Diagrams Reading

We tested our model’s OCR and document and diagram comprehension on DocVQA [^66], ChartQA [^65],InfoVQA [^66], TextVQA [^87],AI2D [^40] datasets. The DocVQA/InfoVQA/ChartQA dataset focuses on the model’s ability to comprehend text in documents/high-resolution infographics/charts, while the TextVQA dataset examines the ability to comprehend text in naturalistic images. The OCRBench dataset is a a dataset of mixed tasks, which focuses on mathematical formula parsing and information extraction in addition to the text-based VQA. The AI2D dataset focuses on multiple-choice questions on scientific diagrams containing text. In addition, we also tested the OCR and formula recognition capabilities of our model on OCRBench [^57], as well as the multilingual OCR capabilities of our model on the MTVQA [^92] dataset.

The experimental results show that our model achieves SoTA level in several metrics, including DocVQA, InfoVQA, TextVQA and OCRBench, demonstrating that our model has good comprehension of textual content in images from multiple domains.

#### 3.2.3 Multilingual Text Recognition and Understanding

In particular, our model surpasses all existing general-purpose LVLMs in multilingual OCR. Our model not only outperforms existing LVLMs (including proprietary models such as GPT-4o, Claude 3.5 Sonnet, etc.) on the public-available MTVQA dataset, it also outperforms GPT-4o on the in-house internal benchmark across all foreign languages except Arabic (Table 3).

#### 3.2.4 Mathematical Reasoning

We’ve conducted experiments on the MathVista [^61] and MathVision [^96] datasets to assess mathematical reasoning capabilities. MathVista is a comprehensive benchmark featuring 6,141 diverse examples of mathematical and visual tasks. The MathVision dataset comprises 3,040 math problems embedded in visual contexts from actual math competitions, covering 16 mathematical disciplines and varying in difficulty across five levels. These challenges underscore the necessity for LVLMs to exhibit strong visual comprehension, a deep understanding of mathematics, and sound logical reasoning skills. The Qwen2-VL series has demonstrated superior performance on MathVista, achieving a 70.5 outperforming other LVLMs. Additionally, it has set a new open-source benchmark on MathVision with 25.9.

#### 3.2.5 Referring Expression Comprehension

Regarding visual localization task, we evaluate Qwen2-VL on RefCOCO, RefCOCO+, and RefCOCOg datasets [^39] [^64]. The results, as depicted in Table 6, demonstrate that Qwen2-VL attains top-tier results among generalist models. Benefiting from a more rational structure design, Qwen2-VL is able to perceive details in high-resolution images, leading to significant improvements over Qwen-VL. The superiority of these models in comparison to both generalist and specialized models highlights their potential for advancing the field of visual localization and their capacity for real-world implementation in tasks requiring precise visual understanding.

Table 6: Performance Comparison on Referring Expression Comprehension Task.

<table><tbody><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td></tr><tr><td>val</td><td>test-A</td><td>test-B</td><td>val</td><td>test-A</td><td>test-B</td><td>val</td><td>test</td></tr><tr><td rowspan="11">Generalist</td><td>OFA-L <sup><a href="#fn:97">97</a></sup></td><td>80.0</td><td>83.7</td><td>76.4</td><td>68.3</td><td>76.0</td><td>61.8</td><td>67.6</td><td>67.6</td></tr><tr><td>Shikra <sup><a href="#fn:13">13</a></sup></td><td>87.0</td><td>90.6</td><td>80.2</td><td>81.6</td><td>87.4</td><td>72.1</td><td>82.3</td><td>82.2</td></tr><tr><td>Qwen-VL <sup><a href="#fn:11">11</a></sup></td><td>89.4</td><td>92.3</td><td>85.3</td><td>83.1</td><td>88.3</td><td>77.2</td><td>85.6</td><td>85.5</td></tr><tr><td>Ferretv2 <sup><a href="#fn:116">116</a></sup></td><td>92.6</td><td>95.0</td><td>88.9</td><td>87.4</td><td>92.1</td><td>81.4</td><td>89.4</td><td>90.0</td></tr><tr><td>CogVLM <sup><a href="#fn:99">99</a></sup></td><td>92.8</td><td>94.8</td><td>89.0</td><td>88.7</td><td>92.9</td><td>83.4</td><td>89.8</td><td>90.8</td></tr><tr><td>InternVL2 <sub>2b</sub> <sup><a href="#fn:20">20</a></sup></td><td>82.3</td><td>88.2</td><td>75.9</td><td>73.5</td><td>82.8</td><td>63.3</td><td>77.6</td><td>78.3</td></tr><tr><td>InternVL2 <sub>8b</sub> <sup><a href="#fn:20">20</a></sup></td><td>87.1</td><td>91.1</td><td>80.7</td><td>79.8</td><td>87.9</td><td>71.4</td><td>82.7</td><td>82.7</td></tr><tr><td>InternVL2 <sub>76b</sub> <sup><a href="#fn:20">20</a></sup></td><td>92.2</td><td>94.8</td><td>88.4</td><td>88.8</td><td>93.1</td><td>82.8</td><td>89.5</td><td>90.3</td></tr><tr><td>Qwen2-VL <sub>2b</sub></td><td>87.6</td><td>90.6</td><td>82.3</td><td>79.0</td><td>84.9</td><td>71.0</td><td>81.2</td><td>80.3</td></tr><tr><td>Qwen2-VL <sub>7b</sub></td><td>91.7</td><td>93.6</td><td>87.3</td><td>85.8</td><td>90.5</td><td>79.5</td><td>87.3</td><td>87.8</td></tr><tr><td>Qwen2-VL <sub>72b</sub></td><td>93.2</td><td>95.3</td><td>90.7</td><td>90.1</td><td>93.8</td><td>85.6</td><td>89.9</td><td>90.4</td></tr><tr><td rowspan="3">Specialist</td><td>G-DINO-L <sup><a href="#fn:55">55</a></sup></td><td>90.6</td><td>93.2</td><td>88.2</td><td>82.8</td><td>89.0</td><td>75.9</td><td>86.1</td><td>87.0</td></tr><tr><td>UNINEXT-H <sup><a href="#fn:102">102</a></sup></td><td>92.6</td><td>94.3</td><td>91.5</td><td>85.2</td><td>89.6</td><td>79.8</td><td>88.7</td><td>89.4</td></tr><tr><td>ONE-PEACE <sup><a href="#fn:98">98</a></sup></td><td>92.6</td><td>94.2</td><td>89.3</td><td>88.8</td><td>92.2</td><td>83.2</td><td>89.2</td><td>89.3</td></tr></tbody></table>

#### 3.2.6 Video Understanding

We evaluate our models on various video understanding tasks, with related benchmarks covering short videos of a few seconds to long videos of up to one hour. Table 4 presents the performance of Qwen2-VL and baseline models. Overall, Qwen2-VL demonstrates strong results across 2B, 7B, and 72B sizes, with Qwen2-VL-72B achieving the best performance on MVBench [^49], PerceptionTest [^75], and EgoSchema [^63]. This showcases Qwen2-VL’s superior capabilities in video understanding tasks, and scaling up Qwen2-VL yields significant improvements. For the challenging Video-MME benchmark [^31], which includes videos up to one hour, it is noteworthy that we limited the maximum number of frames extracted per video to 768 during evaluation, potentially impacting performance on longer videos. Future work will focus on extending Qwen2-VL to support longer sequences, thereby accommodating longer videos.

#### 3.2.7 Visual Agent

Qwen2-VL is evaluated first for its ability to interact with the environment via function calls and then for its capacity to complete complex sequential decision tasks through multiple rounds of interaction. The implementation is based on the Qwen-Agent framework [^77].

##### Function Calling

Unlike function calling in LLMs [^103] [^88] [^18], function calling in LVLMs often involves extracting information from visual cues. Due to the absence of public benchmarks for evaluating the capabilities of LVLMs in function calling, we constructed our internal evaluation dataset.

To construct the evaluation dataset, we undertook the following procedures [^18]: Scene Categorization, Image Collection, Image Content Extraction, and Question/Functions/Arguments Generation. Firstly, we classified scenes into categories based on different visual applications. Subsequently, we downloaded and meticulously selected high-quality, representative images from the internet for each category. Thereafter, utilizing an advanced LVLM [^11], we analyzed each image to extract key visual elements and textual information. Finally, based on the content information from the images, we used an advanced LLM [^104] to generate a series of questions that required specific functions to answer, along with specifying the input parameters needed for these function calls.

Similar to the function calling evaluation method in LLMs [^103], we designed two metrics to evaluate the accuracy of the function selection and the correctness of the arguments input. Specifically, Type Match(TM), is calculated as the ratio of times the model successfully invoked the correct function to the total number of calls attempted. Exact Match(EM), for each function calling, we checked whether the arguments passed to the function exactly matched those recorded in the image’s content information, calculating this correctness ratio.

As shown in Table 5, the performance of Qwen2-VL in both Type Match(93.1 vs. 90.2) and Exact Match(53.2 vs. 50.0) over GPT-4o substantiates the efficacy of Qwen2-VL’s capability in function calling, thereby underscoring its significant potential for application expansion through external tool integration.

The evaluation results demonstrated that GPT-4o underperformed, primarily due to two factors: in scenarios where uncertainty arises, GPT-4o demonstrates a conservative approach by avoiding using external tools. The Optical Character Recognition (OCR) capability of GPT-4o is outperformed by Qwen2-VL, particularly in the context of Chinese characters.

##### UI Operations/Games/Robotics/Navigation

To assess Qwen2-VL’s ability to generally handle complex tasks, we conduct evaluations across multiple VL agent tasks, including mobile operations [^117] [^81] [^62] [^80], robotic control [^42] [^84] [^37] [^59] [^38] [^35], card games [^113], and vision-language navigation [^5] [^76]. As these tasks need multiple actions to complete tasks, we keep the history (observation, action) through Qwen2-VL supports a 32K context length, then append each new observation image after every action, enabling continuous reasoning about subsequent steps.

UI Operations: we evaluate Qwen2-VL using the AITZ task [^117], which constructs a core clean test set derived from AITW [^81]. Based on common operation patterns of phone, we define actions such as tap, input and swipe [^81] for Qwen2-VL to interact with on-screen icons for task completion. For example, when Qwen2-VL is tasked with finding a pizza restaurant nearby by Google Maps, it should input ”pizza” in the search term, swipe to select the appropriate restaurant, and tap the corresponding link. Following the AITZ setting, we report both type match (correctness of tap, input, or swipe) and exact match (correctness of tap location, input text, or swipe direction). With the support of grounding capability on UI, Qwen2-VL surpasses GPT-4 and previous SoTA [^117] [^114].

Robotic Control: we evaluate Qwen2-VL on the ALFRED task [^84] in AI2THOR [^42]. The task requires agent to perform complex household tasks, such as toasting bread and slicing an apple to prepare a meal. To work in the virtual environment, we define high-level actions (GotoLocation, Pickup, PutDown, Open, Close, Clean, Heat, Cool, Slice) [^85] as the action set. Moreover, agent needs to localize objects for manipulation (e.g., it can only pick up an apple if the apple is recognized). To improve the accuracy of manipulation, we integrate SAM [^41]. ALFRED task reports task success rate (SR) (e.g., preparing dinner) and sub-goal completion metrics (GC) (e.g., whether the bread is toasted or the apple is sliced). Qwen2-VL slightly outperforms the previously specialized model ThinkBot [^59] on the valid-unseen set.

Card Games: we leverage the card game environment from RL4VLM [^113] to assess Qwen2-VL’s performance in a series of card-based games: Number Line, BlackJack, EZPoint, and Point24. Each game presents distinct challenges: (1) reaching a target number using +1 or -1 operations, (2) drawing or holding cards to compete against the dealer, (3) applying basic arithmetic operations to reach a total of 12, and (4) using arithmetic operations to achieve a total of 24. We report the success rate of the tasks. They not only evaluate agent capabilities but also require strong OCR skills to recognize these cards and understand the progression of the game. Qwen2-VL demonstrates superior performance across all tasks.

Vision-Language Navigation: we evaluate Qwen2-VL on the Vision-and-Language Navigation (VLN) task using the R2R [^5] and REVERIE [^76]. In VLN, the model must autonomously determine the next location based on instruction, current observations. We report the success rate (SR) of VLM in reaching the predetermined destination for this task. The performance of Qwen2-VL is comparable to that of GPT-4o, but both models fall significantly behind current specialized VLN models [^16] [^86]. We attribute this gap to the incomplete and unstructured map information generated by the model from multiple images. Accurately modeling maps and locations in a 3D environment remains a major challenge for multimodal models.

### 3.3 Ablation Study

In this section, we present ablation studies on image dynamic resolution, M-RoPE, and model scale. These experiments aim to provide insights into the impact of these key components on our model’s performance.

#### 3.3.1 Dynamic Resolution

Table 7: Qwen2-VL-7B under fixed/dynamic image tokens. Adjusting image sizes only results in small perturbations in performance, demonstrating the robustness to varying image sizes. Moreover, the dynamic resolution strategy achieves top-tier performance while consuming fewer tokens on average, demonstrating the efficiency of our model.

<table><tbody><tr><td>Strategy</td><td>Average Image Tokens</td><td>InfoVQA <sub>val</sub></td><td>RealWorldQA</td><td>OCRBench</td><td>MMMU</td></tr><tr><td rowspan="4">Fixed Image Tokens</td><td>64</td><td>28.85</td><td>56.47</td><td>572</td><td>53.33</td></tr><tr><td>576</td><td>65.72</td><td>65.88</td><td>828</td><td>52.78</td></tr><tr><td>1600</td><td>74.99</td><td>69.54</td><td>824</td><td>52.89</td></tr><tr><td>3136</td><td>77.27</td><td>70.59</td><td>786</td><td>53.44</td></tr><tr><td>Dynamic Image Tokens</td><td>1924</td><td>75.89</td><td>70.07</td><td>866</td><td>53.44</td></tr></tbody></table>

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/minpixels_resolution.png)

Figure 4: Qwen2-VL-7B with different min\_pixels. Small images are upscaled to surpass a specified min\_pixels threshold before input into the model. Increasing the image size within a reasonable range shows enhanced performance on perceptual tasks like InfoVQA, HallusionBench, and OCRBench.

As shown in Table 7, we compare the performance between dynamic resolution and fixed resolution. For fixed resolution, we resize the images to ensure a constant number of image tokens being input to the model, rather than resizing to a specific height and width, as this would distort the original aspect ratio. For dynamic resolution, we only set min\_pixels $=100\times 28\times 28$ and max\_pixels $=16384\times 28\times 28$, allowing the number of image tokens depend primarily on the image’s native resolution. It can be observed that adjusting image sizes only results in small perturbations in performance, demonstrating the model robustness to varying image sizes. Moreover, dynamic resolution approach is more efficient. We can observe that no single fixed resolution achieves optimal performance across all benchmarks. In contrast, the dynamic resolution approach consistently achieves top-tier performance while consuming fewer tokens on average.

Additionally, we observe that merely increasing the image size does not always lead to improved performance. It is more important to choose an appropriate resolution for different images. As detailed in Figure 4, we upscale small images to surpass a specified min\_pixels threshold. Evaluations on upscaled images shows enhanced performance on perceptual tasks like InfoVQA, HallusionBench, and OCRBench. We attribute these gains to increased computational load. However, for OCRBench, a too-high min\_pixels value leads to a severe performance decline. This is likely because OCRBench contains numerous extremely small images, and excessive enlargement causes these images to deviate from the training data distribution, turning them into out-of-distribution samples. In contrast, the effect of increasing min\_pixels on the MMMU benchmark is negligible. We hypothesize that the performance bottleneck in MMMU is more related to the model’s reasoning capability rather than image resolution.

#### 3.3.2 M-RoPE

Table 8: Ablation studies of M-RoPE. Compared to 1D-RoPE, using M-RoPE achieves better performance in downstream tasks, particularly in video benchmarks. RWQ means RealworldQA.

<table><tbody><tr><td></td><td colspan="8">Image Benchmarks</td><td colspan="3">Video Benchmarks</td></tr><tr><td></td><td>MathVista</td><td>MMB</td><td>MMStar</td><td>RWQ</td><td>DocVQA</td><td>ChartQA</td><td>InfoVQA</td><td>TextVQA</td><td>PerceptionTest</td><td>NextQA</td><td>STAR</td></tr><tr><td>1D-RoPE</td><td>39.2</td><td>58.6</td><td>36.7</td><td>54.5</td><td>82.5</td><td>68.0</td><td>50.8</td><td>71.3</td><td>46.6</td><td>43.9</td><td>55.5</td></tr><tr><td>M-RoPE</td><td>43.4</td><td>60.6</td><td>36.7</td><td>53.7</td><td>82.8</td><td>68.4</td><td>50.3</td><td>71.8</td><td>47.4</td><td>46.0</td><td>57.9</td></tr></tbody></table>

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/x1.png)

Figure 5: Evaluate the length extrapolation capability of Qwen2-VL-72B on Video-MME Medium Video. With the help of M-RoPE, the model demonstrated robust performance when the inference length exceeded the maximum training length of 16384 tokens.

In this subsection, we demonstrate the effectiveness of M-RoPE. First, we validate its capability on various downstream tasks. We employ Qwen2-1.5B and ViT-L as the backbone and report the results of the pre-trained models. As shown in Table 8, compared to 1D-RoPE, using M-RoPE achieves better performance in downstream tasks, particularly in video benchmarks. Furthermore, we assess the length extrapolation capability of M-RoPE on Video-MME medium-length videos. Figure 5 illustrates the performance of Qwen2-VL-72B at different inference lengths. Leveraging M-RoPE, the model demonstrates robust results across various inference lengths. Notably, despite limiting the maximum tokens per video to 16K during training, the model still exhibits exceptional performance at a maximum inference length of 80K tokens.

#### 3.3.3 Model Scaling

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/scale.jpg)

Figure 6: Model Performance Scaling Across Capabilities and Training Progress. As model size and the volume of training data increase, performance consistently improves across a range of capabilities and benchmarks.

We evaluate the performance of models of varying scales across multiple capability dimensions. Specifically, we categorize these dimensions into complex college-level problem-solving, mathematical abilities, document and table comprehension, general scenario question-answering, and video comprehension. The overall capability of a model is assessed by averaging its scores across different benchmarks associated with each dimension.

In particular, we use the MMMU [^111] benchmark to represent college-level problem-solving ability, while the average scores from MathVista [^61] and MathVision [^96] serve as indicators of mathematical ability. For general scenario question-answering, we compute the average score across the RealWorldQA [^100], MMBench-V1.1 [^56], MMT-Bench [^109], HallBench [^32], MMVet [^110], and MMStar [^15] benchmarks. Document and table comprehension capability is reflected through the average score from benchmarks like DocVQA [^66], InfoVQA [^66], ChartQA [^65], TextVQA [^87], OCRBench [^57], and MTVQA [^92]. Lastly, video comprehension ability is measured by averaging scores across MVBench [^49], PerceptionTest [^75], EgoSchema [^63], and Video-MME [^31].

As illustrated in Figure 6(a), there is a consistent improvement in performance with increasing model size, particularly with respect to mathematical abilities, which show a positive correlation with the number of model parameters. On the other hand, for optical character recognition (OCR)-related tasks, even smaller-scale models exhibit relatively strong performance.

As shown in Figure 6(b), we visualize the relationship between model performance and the number of training tokens during the second stage of pretraining for Qwen2-VL-7B. As the number of training tokens increases, the model performance improves; however, performance on vision question answering (VQA) tasks exhibits some fluctuation. In contrast, for tasks such as AI2D [^40] and InfoVQA [^66] —both of which involve understanding textual and graphical information in images—the model performance shows steady improvement as training data is augmented.

## Conclusion

We have presented the Qwen2-VL series, the versatile large vision-language models, including three open-weight models with total parameter counts of 2, 8, and 72 billion. Qwen2-VL matches the performance of top-tier models like GPT-4o and Claude3.5-Sonnet in a range of multimodal scenarios, surpassing all other open-weight LVLM models. Qwen2-VL series introduces naive dynamic resolution and multimodal rotary position embedding (M-RoPE) to fuse information across modals effectively and be capable of understanding videos over 20 minutes in length. With advanced reasoning and decision-making abilities, Qwen2-VL can be integrated with devices such as mobile phones, robots, etc. Furthermore, Qwen2-VL now supports understanding multilingual texts within images, including most European languages, Japanese, Korean, Arabic, Vietnamese, and others.

We have made the Qwen2-VL model weights openly accessible, which enables researchers and developers to harness the full potential in a variety of applications and research projects. We aim to advance AI technologies and enhance their beneficial effects on society by dedicating ourselves to these endeavors.

## Acknowledgements

We express our gratitude to Juan Zhu, Fan Hong, Jie Zhang, Yong Li of Alibaba Cloud’s PAI team [^4] for supporting the training infrastructure of Qwen2-VL. This work was also supported by Qwen LLM team [^104], and we especially thank Na Ni, Yichang Zhang, Jianxin Ma, Bowen Yu, Zheren Fu for their data contribution and insightful discussion.

## References

## Appendix A Model Capabilities and Qualitative Examples

In this section, we present some practical examples of our Qwen2-VL.

### A.1 General Chat and OCR

The Qwen2-VL models are now more adept at accurately describing and identifying complex information within images, as well as providing detailed background and answering related questions. Besides, the text processing capabilities of the Qwen2-VL models have seen significant improvements, particularly concerning the recognition of Chinese and English text within images.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/boxes.png)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/flowers.jpg)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/arxiv_sample.png)

Convert the content in the image to Markdown

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/screenshot_multilingual4.png)

转录图中内容，并识别图中每段内容的语言。

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/4_text_information_1.jpeg)

The text in the image reads:

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/multilingual_2.png)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/Handwritten_chinese.jpeg)

Refer to caption

### A.2 Information extraction and Visual Reasoning

A notable advancement in the Qwen2-VL models is their enhanced visual reasoning capability. This advancement allows the models to interpret and comprehend complex representations such as flowcharts, diagrams, and other symbolic systems.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2-1-triangle.jpg)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/3_1_math_cube_2x.jpg)

The image shows two geometric figures, a rectangular prism and a cube, with their dimensions labeled. The problem asks to calculate the surface area and volume of these figures.

<svg id="A1.SS2.p2.pic1" height="362.38" overflow="visible" version="1.1" width="600"><g transform="translate(0,362.38) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 356.48 C 0 359.74 2.64 362.38 5.91 362.38 L 594.09 362.38 C 597.36 362.38 600 359.74 600 356.48 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 338.27 L 598.03 338.27 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 344.18)"><foreignObject width="556.69" height="12.3" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="A1.SS2.p2.pic1.2.2.2.1.1" style="width:402.3pt;"><span id="A1.SS2.p2.pic1.2.2.2.1.1.1">Algorithmic Problem Solving</span></span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="312.68" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><figure><img src="https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2-2-Code.jpeg" id="A1.SS2.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1" width="589" height="299" alt="[Uncaptioned image]"><figcaption>[Uncaptioned image]</figcaption></figure></foreignObject></g></g></svg><svg id="A1.F16.pic1" height="4543.7" overflow="visible" version="1.1" width="600"><g transform="translate(0,4543.7) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 4537.79 C 0 4541.05 2.64 4543.7 5.91 4543.7 L 594.09 4543.7 C 597.36 4543.7 600 4541.05 600 4537.79 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 4520.82 L 598.03 4520.82 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 4526.72)"><foreignObject width="556.69" height="11.07" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="A1.F16.pic1.3.3.3.1.1" style="width:402.3pt;"><span id="A1.F16.pic1.3.3.3.1.1.1"><span id="A1.F16.pic1.3.3.3.1.1.1.1" style="font-size:90%;">Model Response</span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="4495.23" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2" style="width:402.3pt;"><span id="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1"><span id="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1" style="font-size:90%;">To solve the “Snake in Matrix” problem, we need to implement a function that takes the size of the matrix</span> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="n"><semantics id="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mi mathsize="90%" id="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">n</mi> <annotation-xml encoding="MathML-Content" id="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><ci id="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1">𝑛</ci></annotation-xml> <annotation encoding="application/x-tex" id="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1c">n</annotation></semantics></math> <span id="A1.F16.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.2" style="font-size:90%;">and a list of commands, and returns the final position of the snake after executing all the commands. Here’s a step-by-step approach to solve this problem:</span></span> <span id="A1.I1"><span id="A1.I1.i1" style="list-style-type:none;">1. <span id="A1.I1.i1.p1"><span id="A1.I1.i1.p1.1"><span id="A1.I1.i1.p1.1.1" style="font-size:90%;">Initialize Variables</span><span id="A1.I1.i1.p1.1.2" style="font-size:90%;">:</span></span> <span id="A1.I1.i1.I1"><span id="A1.I1.i1.I1.i1" style="list-style-type:none;">• <span id="A1.I1.i1.I1.i1.p1"><span id="A1.I1.i1.I1.i1.p1.1"><span id="A1.I1.i1.I1.i1.p1.1.1" style="font-size:90%;">Initialize the current position of the snake as</span> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="00"><semantics id="A1.I1.i1.I1.i1.p1.1.m1.1a"><mn mathsize="90%" id="A1.I1.i1.I1.i1.p1.1.m1.1.1" xref="A1.I1.i1.I1.i1.p1.1.m1.1.1.cmml">0</mn> <annotation-xml encoding="MathML-Content" id="A1.I1.i1.I1.i1.p1.1.m1.1b"><cn type="integer" id="A1.I1.i1.I1.i1.p1.1.m1.1.1.cmml" xref="A1.I1.i1.I1.i1.p1.1.m1.1.1">0</cn></annotation-xml></semantics></math><span id="A1.I1.i1.I1.i1.p1.1.2" style="font-size:90%;">.</span></span></span></span> <span id="A1.I1.i1.I1.i2" style="list-style-type:none;">• <span id="A1.I1.i1.I1.i2.p1"><span id="A1.I1.i1.I1.i2.p1.1"><span id="A1.I1.i1.I1.i2.p1.1.1" style="font-size:90%;">Initialize the direction of the snake as</span> <span id="A1.I1.i1.I1.i2.p1.1.2" style="font-size:90%;">RIGHT</span> <span id="A1.I1.i1.I1.i2.p1.1.3" style="font-size:90%;">(initially, the snake moves to the right).</span></span></span></span></span></span></span> <span id="A1.I1.i2" style="list-style-type:none;">2. <span id="A1.I1.i2.p1"><span id="A1.I1.i2.p1.1"><span id="A1.I1.i2.p1.1.1" style="font-size:90%;">Define Movement Functions</span><span id="A1.I1.i2.p1.1.2" style="font-size:90%;">:</span></span> <span id="A1.I1.i2.I1"><span id="A1.I1.i2.I1.i1" style="list-style-type:none;">• <span id="A1.I1.i2.I1.i1.p1"><span id="A1.I1.i2.I1.i1.p1.1"><span id="A1.I1.i2.I1.i1.p1.1.1" style="font-size:90%;">Create a function to move the snake based on the current direction and the size of the matrix</span> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="n"><semantics id="A1.I1.i2.I1.i1.p1.1.m1.1a"><mi mathsize="90%" id="A1.I1.i2.I1.i1.p1.1.m1.1.1" xref="A1.I1.i2.I1.i1.p1.1.m1.1.1.cmml">n</mi> <annotation-xml encoding="MathML-Content" id="A1.I1.i2.I1.i1.p1.1.m1.1b"><ci id="A1.I1.i2.I1.i1.p1.1.m1.1.1.cmml" xref="A1.I1.i2.I1.i1.p1.1.m1.1.1">𝑛</ci></annotation-xml> <annotation encoding="application/x-tex" id="A1.I1.i2.I1.i1.p1.1.m1.1c">n</annotation></semantics></math><span id="A1.I1.i2.I1.i1.p1.1.2" style="font-size:90%;">.</span></span></span></span> <span id="A1.I1.i2.I1.i2" style="list-style-type:none;">• <span id="A1.I1.i2.I1.i2.p1"><span id="A1.I1.i2.I1.i2.p1.1"><span id="A1.I1.i2.I1.i2.p1.1.1" style="font-size:90%;">Update the position of the snake accordingly.</span></span></span></span></span></span></span> <span id="A1.I1.i3" style="list-style-type:none;">3. <span id="A1.I1.i3.p1"><span id="A1.I1.i3.p1.1"><span id="A1.I1.i3.p1.1.1" style="font-size:90%;">Process Commands</span><span id="A1.I1.i3.p1.1.2" style="font-size:90%;">:</span></span> <span id="A1.I1.i3.I1"><span id="A1.I1.i3.I1.i1" style="list-style-type:none;">• <span id="A1.I1.i3.I1.i1.p1"><span id="A1.I1.i3.I1.i1.p1.1"><span id="A1.I1.i3.I1.i1.p1.1.1" style="font-size:90%;">Iterate through each command in the list of commands.</span></span></span></span> <span id="A1.I1.i3.I1.i2" style="list-style-type:none;">• <span id="A1.I1.i3.I1.i2.p1"><span id="A1.I1.i3.I1.i2.p1.1"><span id="A1.I1.i3.I1.i2.p1.1.1" style="font-size:90%;">For each command, update the direction and position of the snake.</span></span></span></span></span></span></span> <span id="A1.I1.i4" style="list-style-type:none;">4. <span id="A1.I1.i4.p1"><span id="A1.I1.i4.p1.1"><span id="A1.I1.i4.p1.1.1" style="font-size:90%;">Return Final Position</span><span id="A1.I1.i4.p1.1.2" style="font-size:90%;">:</span></span> <span id="A1.I1.i4.I1"><span id="A1.I1.i4.I1.i1" style="list-style-type:none;">• <span id="A1.I1.i4.I1.i1.p1"><span id="A1.I1.i4.I1.i1.p1.1"><span id="A1.I1.i4.I1.i1.p1.1.1" style="font-size:90%;">After processing all commands, return the final position of the snake.</span></span></span></span></span></span></span></span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.3"><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.3.1" style="font-size:90%;">Here’s the Python code to implement the solution:</span></span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.4"><a href="data:text/plain;base64,Y2xhc3MgU29sdXRpb246CiAgICBkZWYgZmluYWxQb3NpdGlvbk9mU25ha2Uoc2VsZiwgbjogaW50LCBjb21tYW5kczogTGlzdFtzdHJdKSAtPiBpbnQ6CiAgICAgICAgIyBJbml0aWFsaXplIHZhcmlhYmxlcwogICAgICAgIHBvc2l0aW9uID0gMAogICAgICAgIGRpcmVjdGlvbiA9ICJSSUdIVCIKICAgICAgICAjIERlZmluZSBtb3ZlbWVudCBmdW5jdGlvbnMKICAgICAgICBkZWYgbW92ZV9yaWdodCgpOgogICAgICAgICAgICBub25sb2NhbCBwb3NpdGlvbgogICAgICAgICAgICBwb3NpdGlvbiArPSAxCiAgICAgICAgZGVmIG1vdmVfZG93bigpOgogICAgICAgICAgICBub25sb2NhbCBwb3NpdGlvbgogICAgICAgICAgICBwb3NpdGlvbiArPSBuCiAgICAgICAgZGVmIG1vdmVfbGVmdCgpOgogICAgICAgICAgICBub25sb2NhbCBwb3NpdGlvbgogICAgICAgICAgICBwb3NpdGlvbiAtPSAxCiAgICAgICAgZGVmIG1vdmVfdXAoKToKICAgICAgICAgICAgbm9ubG9jYWwgcG9zaXRpb24KICAgICAgICAgICAgcG9zaXRpb24gLT0gbgogICAgICAgICMgUHJvY2VzcyBjb21tYW5kcwogICAgICAgIGZvciBjb21tYW5kIGluIGNvbW1hbmRzOgogICAgICAgICAgICBpZiBjb21tYW5kID09ICJSSUdIVCI6CiAgICAgICAgICAgICAgICBtb3ZlX3JpZ2h0KCkKICAgICAgICAgICAgZWxpZiBjb21tYW5kID09ICJET1dOIjoKICAgICAgICAgICAgICAgIG1vdmVfZG93bigpCiAgICAgICAgICAgIGVsaWYgY29tbWFuZCA9PSAiTEVGVCI6CiAgICAgICAgICAgICAgICBtb3ZlX2xlZnQoKQogICAgICAgICAgICBlbGlmIGNvbW1hbmQgPT0gIlVQIjoKICAgICAgICAgICAgICAgIG1vdmVfdXAoKQogICAgICAgICMgUmV0dXJuIGZpbmFsIHBvc2l0aW9uCiAgICAgICAgcmV0dXJuIHBvc2l0aW9u" download="">⬇</a> <span id="lstnumberx1"><span id="lstnumberx1.1" style="font-size:90%;">class</span> <span id="lstnumberx1.3" style="font-size:90%;">Solution</span><span id="lstnumberx1.4" style="font-size:90%;">:</span></span> <span id="lstnumberx2"><span id="lstnumberx2.2" style="font-size:90%;">def</span> <span id="lstnumberx2.4" style="font-size:90%;">finalPositionOfSnake</span> <span id="lstnumberx2.5" style="font-size:90%;">(</span><span id="lstnumberx2.6" style="font-size:90%;">self</span><span id="lstnumberx2.7" style="font-size:90%;">,</span><span id="lstnumberx2.9" style="font-size:90%;">n</span><span id="lstnumberx2.10" style="font-size:90%;">:</span><span id="lstnumberx2.12" style="font-size:90%;">int</span><span id="lstnumberx2.13" style="font-size:90%;">,</span><span id="lstnumberx2.15" style="font-size:90%;">commands</span><span id="lstnumberx2.16" style="font-size:90%;">:</span><span id="lstnumberx2.18" style="font-size:90%;">List</span> <span id="lstnumberx2.19" style="font-size:90%;">[</span><span id="lstnumberx2.20" style="font-size:90%;">str</span><span id="lstnumberx2.21" style="font-size:90%;">])</span> <span id="lstnumberx2.23" style="font-size:90%;">-&gt;</span> <span id="lstnumberx2.25" style="font-size:90%;">int</span><span id="lstnumberx2.26" style="font-size:90%;">:</span></span> <span id="lstnumberx3"><span id="lstnumberx3.2" style="font-size:90%;">#Initializevariables</span> </span><span id="lstnumberx4"><span id="lstnumberx4.2" style="font-size:90%;">position</span> <span id="lstnumberx4.4" style="font-size:90%;">=</span> <span id="lstnumberx4.6" style="font-size:90%;">0</span> </span><span id="lstnumberx5"><span id="lstnumberx5.2" style="font-size:90%;">direction</span> <span id="lstnumberx5.4" style="font-size:90%;">=</span> <span id="lstnumberx5.6" style="font-size:90%;">"RIGHT"</span> </span><span id="lstnumberx6"><span id="lstnumberx6.2" style="font-size:90%;">#Definemovementfunctions</span> </span><span id="lstnumberx7"><span id="lstnumberx7.2" style="font-size:90%;">def</span> <span id="lstnumberx7.4" style="font-size:90%;">move_right</span> <span id="lstnumberx7.5" style="font-size:90%;">():</span></span> <span id="lstnumberx8"><span id="lstnumberx8.2" style="font-size:90%;">nonlocal</span> <span id="lstnumberx8.4" style="font-size:90%;">position</span> </span><span id="lstnumberx9"><span id="lstnumberx9.2" style="font-size:90%;">position</span> <span id="lstnumberx9.4" style="font-size:90%;">+=</span> <span id="lstnumberx9.6" style="font-size:90%;">1</span> </span><span id="lstnumberx10"><span id="lstnumberx10.2" style="font-size:90%;">def</span> <span id="lstnumberx10.4" style="font-size:90%;">move_down</span> <span id="lstnumberx10.5" style="font-size:90%;">():</span></span> <span id="lstnumberx11"><span id="lstnumberx11.2" style="font-size:90%;">nonlocal</span> <span id="lstnumberx11.4" style="font-size:90%;">position</span> </span><span id="lstnumberx12"><span id="lstnumberx12.2" style="font-size:90%;">position</span> <span id="lstnumberx12.4" style="font-size:90%;">+=</span> <span id="lstnumberx12.6" style="font-size:90%;">n</span> </span><span id="lstnumberx13"><span id="lstnumberx13.2" style="font-size:90%;">def</span> <span id="lstnumberx13.4" style="font-size:90%;">move_left</span> <span id="lstnumberx13.5" style="font-size:90%;">():</span></span> <span id="lstnumberx14"><span id="lstnumberx14.2" style="font-size:90%;">nonlocal</span> <span id="lstnumberx14.4" style="font-size:90%;">position</span> </span><span id="lstnumberx15"><span id="lstnumberx15.2" style="font-size:90%;">position</span> <span id="lstnumberx15.4" style="font-size:90%;">-=</span> <span id="lstnumberx15.6" style="font-size:90%;">1</span> </span><span id="lstnumberx16"><span id="lstnumberx16.2" style="font-size:90%;">def</span> <span id="lstnumberx16.4" style="font-size:90%;">move_up</span> <span id="lstnumberx16.5" style="font-size:90%;">():</span></span> <span id="lstnumberx17"><span id="lstnumberx17.2" style="font-size:90%;">nonlocal</span> <span id="lstnumberx17.4" style="font-size:90%;">position</span> </span><span id="lstnumberx18"><span id="lstnumberx18.2" style="font-size:90%;">position</span> <span id="lstnumberx18.4" style="font-size:90%;">-=</span> <span id="lstnumberx18.6" style="font-size:90%;">n</span> </span><span id="lstnumberx19"><span id="lstnumberx19.2" style="font-size:90%;">#Processcommands</span> </span><span id="lstnumberx20"><span id="lstnumberx20.2" style="font-size:90%;">for</span> <span id="lstnumberx20.4" style="font-size:90%;">command</span> <span id="lstnumberx20.6" style="font-size:90%;">in</span> <span id="lstnumberx20.8" style="font-size:90%;">commands</span><span id="lstnumberx20.9" style="font-size:90%;">:</span></span> <span id="lstnumberx21"><span id="lstnumberx21.2" style="font-size:90%;">if</span> <span id="lstnumberx21.4" style="font-size:90%;">command</span> <span id="lstnumberx21.6" style="font-size:90%;">==</span> <span id="lstnumberx21.8" style="font-size:90%;">"RIGHT"</span><span id="lstnumberx21.9" style="font-size:90%;">:</span></span> <span id="lstnumberx22"><span id="lstnumberx22.2" style="font-size:90%;">move_right</span> <span id="lstnumberx22.3" style="font-size:90%;">()</span> </span><span id="lstnumberx23"><span id="lstnumberx23.2" style="font-size:90%;">elif</span> <span id="lstnumberx23.4" style="font-size:90%;">command</span> <span id="lstnumberx23.6" style="font-size:90%;">==</span> <span id="lstnumberx23.8" style="font-size:90%;">"DOWN"</span><span id="lstnumberx23.9" style="font-size:90%;">:</span></span> <span id="lstnumberx24"><span id="lstnumberx24.2" style="font-size:90%;">move_down</span> <span id="lstnumberx24.3" style="font-size:90%;">()</span> </span><span id="lstnumberx25"><span id="lstnumberx25.2" style="font-size:90%;">elif</span> <span id="lstnumberx25.4" style="font-size:90%;">command</span> <span id="lstnumberx25.6" style="font-size:90%;">==</span> <span id="lstnumberx25.8" style="font-size:90%;">"LEFT"</span><span id="lstnumberx25.9" style="font-size:90%;">:</span></span> <span id="lstnumberx26"><span id="lstnumberx26.2" style="font-size:90%;">move_left</span> <span id="lstnumberx26.3" style="font-size:90%;">()</span> </span><span id="lstnumberx27"><span id="lstnumberx27.2" style="font-size:90%;">elif</span> <span id="lstnumberx27.4" style="font-size:90%;">command</span> <span id="lstnumberx27.6" style="font-size:90%;">==</span> <span id="lstnumberx27.8" style="font-size:90%;">"UP"</span><span id="lstnumberx27.9" style="font-size:90%;">:</span></span> <span id="lstnumberx28"><span id="lstnumberx28.2" style="font-size:90%;">move_up</span> <span id="lstnumberx28.3" style="font-size:90%;">()</span> </span><span id="lstnumberx29"><span id="lstnumberx29.2" style="font-size:90%;">#Returnfinalposition</span> </span><span id="lstnumberx30"><span id="lstnumberx30.2" style="font-size:90%;">return</span> <span id="lstnumberx30.4" style="font-size:90%;">position</span> </span></span><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2"><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1" style="font-size:90%;">This code defines a class</span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2" style="font-size:90%;">Solution</span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.3" style="font-size:90%;">with a method</span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.4" style="font-size:90%;">finalPositionOfSnake</span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.5" style="font-size:90%;">that takes the size of the matrix</span> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="n"><semantics id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.m1.1a"><mi mathsize="90%" id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.m1.1.1" xref="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.m1.1.1.cmml">n</mi> <annotation-xml encoding="MathML-Content" id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.m1.1b"><ci id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.m1.1.1.cmml" xref="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.m1.1.1">𝑛</ci></annotation-xml> <annotation encoding="application/x-tex" id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.m1.1c">n</annotation></semantics></math> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.6" style="font-size:90%;">and a list of commands, and returns the final position of the snake after executing all the commands. The movement functions (</span><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.7" style="font-size:90%;">move_right</span><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.8" style="font-size:90%;">,</span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.9" style="font-size:90%;">move_down</span><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.10" style="font-size:90%;">,</span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.11" style="font-size:90%;">move_left</span><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.12" style="font-size:90%;">,</span> <span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.13" style="font-size:90%;">move_up</span><span id="A1.F16.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.14" style="font-size:90%;">) are used to update the position of the snake based on the current direction.</span></span></span></foreignObject></g></g></svg>

Figure 16: The model displays an adeptness in solving algorithmic problems.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2-3-screenshot1.png)

The page titles in the screenshot of the search result are:

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2-4-math1.jpg)

Please tell me how many days are between Ruka’s birthday and Yuki’s birthday? Think carefully step by step.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2-5-information_extraction.jpg)

What are the Linux versions and their release dates in the picture? Return results as a JSON list.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2-6-table_weather.jpg)

Figure 20: The model displays an adeptness in OCR and following formats.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2-7-screenshot2.png)

Refer to caption

### A.3 Video Understanding

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/video_space_frames.png)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/qwen2-vl/video_part1_frames.png)

视频中的人在做什么？

### A.4 Visual Agent Capability

The Qwen2-VL also excels in location and agent tasks.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2_car_input.jpeg)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/2_news.jpeg)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/func_call_basic.jpg)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/func_call_flowchart.png)

\# Placeholder functions for the modules

![[Uncaptioned image]](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/func_call_table.png)

\[Uncaptioned image\]

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/func_call_table_result.png)

Refer to caption

<svg id="A1.SS4.p2.pic1" height="4915.07" overflow="visible" version="1.1" width="600"><g transform="translate(0,4915.07) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 4909.17 C 0 4912.43 2.64 4915.07 5.91 4915.07 L 594.09 4915.07 C 597.36 4915.07 600 4912.43 600 4909.17 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 4890.96 L 598.03 4890.96 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 4896.87)"><foreignObject width="556.69" height="12.3" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="A1.SS4.p2.pic1.11.11.11.1.1" style="width:402.3pt;"><span id="A1.SS4.p2.pic1.11.11.11.1.1.1">Function Calling - Code Interpreter</span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="4865.37" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><img src="https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/func_call_formula.png" id="A1.SS4.p2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1" width="393" height="132" alt="[Uncaptioned image]"></foreignObject></g></g></svg> 

Figure 29: The model understood the formula, implemented the code as required, and successfully executed it in the code interpreter to obtain the results. Image source: [^27]

<svg id="A1.SS4.p3.pic1" height="896.89" overflow="visible" version="1.1" width="600"><g transform="translate(0,896.89) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 890.98 C 0 894.25 2.64 896.89 5.91 896.89 L 594.09 896.89 C 597.36 896.89 600 894.25 600 890.98 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 872.93 L 598.03 872.93 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 878.84)"><foreignObject width="556.69" height="12.15" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="A1.SS4.p3.pic1.5.5.5.1.1" style="width:402.3pt;"><span id="A1.SS4.p3.pic1.5.5.5.1.1.1">VL Agent - UI Operations</span></span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="847.34" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><figure><img src="https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/agent/ui/ui_operation.png" id="A1.SS4.p3.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1" width="479" height="435" alt="[Uncaptioned image]"><figcaption>[Uncaptioned image]</figcaption></figure></foreignObject></g></g></svg><svg id="A1.SS4.p4.pic1" height="812.22" overflow="visible" version="1.1" width="600"><g transform="translate(0,812.22) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 806.31 C 0 809.57 2.64 812.22 5.91 812.22 L 594.09 812.22 C 597.36 812.22 600 809.57 600 806.31 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 788.26 L 598.03 788.26 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 794.17)"><foreignObject width="556.69" height="12.15" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="A1.SS4.p4.pic1.6.6.6.1.1" style="width:402.3pt;"><span id="A1.SS4.p4.pic1.6.6.6.1.1.1">VL Agent - UI Operations</span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="762.67" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3" style="width:402.3pt;"><span id="A1.SS4.p4.pic1.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.1.p3"><span id="A1.SS4.p4.pic1.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.1.p3.1">## Continuously take action until the task is completed.</span> <span id="A1.SS4.p4.pic1.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.1.p3.2">*FUNCTION*: The action to take, should be one of {Click,Type,Swipe,Back,Home,Enter,Done}</span> <span id="A1.SS4.p4.pic1.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.1.p3.3">*ARGS*: The input of the action.</span> <span id="A1.SS4.p4.pic1.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.1.p3.4">*RESULT*: Action results.</span> <span id="A1.SS4.p4.pic1.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.1.p3.5">*RETURN*: Reply based on action results.</span></span> <span id="A1.SS4.p4.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.2.p4"><span id="A1.SS4.p4.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.2.p4.1">Your task is: What’s a good restaurant in San Diego?</span> <span id="A1.SS4.p4.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.2.p4.2">![](Screenshot_1)</span> </span><span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5"><span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.1">The screen shows the email setup page, which is not relevant to finding a restaurant. Possible actions are to exit the em ail setup and navigate to a search engine or map application where I can look up restaurants in San Diego. *FUNCTION*: Home *ARGS*: {}</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.2">*RESULT*:![](Screenshot_2)</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.3">*RETURN*: The screen shows the home screen with various apps, including Google Chrome and Maps which could be used to search for a restaurant in San Diego. Possible actions are to tap on the Google Chrome app to open it and then use it to search for a good restaurant in San Diego.</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.4">*FUNCTION*: Click *ARGS*: {”point”: ”(687, 792)”}</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.5">*RESULT*:![](Screenshot_4)</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.6">*RETURN*: The current screen shows search results for skateboards, which is not related to the query about finding a good restaurant in San Diego. Possible actions are to click on the Google search bar at the top of the screen and enter the correct search term ”good restaurant in San Diego” to find relevant information.</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.7">*FUNCTION*: Click *ARGS*: {”point”: ” (596, 213)”}</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.8">*RESULT*:![](Screenshot_6)</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.9">*RETURN*: The current screen shows search suggestions related to skateboard speeds, which is not relevant to the query about finding a good restaurant in San Diego. Possible actions are to click on the ”X” button to he correct query for finding a good restaurant in San Diego.</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.10">*FUNCTION*: Click *ARGS*: {”point”: ”(927, 135)”}</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.11">*RESULT*:![](Screenshot_8)</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.12">*RETURN*: The screen shows the Google search bar with a list of previous searches, none of which are related to finding a restauran t in San Diego. Possible actions are to type ”good restaurant in San Diego” into the search bar and then press enter to perform the search.</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.13">*FUNCTION*: Type *ARGS*: {”text”: ”What’s a good restaurant in San Diego?”}</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.14">*RESULT*:![](Screenshot_9)</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.15">*RETURN*: The screen shows a list of search suggestions related to finding good restaurants in San Diego, indicating that the user has typed in a query but not yet executed it. Possible actions are to select one of the suggested searches or press enter on the keyboard to execute the current search and view the results.</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.16">*FUNCTION*: Enter *ARGS*: {}</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.17">*RESULT*:![](Screenshot_10)</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.18">*RETURN*: The screen shows a list of top-rated restaurants in San Diego with details such as ratings, price range, and opening hours. Possible actions are to stop and set the query as completed.</span> <span id="A1.SS4.p4.pic1.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.3.p5.19">*FUNCTION*: Done *ARGS*: {}</span></span></span></foreignObject></g></g></svg>

Figure 30: Qwen2-VL as an agent understands the query with respect to UI operation, utilizes the pre-defined actions in system message, and fulfill the task step-by-step.

<svg id="A1.SS4.p5.pic1" height="731.7" overflow="visible" version="1.1" width="600"><g transform="translate(0,731.7) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g fill="#000000" fill-opacity="1.0"><path d="M 0 5.91 L 0 725.8 C 0 729.06 2.64 731.7 5.91 731.7 L 594.09 731.7 C 597.36 731.7 600 729.06 600 725.8 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 707.59 L 598.03 707.59 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 713.5)"><foreignObject width="556.69" height="12.3" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#FFFFFF"><span id="A1.SS4.p5.pic1.9.9.9.1.1" style="width:402.3pt;"><span id="A1.SS4.p5.pic1.9.9.9.1.1.1">VL Agent - Card Game</span></span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject width="556.69" height="682" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible" color="#000000"><figure><img src="https://ar5iv.labs.arxiv.org/html/2409.12191/assets/images/agent/game/blackjack.png" id="A1.SS4.p5.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1" width="479" height="185" alt="[Uncaptioned image]"><figcaption>[Uncaptioned image]</figcaption></figure></foreignObject></g></g></svg>

Figure 31: Qwen2-VL recognizes these cards and utilizes Hit and Stand to play the blackjack.

[^1]: Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In *NeurIPS*, 2022.

[^2]: Alibaba-Cloud. Cloud parallel file storage (cpfs), 2024a. URL [https://www.alibabacloud.com/en/product/cpfs](https://www.alibabacloud.com/en/product/cpfs).

[^3]: Alibaba-Cloud. Object storage service (oss), 2024b. URL [https://www.alibabacloud.com/en/product/object-storage-service](https://www.alibabacloud.com/en/product/object-storage-service).

[^4]: Alibaba-Cloud. Pai-lingjun intelligent computing service, 2024c. URL [https://www.alibabacloud.com/en/product/pai-lingjun](https://www.alibabacloud.com/en/product/pai-lingjun).

[^5]: Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In *CVPR*, 2018.

[^6]: Jason Ansel, Edward Z. Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Shunting Zhang, Michael Suo, Phil Tillet, Xu Zhao, Eikan Wang, Keren Zhou, Richard Zou, Xiaodong Wang, Ajit Mathews, William Wen, Gregory Chanan, Peng Wu, and Soumith Chintala. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In *ASPLOS*, 2024.

[^7]: Anthropic. Claude 3.5 sonnet, 2024. URL [https://www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet).

[^8]: Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. Vivit: A video vision transformer. In *ICCV*, 2021.

[^9]: Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. *arXiv:1607.06450*, 2016.

[^10]: Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. *arXiv:2309.16609*, 2023a.

[^11]: Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. *arXiv:2308.12966*, 2023b.

[^12]: Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In *CVPR*, 2017.

[^13]: Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. *arXiv:2306.15195*, 2023a.

[^14]: Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. *arXiv:2311.12793*, 2023b.

[^15]: Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? *arXiv:2403.20330*, 2024a.

[^16]: Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, and Ivan Laptev. Think global, act local: Dual-scale graph transformer for vision-and-language navigation. In *CVPR*, 2022.

[^17]: Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. *arXiv:1604.06174*, 2016.

[^18]: Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, et al. T-eval: Evaluating the tool utilization capability step by step. *arXiv:2312.14033*, 2023c.

[^19]: Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *arXiv:2404.16821*, 2024b.

[^20]: Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. Internvl2: Better than the best—expanding performance boundaries of open-source multimodal models with the progressive scaling strategy, 2024c. URL [https://internvl.github.io/blog/2024-07-02-InternVL-2.0](https://internvl.github.io/blog/2024-07-02-InternVL-2.0).

[^21]: Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, 2023. URL [https://lmsys.org/blog/2023-03-30-vicuna/](https://lmsys.org/blog/2023-03-30-vicuna/).

[^22]: Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. *arXiv:2305.06500*, 2023.

[^23]: Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In *ICLR*, 2024.

[^24]: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In *NeurIPS*, 2022.

[^25]: Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. In *NeurIPS*, 2024.

[^26]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*, 2021.

[^27]: Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv:2407.21783*, 2024.

[^28]: Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. *arXiv:2309.17425*, 2023.

[^29]: FFmpeg-Developers. ffmpeg tool, 2024. URL [http://ffmpeg.org/](http://ffmpeg.org/).

[^30]: Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. *arXiv:2306.13394*, 2023.

[^31]: Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. *arXiv:2405.21075*, 2024.

[^32]: Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large vision-language models. *arXiv:2310.14566*, 2023.

[^33]: Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. *arXiv:2312.08914*, 2023.

[^34]: Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. *arXiv:2302.14045*, 2023a.

[^35]: Siyuan Huang, Zhengkai Jiang, Hao Dong, Yu Qiao, Peng Gao, and Hongsheng Li. Instruct2act: Mapping multi-modality instructions to robotic actions with large language model. *arXiv:2305.11176*, 2023b.

[^36]: Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Xu Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism. In *NeurIPS*, 2019.

[^37]: Yuki Inoue and Hiroki Ohashi. Prompter: Utilizing large language model prompting for a data efficient embodied instruction following. *arXiv:2211.03267*, 2022.

[^38]: Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. *arXiv:2210.03094*, 2022.

[^39]: Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In *EMNLP*, 2014.

[^40]: Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In *ECCV*, 2016.

[^41]: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In *ICCV*, 2023.

[^42]: Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. *arXiv:1712.05474*, 2017.

[^43]: Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. In *MLSys*, 2023.

[^44]: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In *NeurIPS*, 2012.

[^45]: Joel Lamy-Poirier. Breadth-first pipeline parallelism. In *MLSys*, 2023.

[^46]: Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A high-resolution multi-modality model. *arXiv:2311.04219*, 2023a.

[^47]: Chen Li, Yixiao Ge, Dian Li, and Ying Shan. Vision-language instruction tuning: A review and analysis. *arXiv:2311.08172*, 2023b.

[^48]: Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *arXiv:2301.12597*, 2023c.

[^49]: Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In *CVPR*, 2024.

[^50]: Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, et al. Pytorch distributed: Experiences on accelerating data parallel training. In *VLDB*, 2020.

[^51]: Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. *arXiv:2311.06607*, 2023d.

[^52]: Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, and Yu Jiao Qiao. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. *arXiv:2311.07575*, 2023.

[^53]: Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. *arXiv:2310.03744*, 2023a.

[^54]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *arXiv:2304.08485*, 2023b.

[^55]: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chun yue Li, Jianwei Yang, Hang Su, Jun-Juan Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. *arXiv:2303.05499*, 2023c.

[^56]: Yuan Liu, Haodong Duan, Bo Li Yuanhan Zhang, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? *arXiv:2307.06281*, 2023d.

[^57]: Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models. *arXiv:2305.07895*, 2023e.

[^58]: Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *ICLR*, 2019.

[^59]: Guanxing Lu, Ziwei Wang, Changliu Liu, Jiwen Lu, and Yansong Tang. Thinkbot: Embodied instruction following with thought chain reasoning. *arXiv:2312.07062*, 2023.

[^60]: Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In *ACL*, 2021.

[^61]: Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In *ICLR*, 2024a.

[^62]: Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. *arXiv:2406.08451*, 2024b.

[^63]: Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In *NeurIPS*, 2023.

[^64]: Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In *CVPR*, 2016.

[^65]: Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. *arXiv:2203.10244*, 2022.

[^66]: Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In *WACV*, 2021.

[^67]: Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on GPU clusters using megatron-lm. In *SC*, 2021.

[^68]: Nvidia. Apex, 2024a. URL [https://github.com/NVIDIA/apex](https://github.com/NVIDIA/apex).

[^69]: Nvidia. Cuda, 2024b. URL [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit).

[^70]: OpenAI. Gpt-4 technical report. *arXiv:2303.08774*, 2023.

[^71]: OpenAI. Gpt-4v(ision) system card, 2023. URL [https://openai.com/research/gpt-4v-system-card](https://openai.com/research/gpt-4v-system-card).

[^72]: Openai. Chatml documents, 2024. URL [https://github.com/openai/openai-python/blob/main/chatml.md](https://github.com/openai/openai-python/blob/main/chatml.md).

[^73]: OpenAI. Hello gpt-4o, 2024. URL [https://openai.com/index/hello-gpt-4o](https://openai.com/index/hello-gpt-4o).

[^74]: Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In *NeurIPS*, 2019.

[^75]: Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. In *NeurIPS*, 2024.

[^76]: Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang, William Yang Wang, Chunhua Shen, and Anton van den Hengel. Reverie: Remote embodied visual referring expression in real indoor environments. In *CVPR*, 2020.

[^77]: Alibaba Group Qwen Team. Qwen-agent framework, 2024. URL [https://github.com/QwenLM/Qwen-Agent](https://github.com/QwenLM/Qwen-Agent).

[^78]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *ICML*, 2021.

[^79]: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward training trillion parameter models. In *SC*, 2020.

[^80]: Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. *arXiv:2405.14573*, 2024a.

[^81]: Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Androidinthewild: A large-scale dataset for android device control. In *NeurIPS*, 2024b.

[^82]: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. *arXiv:2407.08608*, 2024.

[^83]: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. *arXiv:1909.08053*, 2019.

[^84]: Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In *CVPR*, 2020a.

[^85]: Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. *arXiv:2010.03768*, 2020b.

[^86]: Gunnar A Sigurdsson, Jesse Thomason, Gaurav S Sukhatme, and Robinson Piramuthu. Rrex-bot: Remote referring expressions with a bag of tricks. In *IROS*, 2023.

[^87]: Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In *CVPR*, 2019.

[^88]: Venkat Krishna Srinivasan, Zhen Dong, Banghua Zhu, Brian Yu, Damon Mosk-Aoyama, Kurt Keutzer, Jiantao Jiao, and Jian Zhang. Nexusraven: a commercially-permissive language model for function calling. In *NeurIPS Workshop*, 2023.

[^89]: Jianlin Su. Transformer upgrade path: 4. rotary position encoding for two-dimensional positions, 2021. URL [https://www.spaces.ac.cn/archives/8397](https://www.spaces.ac.cn/archives/8397).

[^90]: Jianlin Su. Transformer upgrade path: 17. insights into multimodal positional encoding, 2024. URL [https://spaces.ac.cn/archives/10040](https://spaces.ac.cn/archives/10040).

[^91]: Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. In *Neurocomputing*, 2024.

[^92]: Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri Faiz Bin Mahmood, Hao Feng, Zhen Zhao, Yanjie Wang, Yuliang Liu, Hao Liu, Xiang Bai, and Can Huang. Mtvqa: Benchmarking multilingual text-centric visual question answering. *arXiv:2405.11985*, 2024.

[^93]: Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: A family of highly capable multimodal models. *arXiv:2312.11805*, 2023.

[^94]: Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv:2302.13971*, 2023a.

[^95]: Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*, 2023b.

[^96]: Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. *arXiv:2402.14804*, 2024.

[^97]: Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In *ICML*, 2022.

[^98]: Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities. *arXiv:2305.11172*, 2023a.

[^99]: Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. *arXiv:2311.03079*, 2023b.

[^100]: X.AI. Grok-1.5 vision preview. [https://x.ai/blog/grok-1.5v](https://x.ai/blog/grok-1.5v), 2024a.

[^101]: X.AI. Grok-2 beta release. [https://x.ai/blog/grok-2](https://x.ai/blog/grok-2), 2024b.

[^102]: B. Yan, Yi Jiang, Jiannan Wu, D. Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In *CVPR*, 2023.

[^103]: Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Berkeley function calling leaderboard, 2024. URL [https://gorilla.cs.berkeley.edu/blogs/8\_berkeley\_function\_calling\_leaderboard.html](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html).

[^104]: An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. *arXiv:2407.10671*, 2024.

[^105]: Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). *arXiv:2309.17421*, 2023.

[^106]: Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. *arXiv:2408.01800*, 2024.

[^107]: Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. *arXiv:2304.14178*, 2023a.

[^108]: Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. *arXiv:2311.04257*, 2023b.

[^109]: Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. *arXiv:2404.16006*, 2024.

[^110]: Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In *ICML*, 2024.

[^111]: Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. *arXiv:2311.16502*, 2023.

[^112]: Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. *arXiv preprint arXiv:2409.02813*, 2024.

[^113]: Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. *arXiv:2405.10292*, 2024.

[^114]: Zhuosheng Zhan and Aston Zhang. You only look at screens: Multimodal chain-of-action agents. *arXiv:2309.11436*, 2023.

[^115]: Biao Zhang and Rico Sennrich. Root mean square layer normalization. In *NeurIPS*, 2019.

[^116]: Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, and Yinfei Yang. Ferret-v2: An improved baseline for referring and grounding with large language models. *arXiv:2404.07973*, 2024a.

[^117]: Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for gui agents. *arXiv:2403.02713*, 2024b.

[^118]: Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. *arXiv:2309.15112*, 2023.

[^119]: Tianyu Zhang, Suyuchen Wang, Lu Li, Ge Zhang, Perouz Taslakian, Sai Rajeswar, Jie Fu, Bang Liu, and Yoshua Bengio. Vcr: Visual caption restoration. *arXiv:2406.06462*, 2024c.

[^120]: Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. *arXiv:2304.10592*, 2023.