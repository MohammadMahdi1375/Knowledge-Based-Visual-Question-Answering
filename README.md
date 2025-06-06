# 🚀 GC-KBVQA: A New Four-Stage Framework for Enhancing Knowledge Based Visual Question Answering Performance

The code of our paper "GC-KBVQA: A New Four-Stage Framework for Enhancing Knowledge Based Visual Question Answering Performance" [PDF](https://arxiv.org/pdf/2505.19354)

## Overview
Knowledge-Based Visual Question Answering (KB-VQA) methods focus on tasks that demand reasoning with information extending beyond the explicit content depicted in the image. Early methods relied on explicit knowledge bases to provide this auxiliary information. Recent approaches leverage Large Language Models (LLMs) as implicit knowledge sources. While KB-VQA methods have demonstrated promising results, their potential remains constrained as the auxiliary text provided may not be relevant to the question context, and may also include irrelevant information that could misguide the answer predictor. We introduce a novel four-stage framework called Grounding Caption-Guided Knowledge-Based Visual Question Answering (GC-KBVQA), which enables LLMs to effectively perform zero-shot VQA tasks without the need for end-to-end multimodal training. Innovations include grounding question-aware caption generation to move beyond generic descriptions and have compact, yet detailed and context-rich information. This is combined with knowledge from external sources to create highly informative prompts for the LLM. GC-KBVQA can address a variety of VQA tasks, and does not require task-specific fine-tuning, thus reducing both costs and deployment complexity by leveraging general-purpose, pre-trained LLMs. Comparison with competing KB-VQA methods shows significantly improved performance.
<img src="./Imgs/framework.png" alt="drawing" width="800" height="400"/>

## Results
This table compares our GC-KBVQA, categorized as zero-shot evaluation without end-to-end training, with competing approaches across three VQA paradigms for SOTA models. The first is zero-shot evaluation without task-specific fine-tuning, leveraging the inherent generalization of pre-trained models. The second involves zero-shot evaluation with end-to-end fine-tuning on task-specific data to improve performance. The third is few-shot evaluation, which uses a small amount of labeled data to refine predictions by combining pre-trained knowledge with limited supervision.

<img src="./Imgs/table1.png" alt="drawing" width="600" height="600"/>

The following table presents the impact of different LLMs and overall framework sizes, on performance. This demonstrates that our GC‑KBVQA framework is truly “plug‑and‑play”: one can simply substitute Llama‑3‑8B for another model of similar scale (e.g., Mistral‑7B) using identical configurations and prompts—without any additional prompt tuning—and still achieve competitive results.

<img src="./Imgs/table2.png" alt="drawing" width="350" height="200"/>

The following table evaluates the performance of caption generation strategies, comparing the use of LLaVA and InstructBLIP individually against their combined use. The use of two caption generators mitigates the limitations of incomplete or narrow captions, enhancing generalization across datasets and question formats. Although this approach incurs slightly higher computational costs, it yields substantial accuracy gains, justifying the trade-off.

<img src="./Imgs/table3.png" alt="drawing" width="350" height="200"/>

The following table highlights the effect of varying the number of captions on GC-KBVQA performance. The use of the top three captions yielded the best results in all datasets. This configuration effectively balances reducing redundancy and maximizing contextual utility for the LLM. However, adding a fourth or fifth caption seems to increase the likelihood of irrelevant or less question-focused content, which can dilute overall information quality and impede the LLM’s reasoning process.

<img src="./Imgs/table4.png" alt="drawing" width="350" height="200"/>

The following table evaluates the impact of different prompt designs. In summary, captions lead to more accurate knowledge retrieval, QA pairs enhance reasoning patterns, and instructions align the prompt with the task objective. This cohesive design enables the model to tackle diverse and complex questions effectively, ensuring comprehensive contextual understanding and strong overall performance.

<img src="./Imgs/table5.png" alt="drawing" width="350" height="200"/>

## Qualitative results
A diverse range of samples and our proposed GC-KBVQA-generated outputs.

<img src="./Imgs/sample1.png" alt="drawing" width="800" height="300"/>
<img src="./Imgs/sample2.png" alt="drawing" width="800" height="300"/>
<img src="./Imgs/sample3.png" alt="drawing" width="800" height="300"/>
<img src="./Imgs/sample4.png" alt="drawing" width="800" height="300"/>
<img src="./Imgs/sample5.png" alt="drawing" width="800" height="300"/>
<img src="./Imgs/sample6.png" alt="drawing" width="800" height="300"/>
<img src="./Imgs/sample7.png" alt="drawing" width="800" height="200"/>
<img src="./Imgs/sample8.png" alt="drawing" width="800" height="200"/>
