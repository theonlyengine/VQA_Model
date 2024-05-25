---
pipeline_tag: visual-question-answering
language:
- en
- zh
datasets:
- HaoyeZhang/RLAIF-V-Dataset
---


<h1>A GPT-4V Level Multimodal LLM on Your Phone</h1>

[GitHub](https://github.com/OpenBMB/MiniCPM-V) | [Demo](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5)


## News <!-- omit in toc -->

#### 📌 Pinned

* [2024.05.25] 🚀🚀🚀 MiniCPM-Llama3-V 2.5 now supports [Ollama](https://github.com/OpenBMB/ollama/tree/minicpm-v2.5/examples/minicpm-v2.5) for efficient inference. Try it now!
* [2024.05.23] 🔍 We've released a comprehensive comparison between Phi-3-vision-128k-instruct and MiniCPM-Llama3-V 2.5, including benchmarks evaluations, multilingual capabilities, and inference efficiency 🌟📊🌍🚀. Click [here](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/compare_with_phi-3_vision.md) to view more details.
* [2024.05.23] 🔥🔥🔥 MiniCPM-V tops GitHub Trending and HuggingFace Trending! Our demo, recommended by Hugging Face Gradio’s official account, is available [here](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5). Come and try it out!

<br>

* [2024.05.25] MiniCPM-Llama3-V 2.5 now supports streaming outputs and customized system prompts. Try it at [here](#usage)!
* [2024.05.24]  We release the [MiniCPM-Llama3-V 2.5 gguf](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf), which supports [llama.cpp](https://github.com/OpenBMB/MiniCPM-V/tree/main?tab=readme-ov-file#inference-with-llamacpp) inference and provides a 6~8 token/s smooth decoding on mobile phones. Try it now!
* [2024.05.20] We open-soure MiniCPM-Llama3-V 2.5, it has improved OCR capability and supports 30+ languages, representing the first end-side MLLM achieving GPT-4V level performance! We provide [efficient inference](#deployment-on-mobile-phone) and [simple fine-tuning](https://github.com/OpenBMB/MiniCPM-V/blob/main/finetune/readme.md). Try it now!


## Model Summary

**MiniCPM-Llama3-V 2.5** is the latest model in the MiniCPM-V series. The model is built on SigLip-400M and Llama3-8B-Instruct with a total of 8B parameters. It exhibits a significant performance improvement over MiniCPM-V 2.0. Notable features of MiniCPM-Llama3-V 2.5 include:

- 🔥 **Leading Performance.**
  MiniCPM-Llama3-V 2.5 has achieved an average score of 65.1 on OpenCompass, a comprehensive evaluation over 11 popular benchmarks. **With only 8B parameters, it surpasses widely used proprietary models like GPT-4V-1106, Gemini Pro, Claude 3 and Qwen-VL-Max** and greatly outperforms other Llama 3-based MLLMs.

- 💪 **Strong OCR Capabilities.**
  MiniCPM-Llama3-V 2.5 can process images with any aspect ratio and up to 1.8 million pixels (e.g., 1344x1344), achieving an **700+ score on OCRBench, surpassing proprietary models such as GPT-4o, GPT-4V-0409, Qwen-VL-Max and Gemini Pro**. Based on recent user feedback, MiniCPM-Llama3-V 2.5 has now enhanced full-text OCR extraction, table-to-markdown conversion, and other high-utility capabilities, and has further strengthened its instruction-following and complex reasoning abilities, enhancing multimodal interaction experiences.

- 🏆 **Trustworthy Behavior.**
  Leveraging the latest [RLAIF-V](https://github.com/RLHF-V/RLAIF-V/) method (the newest technology in the [RLHF-V](https://github.com/RLHF-V) [CVPR'24] series), MiniCPM-Llama3-V 2.5 exhibits more trustworthy behavior. It achieves **10.3%** hallucination rate on Object HalBench, lower than GPT-4V-1106 (13.6%), achieving the best-level performance within the open-source community.

- 🌏 **Multilingual Support.**
  Thanks to the strong multilingual capabilities of Llama 3 and the cross-lingual generalization technique from [VisCPM](https://github.com/OpenBMB/VisCPM), MiniCPM-Llama3-V 2.5 extends its bilingual (Chinese-English) multimodal capabilities to **over 30 languages including German, French, Spanish, Italian, Russian etc.** [All Supported Languages](./assets/minicpm-llama-v-2-5_languages.md).

- 🚀 **Efficient Deployment.**
  MiniCPM-Llama3-V 2.5 systematically employs **model quantization, CPU optimizations, NPU optimizations and compilation optimizations**, achieving high-efficiency deployment on edge devices. For mobile phones with Qualcomm chips, we have integrated the NPU acceleration framework QNN into llama.cpp for the first time. After systematic optimization, MiniCPM-Llama3-V 2.5 has realized a **150-fold acceleration in multimodal large model end-side image encoding** and a **3-fold increase in language decoding speed**.

### Evaluation <!-- omit in toc -->

Results on TextVQA, DocVQA, OCRBench, OpenCompass MultiModal Avg , MME, MMBench, MMMU, MathVista, LLaVA Bench, RealWorld QA, Object HalBench.

<div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/64abc4aa6cadc7aca585dddf/v2KE3wqQgM05ZW3dH2wbx.png" width="110%" />
</div>


Evaluation results of multilingual LLaVA Bench 
<div align="center">
    <img src="assets/minicpmv-llama3-v2.5/llavabench_compare.png" width="110%" />
</div>


### Examples <!-- omit in toc -->

<table align="center">
    <p align="center">
      <img src="assets/minicpmv-llama3-v2.5/cases_all.png" width=95%/>
    </p>
</table>

We deploy MiniCPM-Llama3-V 2.5 on end devices. The demo video is the raw screen recording on a Xiaomi 14 Pro without edition.

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/ticket.gif" width=40% style="display:inline-block;"/>
      <img src="assets/gif_cases/meal_plan.gif" width=40% style="display:inline-block;"/>
    </p>
</table>

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/1-4.gif" width=80%/>
    </p>
</table>



## Demo
Click here to try out the Demo of [MiniCPM-Llama3-V 2.5](https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5).

## Deployment on Mobile Phone
Coming soon.

## Usage
Inference using Huggingface transformers on NVIDIA GPUs. Requirements tested on python 3.10：
```
Pillow==10.1.0
torch==2.1.2
torchvision==0.16.2
transformers==4.40.0
sentencepiece==0.1.99
```

```python
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

image = Image.open('xx.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question}]

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.7，
    # system_prompt='' # pass system_prompt if needed
)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    stream=True
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
```

Please look at [GitHub](https://github.com/OpenBMB/MiniCPM-V) for more detail about usage.


## Inference with llama.cpp<a id="llamacpp"></a>
MiniCPM-Llama3-V 2.5 can run with llama.cpp now! See our fork of [llama.cpp](https://github.com/OpenBMB/llama.cpp/tree/minicpm-v2.5/examples/minicpmv) for more detail.


## Int4 quantized version
Download the int4 quantized version for lower GPU memory (8GB) usage:  [MiniCPM-Llama3-V-2_5-int4](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4).

## MiniCPM-V 2.0 <!-- omit in toc -->
Please see the info about MiniCPM-V 2.0 [here](https://huggingface.co/openbmb/MiniCPM-V-2).

## License
#### Model License
* The code in this repo is released according to [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)
* The usage of MiniCPM-Llama3-V 2.5's parameters is subject to ["General Model License Agreement - Source Notes - Publicity Restrictions - Commercial License"](https://github.com/OpenBMB/General-Model-License/blob/main/)
* The parameters are fully open to acedemic research
* Please contact cpm@modelbest.cn to obtain a written authorization for commercial uses. Free commercial use is also allowed after registration.

#### Statement
* As a LLM, MiniCPM-Llama3-V 2.5 generates contents by learning a large mount of texts, but it cannot comprehend, express personal opinions or make value judgement. Anything generated by MiniCPM-Llama3-V 2.5 does not represent the views and positions of the model developers
* We will not be liable for any problems arising from the use of the MinCPM-V open Source model, including but not limited to data security issues, risk of public opinion, or any risks and problems arising from the misdirection, misuse, dissemination or misuse of the model.

## Other Multimodal Projects from Our Team

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD)  | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)

## Citation

If you find our work helpful, please consider citing the following papers

```bib
@article{yu2023rlhf,
  title={Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback},
  author={Yu, Tianyu and Yao, Yuan and Zhang, Haoye and He, Taiwen and Han, Yifeng and Cui, Ganqu and Hu, Jinyi and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong and others},
  journal={arXiv preprint arXiv:2312.00849},
  year={2023}
}
@article{viscpm,
    title={Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages}, 
    author={Jinyi Hu and Yuan Yao and Chongyi Wang and Shan Wang and Yinxu Pan and Qianyu Chen and Tianyu Yu and Hanghao Wu and Yue Zhao and Haoye Zhang and Xu Han and Yankai Lin and Jiao Xue and Dahai Li and Zhiyuan Liu and Maosong Sun},
    journal={arXiv preprint arXiv:2308.12038},
    year={2023}
}
@article{xu2024llava-uhd,
  title={{LLaVA-UHD}: an LMM Perceiving Any Aspect Ratio and High-Resolution Images},
  author={Xu, Ruyi and Yao, Yuan and Guo, Zonghao and Cui, Junbo and Ni, Zanlin and Ge, Chunjiang and Chua, Tat-Seng and Liu, Zhiyuan and Huang, Gao},
  journal={arXiv preprint arXiv:2403.11703},
  year={2024}
}
```