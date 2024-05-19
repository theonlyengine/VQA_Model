---
pipeline_tag: visual-question-answering
language:
- en
- zh
datasets:
- HaoyeZhang/RLHF-V-Dataset
- Yirany/UniMM-Chat
---

[GitHub](https://github.com/OpenBMB/MiniCPM-V) | [Demo](http://120.92.209.146:8889/)


## MiniCPM-Llama3-V 2.5

**MiniCPM-Llama3-V 2.5** is the latest model in the MiniCPM-V series. The model is built on SigLip-400M and Llama3-8B-Instruct with a total of 8B parameters. It exhibits a significant performance improvement over MiniCPM-V 2.0. Notable features of MiniCPM-Llama3-V 2.5 include:

- 🔥 **Leading Performance.**
  MiniCPM-Llama3-V 2.5 has achieved an average score of 65.0 on OpenCompass, a comprehensive evaluation over 11 popular benchmarks. **It surpasses widely used proprietary models like GPT-4V-1106, Gemini Pro and Claude 3 with 8B parameters**, greatly outperforming other multimodal large models built on Llama 3.

- 💪 **Strong OCR Capabilities.**
  MiniCPM-Llama3-V 2.5 can process images with any aspect ratio up to 1.8 million pixels, achieving an **700+ score on OCRBench, surpassing proprietary models such as GPT-4o, GPT-4V-0409, QWEN-VL-Max and Gemini Pro**. Based on recent user feedback, MiniCPM-Llama3-V 2.5 has now enhanced full-text OCR extraction, table-to-markdown conversion, and other high-utility capabilities, and has further strengthened its instruction-following and complex reasoning abilities, enhancing multimodal interaction experiences.

- 🏆 **Trustworthy Behavior.**
  Leveraging the latest [RLAIF-V](https://github.com/RLHF-V/RLAIF-V/) method (the newest technology in the [RLHF-V](https://github.com/RLHF-V) [CVPR'24] series), MiniCPM-Llama3-V 2.5 exhibits trustworthy multimodal behavior. It achieves **10.3%** hallucination rate on Object HalBench, lower than GPT-4V-1106 (13.6%), achieving the best level within the open-source community.

- 🌏 **Multilingual Support.**
  Thanks to Llama 3’s robust multilingual capabilities and VisCPM's cross-lingual generalization technology, MiniCPM-Llama3-V 2.5 extends its foundational bilingual (Chinese-English) multimodal capabilities to support **30+ languages including German, French, Spanish, Italian, Russian etc.** We achieve this extension through only minimal instruction-tuning with translated multimodal data. [All Supported Languages](./assets/minicpm-llama-v-2-5_languages.md).

- 🚀 **Efficient Deployment.**
  MiniCPM-Llama3-V 2.5 systematically employs **model quantization, CPU optimizations, NPU optimizations and compilation optimizations** as acceleration techniques, achieving high-efficiency deployment on edge devices. For mobile phones with Qualcomm chips, we have integrated the NPU acceleration framework QNN into llama.cpp for the first time. After systematic optimization, MiniCPM-Llama3-V 2.5 has realized a **150-fold acceleration in multimodal large model edge-side image encoding** and a **3-fold increase in language decoding speed**.

### Evaluation <!-- omit in toc -->

<div align="center">
    <img src=assets/MiniCPM-Llama3-V-2.5-peformance.png width=66% />
</div>
<details>
<summary>Click to view results on TextVQA, DocVQA, OCRBench, OpenCompass, MME, MMBench, MMMU, MathVista, LLaVA Bench, RealWorld QA, Object HalBench. </summary>
<div align="center">

<table style="margin: 0px auto;">
    <caption>Results on OCR-specific and General Multimodal Benchmarks</caption>
    <thead>
        <tr>
            <th align="left">Model</th>
            <th>Size</th>
            <th>OCRBench</th>
            <th>TextVQA val</th>
            <th>DocVQA test</th>
            <th>Open-Compass</th>
            <th>MME</th>
            <th>MMB dev (en)</th>
            <th>MMB dev (zh)</th>
            <th>MMMU val</th>
            <th>Math-Vista</th>
            <th>LLaVA Bench</th>
            <th>RealWorld QA</th>
            <th>Object HalBench</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td colspan="14" align="left"><strong>Proprietary</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Gemini Pro</td>
            <td>-</td>
            <td>680</td>
            <td>74.6</td>
            <td>88.1</td>
            <td>63.8</td>
            <td>2148.9</td>
            <td>75.2</td>
            <td>74.0</td>
            <td>48.9</td>
            <td>45.8</td>
            <td>79.9</td>
            <td>60.4</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">GPT-4V (2023.11.06)</td>
            <td>-</td>
            <td>645</td>
            <td>78.0</td>
            <td>88.4</td>
            <td>63.2</td>
            <td>1771.5</td>
            <td>75.1</td>
            <td>75.0</td>
            <td>53.8</td>
            <td>47.8</td>
            <td>93.1</td>
            <td>63.0</td>
            <td>86.4</td>
        </tr>
        <tr>
            <td colspan="14" align="left"><strong>Open-source</strong></td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">DeepSeek-VL-1.3B</td>
            <td>1.7B</td>
            <td>413</td>
            <td>58.4*</td>
            <td>37.9*</td>
            <td>46.0</td>
            <td>1531.6</td>
            <td>64.0</td>
            <td>61.2</td>
            <td>33.8</td>
            <td>29.4</td>
            <td>51.1</td>
            <td>49.7</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Mini-Gemini</td>
            <td>2.2B</td>
            <td>-</td>
            <td>56.2</td>
            <td>34.2*</td>
            <td>-</td>
            <td>1653.0</td>
            <td>59.8</td>
            <td>-</td>
            <td>31.7</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Yi-VL-6B</td>
            <td>6.7B</td>
            <td>290</td>
            <td>45.5*</td>
            <td>17.1*</td>
            <td>49.3</td>
            <td>1915.1</td>
            <td>68.6</td>
            <td>68.3</td>
            <td>40.3</td>
            <td>28.8</td>
            <td>51.9</td>
            <td>53.5</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Qwen-VL-Chat</td>
            <td>9.6B</td>
            <td>488</td>
            <td>61.5</td>
            <td>62.6</td>
            <td>52.1</td>
            <td>1860.0</td>
            <td>60.6</td>
            <td>56.7</td>
            <td>37.0</td>
            <td>33.8</td>
            <td>67.7</td>
            <td>49.3</td>
            <td>56.2 / 80.0</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">DeepSeek-VL-7B</td>
            <td>7.3B</td>
            <td>435</td>
            <td>64.7*</td>
            <td>47.0*</td>
            <td>55.6</td>
            <td>1765.4</td>
            <td>74.1</td>
            <td>72.8</td>
            <td>38.3</td>
            <td>36.8</td>
            <td>77.8</td>
            <td>54.2</td>
            <td><strong>91.5 / 95.3</strong></td>
        </tr>        
        <tr>
            <td nowrap="nowrap" align="left">Yi-VL-34B</td>
            <td>34B</td>
            <td>290</td>
            <td>43.4*</td>
            <td>16.9*</td>
            <td>52.6</td>
            <td>2050.2</td>
            <td>71.1</td>
            <td>71.4</td>
            <td>45.1</td>
            <td>30.7</td>
            <td>62.3</td>
            <td>54.8</td>
            <td>79.3 / 86.0</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">CogVLM-Chat</td>
            <td>17.4B</td>
            <td>590</td>
            <td>70.4</td>
            <td>33.3*</td>
            <td>52.5</td>
            <td>1736.6</td>
            <td>63.7</td>
            <td>53.8</td>
            <td>37.3</td>
            <td>34.7</td>
            <td>73.9</td>
            <td>60.3</td>
            <td>73.6 / 87.4</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">TextMonkey</td>
            <td>9.7B</td>
            <td>558</td>
            <td>64.3</td>
            <td>66.7</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
          <td nowrap="nowrap" align="left">IDEFICS2-8B</td>
          <td>8.0B</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>57.2</td>
          <td>1847.6</td>
          <td>75.7</td>
          <td>68.6</td>
          <td>45.2</td>
          <td>52.2</td>
          <td>49.1</td>
          <td>60.7</td>
          <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">Bunny-LLama-3-8B</td>
            <td>8.4B</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>54.3</td>
            <td>1920.3</td>
            <td>77.0</td>
            <td><strong>73.9</strong></td>
            <td>41.3</td>
            <td>31.5</td>
            <td>61.2</td>
            <td>58.8</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">XTuner-Llama-3-8B-v1.1</td>
            <td>8.4B</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>53.3</td>
            <td>1818.0</td>
            <td>71.7</td>
            <td>63.2</td>
            <td>39.2</td>
            <td>40.0</td>
            <td>69.2</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td nowrap="nowrap" align="left">LLaVA-NeXT Llama-3-8B</td>
            <td>8.4B</td>
            <td>-</td>
            <td>-</td>
            <td>78.2</td>
            <td>-</td>
            <td>1971.5</td>
            <td>72.1</td>
            <td>-</td>
            <td>41.7</td>
            <td>37.5</td>
            <td>80.1</td>
            <td>60.0</td>
            <td>-</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 1.0</td>
            <td>2.8B</td>
            <td>366</td>
            <td>60.6</td>
            <td>38.2</td>
            <td>47.6</td>
            <td>1650.2</td>
            <td>67.9</td>
            <td>65.3</td>
            <td>38.3</td>
            <td>28.9</td>
            <td>51.3</td>
            <td>51.2</td>
            <td>78.4 / 88.5</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-V 2.0</td>
            <td>2.8B</td>
            <td>605</td>
            <td>74.1</td>
            <td>71.9</td>
            <td>55.0</td>
            <td>1808.6</td>
            <td>69.6</td>
            <td>68.1</td>
            <td>38.2</td>
            <td>38.7</td>
            <td>69.2</td>
            <td>55.8</td>
            <td>85.5 / 92.2</td>
        </tr>
        <tr style="background-color: #e6f2ff;">
            <td nowrap="nowrap" align="left">MiniCPM-Llama3-V 2.5</td>
            <td>8.5B</td>
            <td><strong>725</strong></td>
            <td><strong>76.6</strong></td>
            <td><strong>84.8</strong></td>
            <td><strong>65.0</strong></td>
            <td><strong>2024.6</strong></td>
            <td><strong>76.7</strong></td>
            <td><strong>73.4</strong></td>
            <td><strong>45.8</strong></td>
            <td><strong>54.3</strong></td>
            <td><strong>86.7</strong></td>
            <td><strong>63.5</strong></td>
            <td>89.7 / 95.0</td>
        </tr>
    </tbody>
</table>


</div>
* We evaluate the officially released checkpoint by ourselves.

</details>

### Examples <!-- omit in toc -->

<table align="center">
    <p align="center">
      <img src="assets/minicpmv-llama3-v2.5/case_markdown.png" width=95%/>
    </p>
    <p align="center">
      <img src="assets/minicpmv-llama3-v2.5/case_OCR_en.png" width=95%/>
    </p>
    <p align="center">
      <img src="assets/minicpmv-llama3-v2.5/case_long_img.png" width=95%/>
    </p>
    <p align="center">
      <img src="assets/minicpmv-llama3-v2.5/case_complex_reasoning.png" width=95%/>
    </p>
</table>

We deploy MiniCPM-V 2.0 on end devices. The demo video is the raw screen recording on a Xiaomi 14 Pro at double speed.

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/ticket.gif" width=40%/>
      <img src="assets/gif_cases/meal_plan.gif" width=40%/>
    </p>
</table>




## Demo
Click here to try out the Demo of [MiniCPM-Llama3-V 2.5](http://120.92.209.146:8889).

## Deployment on Mobile Phone
MiniCPM-Llama3-V 2.5 can be deployed on mobile phones with Android operating systems. 🚀 Try it out [here](https://github.com/OpenBMB/mlc-MiniCPM).


## Usage
Inference using Huggingface transformers on Nivdia GPUs. Requirements tested on python 3.10：
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
    sampling=True,
    temperature=0.7
)
print(res)
```

Please look at [GitHub](https://github.com/OpenBMB/MiniCPM-V) for more detail about usage.


## MiniCPM-V 2.0 <!-- omit in toc -->
Please see the info about MiniCPM-V 2.0 [here](https://huggingface.co/openbmb/MiniCPM-V).

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

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD)

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