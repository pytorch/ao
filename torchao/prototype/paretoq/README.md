# ParetoQ


This repository contains the training code of ParetoQ introduced in our work: "[ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization](https://arxiv.org/abs/2502.02631)"

In this work, we present ParetoQ, the first unified framework that facilitates rigorous comparisons across 1-bit, 1.58-bit, 2-bit, 3-bit, and 4-bit quantization settings. By optimizing training schemes and refining quantization functions, ParetoQ surpasses all previous methods tailored to specific bit widths.  Specifically, the 1.58-bit ParetoQ LLaMA-3 8B model reduces the performance gap to full precision by relatively 37.8% compared to the 1-bit Era’s 1.58-bit LLaMA-3 8B model, while using only 30% of the training tokens.

<div align=center>
<img width=50% src="./main_result_ternary.jpg"/>
</div>

<div align=center>
<img width=100% src="./main_result_234bit.jpg"/>
</div>

With the SoTA points obtained through ParetoQ, we are able to improve the scaling law analysis. Figure (a) (b) demonstrates that sub-4-bit quantization, including binary, ternary, 2-bit, and 3-bit, often outperform 4-bit quantization. Notably, 2-bit and ternary models reside on the Pareto frontier. When considering hardware-friendliness and real-time speed, we generally recommend exploring 2-bit quantization for on-device applications.

<div align=center>
<img width=100% src="./main_result_scaling_law.jpg"/>
</div>
## Citation

If you find our code useful for your research, please consider citing:
    
    @article{liu2025paretoq,
      title={ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization},
      author={Liu, Zechun and Zhao, Changsheng and Huang, Hanxian and Chen, Sijia and Zhang, Jing and Zhao, Jiawei and Roy, Scott and Jin, Lisa and Xiong, Yunyang and Shi, Yangyang and others},
      journal={arXiv preprint arXiv:2502.02631},
      year={2025}
    }
    
## Run

### 1. Requirements:
* python 3.11
* pip3 install torch
* pip install -r requirement.txt
   
### 2. Steps to run:
* Specify the data path and the pre-trained full-precision model path in run_train.sh file
* Run `bash 1_run_train.sh $w_bit` E.g. `bash 1_run_train.sh 2` for 2-bit weight quantization.

## Comparison to SoTA Ternary LLM methods
The results reported in the paper is run with the internal LLaMA codebase in Meta. We reproduced our experiments with huggingface codebase and released code here. The results are close to those in the paper. 

 | Method | #Params | Arc-e | Arc-c | Boolq | Piqa | Siqa | HellaSwag | Obqa | WinoGrande | Avg. | Wiki |
 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
 | RTN | 600M | 26.2 | 24.6 | 62.2 | 49.5 | 36.3 | 26.1 | 27.1 | 48.8 | 37.6 | 6.60E+05 | 
 | LLM-QAT | 600M | 34.0 | 23.0 | 59.4 | 53.6 | 38.9 | 28.7 | 32.3 | 51.4 | 40.2 | 71.7 | 
 | 1-bit era | 700M | 49.5 | 29.0 | 59.2 | 67.5 | 43.6 | 43.2 | 38.9 | 53.5 | 48.1 | 17.3 | 
 | Spectra | 560M | 50.2 | 21.0 | 57.3 | 67.5 | -- | 33.8 | -- | 53.1 |  | -- | 
 | **ParetoQ** | **600M** | **65.5** | **43.8** | **62.3** | **70.6** | **44.7** | **51.3** | **47.1** | **58.8** | **55.5** | **11.4** |
 | RTN | 1B | 25.7 | 24.8 | 37.8 | 49.3 | 37.1 | 26.2 | 25.2 | 50.2 | 34.5 | 1.40E+05 | 
 | LLM-QAT | 1B | 36.0 | 26.2 | 47.7 | 55.1 | 39.7 | 31.3 | 33.5 | 49.6 | 39.9 | 56.9 | 
 | 1-bit era | 1.3B | 52.4 | 34.1 | 61.9 | 69.1 | 44.7 | 47.4 | 41.1 | 55.3 | 50.8 | 23.6 | 
 | Spectra | 1.1B | 56.3 | 24.6 | 59.1 | 69.3 | -- | 38.8 | -- | 55.5 |  | -- | 
 | **ParetoQ** | **1B** | **68.5** | **47.6** | **62.8** | **72.1** | **45.3** | **57.4** | **52.9** | **61.3** | **58.5** | **10.0** | 
 | RTN | 3B | 26.9 | 23.6 | 62.2 | 51.3 | 37.6 | 26.4 | 27.0 | 49.3 | 38.0 | 4.40E+05 | 
 | LLM-QAT | 3B | 44.5 | 30.7 | 62.1 | 62.7 | 41.0 | 43.4 | 35.0 | 50.6 | 46.3 | 6.50E+02 | 
 | 1-bit era | 3B | 58.7 | 37.2 | 61.3 | 71.3 | 45.2 | 56.0 | 45.8 | 60.3 | 54.5 | 265.6 | 
 | Spectra | 3.9B | 66.0 | 31.9 | 66.5 | 74.4 | -- | 48.3 | -- | 62.1 |  | -- | 
 | **ParetoQ**  | **3B** | **71.5** | **48.6** | **68.2** | **75.5** | **46.4** | **67.9** | **54.3** | **63.1** | **61.9** | **9.9** |

 More results for other bit widths can be found in the [paper](https://arxiv.org/abs/2502.02631).

## Acknowledgement

This code is partially based on HuggingFace [Transformers](https://github.com/huggingface/transformers) repo under [Apache License](https://github.com/huggingface/transformers/blob/main/LICENSE).

## Contact

Zechun Liu, Reality Labs, Meta Inc (zechunliu at meta dot com)

Changsheng Zhao, Reality Labs, Meta Inc (cszhao at meta dot com)

## License

ParetoQ is released under the [BSD 3](https://github.com/facebookresearch/ParetoQ/blob/main/LICENSE) license.

