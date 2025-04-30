# Optimizing Active Learning in Vision-Language Models via Parameter-Efficient Uncertainty Calibration
![GitHub License](https://img.shields.io/github/license/IntelLabs/C_PEAL?style=flat)
![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/C_PEAL/badge)
<!-- UNCOMMENT AS NEEDED
[![Unit Tests](https://github.com/IntelLabs/ConvAssist/actions/workflows/run_unittests.yaml/badge.svg?branch=covassist-cleanup)](https://github.com/IntelLabs/ConvAssist/actions/workflows/run_unittests.yaml)
[![pytorch](https://img.shields.io/badge/PyTorch-v2.4.1-green?logo=pytorch)](https://pytorch.org/get-started/locally/)
![python-support](https://img.shields.io/badge/Python-3.12-3?logo=python)
-->

This repository will host the code for the paper titled **"Optimizing Active Learning in Vision-Language Models via Parameter-Efficient Uncertainty Calibration"**

Stay tuned! The code will be released soon.

## Abstract
Active Learning (AL) has emerged as a powerful approach for minimizing labeling costs by selectively sampling the most informative data for neural network model development. Effective AL for large-scale vision-language models necessitates addressing challenges in uncertainty estimation and efficient sampling given the vast number of parameters involved. In this work, we introduce a novel parameter-efficient learning methodology that incorporates uncertainty calibration loss within the AL framework. We propose a differentiable loss function that promotes uncertainty calibration for effectively selecting fewer and most informative data samples for fine-tuning. Through extensive experiments across several datasets and vision backbones, we demonstrate that our solution can match and exceed the performance of complex feature-based sampling techniques while being computationally very efficient. Additionally, we investigate the efficacy of Prompt learning versus Low-rank adaptation (LoRA) in sample selection, providing a detailed comparative analysis of these methods in the context of efficient AL.
## Citation
If you find this work useful, please consider citing our previous works:

```
@article{narayanan2024parameter,
  title={Parameter-Efficient Active Learning for Foundational models},
  author={Narayanan, Athmanarayanan Lakshmi and Krishnan, Ranganath and Machireddy, Amrutha and Subedar, Mahesh},
  journal={arXiv preprint arXiv:2406.09296},
  year={2024}
}
```

## License
Details about the license will be provided upon release.
