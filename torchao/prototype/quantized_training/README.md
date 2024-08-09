# Quantized training

This folder contains experimental work on quantized training (QT). The main difference from quantization-aware training (QAT) is that in QT, we don't keep a high-precision copy of model weights. We take inspirations from:
- Q-GaLore: [[paper](https://arxiv.org/abs/2407.08296)] [[code](https://github.com/VITA-Group/Q-GaLore)]
- AQT: [[related paper](https://arxiv.org/abs/2105.03536)] [[code](https://github.com/google/aqt)]

Currently we only support weight-only channel-wise INT8 symmetric quantization.
