# benchmarking instructions:

```
conda env create -n "saf-ao" python=3.10
conda activate saf-ao
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
pip3 install git+https://github.com/pytorch-labs/segment-anything-fast.git
pip3 install tqdm fire pandas
cd ../.. && python setup.py install
```

sh setup.sh
sh benchmark_sam.sh
