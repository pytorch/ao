# benchmarking instructions:

Setup your enviornment with:
```
conda env create -n "saf-ao" python=3.10
conda activate saf-ao
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip3 install git+https://github.com/pytorch-labs/segment-anything-fast.git
pip3 install tqdm fire pandas
cd ../.. && python setup.py install
```

Then download data and models by running
```
sh setup.sh
```

Finally, you can run benchmarks with
```
sh benchmark.sh
```

You can check out the result in results.csv
