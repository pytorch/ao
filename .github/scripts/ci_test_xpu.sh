cd ao 
python3 -m pip install torch torchvision torchaudio pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir 
python3 setup.py install

pip install pytest expecttest parameterized accelerate hf_transfer 'modelscope!=1.15.0'

cd test/quantization
pytest -v -s *.py