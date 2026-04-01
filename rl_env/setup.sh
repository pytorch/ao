cd /workspace/ao
USE_CPP=0 pip install -e . --no-build-isolation
pip install -r dev-requirements.txt
pip install -r rl_env/rl-requirements.txt
