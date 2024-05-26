import os

AUTOTUNER_ENABLE = os.environ.get("TORCHAO_AUTOTUNER_ENABLE") == "1"

# Enable cache with a defaul location
current_file_dir = os.path.dirname(os.path.abspath(__file__))
TORCHAO_AUTOTUNER_DATA_PATH = os.path.join(current_file_dir, "configs", "data_a100.pkl")
