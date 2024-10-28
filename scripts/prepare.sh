python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf
python scripts/download.py --repo_id meta-llama/Meta-Llama-3-8B
python scripts/download.py --repo_id meta-llama/Meta-Llama-3.1-8B
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3.1-8B
