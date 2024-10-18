python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf
python scripts/download.py --repo_id meta-llama/Meta-Llama-3-8B
python scripts/download.py --repo_id meta-llama/Meta-Llama-3.1-8B
python scripts/download.py --repo_id neuralmagic/SparseLlama-3-8B-pruned_50.2of4

# need to copy over tokenizer file for SparseLLama checkpoint.
mkdir -p checkpoints/neuralmagic/SparseLlama-3-8B-pruned_50.2of4/original
cp checkpoints/meta-llama/Meta-Llama-3-8B/original/tokenizer.model checkpoints/neuralmagic/SparseLlama-3-8B-pruned_50.2of4/original/tokenizer.model

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3.1-8B
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/neuralmagic/SparseLlama-3-8B-pruned_50.2of4
