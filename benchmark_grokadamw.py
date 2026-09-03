#!/usr/bin/env python3
"""
Benchmark GrokAdamW vs Adam optimizer on a language model.
Tracks: loss curve, wall-clock time per step, final loss.
Generates a markdown comparison table.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import GrokAdamW from torchao
try:
    from torchao.optim import GrokAdamW8bit
    grokadamw_available = True
except ImportError as e:
    print(f"Error importing GrokAdamW: {e}")
    grokadamw_available = False


def create_simple_lm_model(vocab_size=1000, hidden_size=256):
    """Create a simple transformer-based language model for testing."""
    return nn.Sequential(
        nn.Embedding(vocab_size, hidden_size),
        nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=512,
            batch_first=True,
        ),
        nn.Linear(hidden_size, vocab_size),
    )


def create_toy_dataset(num_samples=50, seq_length=32, vocab_size=1000):
    """Create a simple toy dataset for testing."""
    # Random token IDs
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))

    return input_ids, labels


def benchmark_optimizer(model, optimizer_name, optimizer_class, device, num_steps=100):
    """
    Run a training loop with the given optimizer.
    Returns: final_loss, wall_clock_time (seconds), avg_time_per_step, losses
    """
    # Create toy dataset
    input_ids, labels = create_toy_dataset(num_samples=50, seq_length=32, vocab_size=1000)
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize optimizer
    if optimizer_name == "GrokAdamW8bit":
        # GrokAdamW requires block_size parameter
        optimizer = optimizer_class(
            model.parameters(),
            lr=1e-3,
            block_size=256,  # Required for quantized optimizers
            weight_decay=0.01,
        )
    else:
        # Standard PyTorch Adam
        optimizer = optimizer_class(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )

    loss_fn = nn.CrossEntropyLoss()
    model.train()
    losses = []
    step_times = []
    step_count = 0

    start_time = time.time()

    for input_ids, labels in dataloader:
        if step_count >= num_steps:
            break

        step_start = time.time()

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(input_ids)
        loss = loss_fn(logits.reshape(-1, 1000), labels.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (common practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()

        step_time = time.time() - step_start
        step_times.append(step_time)
        losses.append(loss.item())

        step_count += 1

        if (step_count % 25) == 0:
            print(f"  [{optimizer_name}] Step {step_count}/{num_steps}, Loss: {loss.item():.4f}, Time/step: {step_time:.4f}s")

    total_time = time.time() - start_time
    final_loss = losses[-1] if losses else float('inf')
    avg_time_per_step = sum(step_times) / len(step_times) if step_times else 0

    return final_loss, total_time, avg_time_per_step, losses


def main():
    device = "cpu"  # Use CPU for simplicity
    print(f"Using device: {device}")
    print()

    # Create simple model for benchmarking
    print("Creating simple language model...")
    model = create_simple_lm_model(vocab_size=1000, hidden_size=256).to(device)
    model_param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created. Total parameters: {model_param_count:,}")
    print()

    # Number of training steps
    num_steps = 100
    print(f"Running {num_steps} training steps per optimizer...\n")

    results = {}

    # Benchmark 1: PyTorch Adam (baseline)
    print("=" * 60)
    print("Benchmark 1: torch.optim.Adam (baseline)")
    print("=" * 60)
    adam_loss, adam_time, adam_per_step, adam_losses = benchmark_optimizer(
        model, "Adam", torch.optim.Adam, device, num_steps=num_steps
    )
    results["Adam"] = {
        "final_loss": adam_loss,
        "total_time": adam_time,
        "avg_time_per_step": adam_per_step,
        "losses": adam_losses,
    }
    print(f"\nAdam Summary:")
    print(f"  Final Loss: {adam_loss:.4f}")
    print(f"  Total Time: {adam_time:.2f}s")
    print(f"  Avg Time/Step: {adam_per_step:.4f}s")
    print()

    # Benchmark 2: GrokAdamW8bit (if available)
    if grokadamw_available:
        print("=" * 60)
        print("Benchmark 2: GrokAdamW8bit")
        print("=" * 60)
        try:
            grok_loss, grok_time, grok_per_step, grok_losses = benchmark_optimizer(
                model, "GrokAdamW8bit", GrokAdamW8bit, device, num_steps=num_steps
            )
            results["GrokAdamW8bit"] = {
                "final_loss": grok_loss,
                "total_time": grok_time,
                "avg_time_per_step": grok_per_step,
                "losses": grok_losses,
            }
            print(f"\nGrokAdamW8bit Summary:")
            print(f"  Final Loss: {grok_loss:.4f}")
            print(f"  Total Time: {grok_time:.2f}s")
            print(f"  Avg Time/Step: {grok_per_step:.4f}s")
            print()
        except Exception as e:
            print(f"Error running GrokAdamW8bit: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("GrokAdamW not available, skipping GrokAdamW benchmark")
        print()

    # Generate comparison table
    print("=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print()

    markdown_table = generate_markdown_table(results)
    print(markdown_table)
    print()

    # Save results to file
    output_file = "benchmark_results.md"
    with open(output_file, "w") as f:
        f.write("# GrokAdamW Benchmark Results\n\n")
        f.write(f"**Setup:**\n")
        f.write(f"- Model: Simple Transformer ({model_param_count:,} parameters)\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Training Steps: {num_steps}\n")
        f.write(f"- Batch Size: 4\n")
        f.write(f"- Sequence Length: 32\n")
        f.write(f"- Vocabulary Size: 1000\n\n")
        f.write(markdown_table)

    print(f"Results saved to {output_file}")


def generate_markdown_table(results):
    """Generate a markdown comparison table from results."""
    lines = []
    lines.append("| Optimizer | Final Loss | Total Time (s) | Time/Step (ms) | Speedup vs Adam |")
    lines.append("|-----------|-----------|----------------|----------------|-----------------|")

    adam_time_per_step = results["Adam"]["avg_time_per_step"] if "Adam" in results else 1.0

    for opt_name, metrics in sorted(results.items()):
        final_loss = metrics["final_loss"]
        total_time = metrics["total_time"]
        time_per_step = metrics["avg_time_per_step"]

        speedup = adam_time_per_step / time_per_step if time_per_step > 0 else 1.0
        speedup_str = f"{speedup:.2f}x" if opt_name != "Adam" else "-"

        lines.append(
            f"| {opt_name} | {final_loss:.4f} | {total_time:.2f} | {time_per_step*1000:.2f} | {speedup_str} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
