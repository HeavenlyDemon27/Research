import torch
from torch import nn

from src.train import create_optimizer_and_scheduler


# Small dummy model (we don't need to load BART for this test)
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.LayerNorm(20),
    nn.Linear(20, 5),
)

total_training_steps = 100

optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    total_training_steps,
)

print("--- Optimizer Verification ---")
print("Optimizer:", type(optimizer).__name__)
print("Parameter groups:", len(optimizer.param_groups))

for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i} weight decay:", group["weight_decay"])
    print(f"Group {i} parameter tensors:", len(group["params"]))

print("\n--- Scheduler Verification ---")
print("Scheduler:", type(scheduler).__name__)
print("Warmup steps:", int(total_training_steps * 0.06))
print("Total training steps:", total_training_steps)

assert len(optimizer.param_groups) == 2
assert optimizer.param_groups[0]["weight_decay"] == 0.01
assert optimizer.param_groups[1]["weight_decay"] == 0.0

print("\nOptimizer and scheduler verification passed!")