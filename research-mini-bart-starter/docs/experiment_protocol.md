# Experiment Protocol

For every run:
1. Assign a unique experiment ID.
2. Record the Git commit hash and random seed.
3. Save the full config as JSON/YAML with the run.
4. Record dataset name, version, split, and sample counts.
5. Save train/validation curves and evaluation metrics.
6. Save checkpoints to Google Drive, not Git.
7. Record observations and limitations; do not infer causality from one run.
