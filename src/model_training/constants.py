from dataclasses import dataclass


dataclass(frozen=True)
class Constants:
    epochs: int = 1
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 2e-05
    warmup_ratio: float = 0.1
    eval_steps: int = 100
