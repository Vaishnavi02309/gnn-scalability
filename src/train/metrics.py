import time
from dataclasses import dataclass

@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_acc: float
    epoch_time_s: float
