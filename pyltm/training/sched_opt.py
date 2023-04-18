from typing import Iterator, List, Union
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler


class ScheduledOptimizer:
    def __init__(self, parameters: Union[Iterator[nn.Parameter], List[dict]],
                 lr: float, num_training_steps: int, num_warmup_steps: int = 0,
                 weight_decay: float = 1e-3):
        super().__init__()
        self.opt = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        self.scheduler = self.get_opt_scheduler(self.opt, num_training_steps, num_warmup_steps=num_warmup_steps)

    @staticmethod
    def get_opt_scheduler(optimizer: optim.Optimizer, num_training_steps: int, num_warmup_steps=0):
        return get_scheduler(
            'linear',
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def zero_grad(self, set_to_none: bool = False):
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.opt.step()
        self.scheduler.step()

    @staticmethod
    def _get_last_lr(scheduler:  LambdaLR):
        last_lr_list = scheduler.get_last_lr()
        return last_lr_list[0] if len(last_lr_list) > 0 else 0

    def get_last_lr(self):
        return self._get_last_lr(self.scheduler)

    def get_last_lrs(self) -> List[float]:
        return self.scheduler.get_last_lr()
