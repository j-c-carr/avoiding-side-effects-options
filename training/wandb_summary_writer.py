import wandb
from typing import Optional


class WandbSummaryWriter:
    """Class for logging to wandb during training. It is designed to follow the
    same style as the Tensorboard SummaryWriter, since this code base uses that
    for logging the training statistics"""
    def __init__(self, project: str, name: str,
                 group: Optional[str] = None, log_dir: Optional[str] = None):
        self.project = project
        self.name = name
        self.group = group
        self.log_dir = log_dir

        self.run = wandb.init(project=self.project,
                              name=self.name,
                              group=self.group,
                              dir=self.log_dir)

    def add_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Wrapper for wandb.log"""
        self.run.log({name: value}, step=step)

    def flush(self):
        """Needed to match the Tensorboard SummaryWriter"""
        pass

