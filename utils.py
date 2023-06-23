import random

import numpy as np
import omegaconf
import torch
import wandb


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def setup_wandb(cfg):
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        entity=cfg.user.wandb_id,
        project=cfg.wandb.project,
        settings=wandb.Settings(start_method="thread"),
        name=cfg.wandb.exp_name,
        #         reinit=True,
    )
    wandb.config.update(cfg_dict, allow_val_change=True)
