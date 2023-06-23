import csv
import itertools
import os
import time
from collections import defaultdict
import copy

from pathlib import Path
from datetime import date
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
import wandb
import pandas as pd
from PIL import Image

import utils
from dataset import get_loaders


@torch.no_grad()
def test(model, loader, criterion, cfg):
    model.eval()
    all_test_corrects = []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y)
        all_test_corrects.append(torch.argmax(logits, dim=-1) == y)
        total_loss += loss
    acc = torch.cat(all_test_corrects).float().mean().detach().item()
    total_loss = total_loss / len(loader)
    total_loss = total_loss.detach().item()
    return acc, total_loss

def get_lr_weights(model, loader, cfg):
    layer_names = [
        n for n, _ in model.named_parameters() if "bn" not in n
    ] 
    metrics = defaultdict(list)
    average_metrics = defaultdict(float)
    partial_loader = itertools.islice(loader, 5)
    xent_grads, entropy_grads = [], []
    for x, y in partial_loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)

        loss_xent = F.cross_entropy(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=model.parameters(), retain_graph=True
        )
        xent_grads.append([g.detach() for g in grad_xent])

    def get_grad_norms(model, grads, cfg):
        _metrics = defaultdict(list)
        grad_norms, rel_grad_norms = [], []
        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names:
                continue
            if cfg.args.auto_tune == 'eb-criterion':
                tmp = (grad*grad) / (torch.var(grad, dim=0, keepdim=True)+1e-8)
                _metrics[name] = tmp.mean().item()
            else:
                _metrics[name] = torch.norm(grad).item() / torch.norm(param).item()

        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(model, xent_grad, cfg)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)
    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)
    return average_metrics

def train(model, loader, criterion, opt, cfg, orig_model=None):
    all_train_corrects = []
    total_loss = 0.0
    magnitudes = defaultdict(float)

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y)
        all_train_corrects.append(torch.argmax(logits, dim=-1) == y)
        total_loss += loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    acc = torch.cat(all_train_corrects).float().mean().detach().item()
    total_loss = total_loss / len(loader)
    total_loss = total_loss.detach().item()
    return acc, total_loss, magnitudes


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    cfg.args.log_dir = Path.cwd()
    cfg.args.log_dir = os.path.join(
        cfg.args.log_dir, "results", cfg.data.dataset_name, date.today().strftime("%Y.%m.%d"), cfg.args.auto_tune
    )
    print(f"Log dir: {cfg.args.log_dir}")
    os.makedirs(cfg.args.log_dir, exist_ok=True)

    tune_options = [
        "first_two_block",
        "second_block",
        "third_block",
        "last",
        "all",
    ]
    if cfg.data.dataset_name == "imagenet-c":
        tune_options.append("fourth_block")
    if cfg.args.auto_tune != 'none':
        tune_options = ["all"]
    if cfg.args.epochs == 0: tune_options = ['all']
    corruption_types = cfg.data.corruption_types
    for corruption_type in corruption_types:
        cfg.wandb.exp_name = f"{cfg.data.dataset_name}_corruption{corruption_type}"
        if cfg.wandb.use:
            utils.setup_wandb(cfg)
        utils.set_seed_everywhere(cfg.args.seed)
        loaders = get_loaders(cfg, corruption_type, cfg.data.severity)

        for tune_option in tune_options:
            tune_metrics = defaultdict(list)
            lr_wd_grid = [
                (1e-1, 1e-4),
                (1e-2, 1e-4),
                (1e-3, 1e-4),
                (1e-4, 1e-4),
                (1e-5, 1e-4),
            ]
            for lr, wd in lr_wd_grid:
                dataset_name = (
                    "imagenet"
                    if cfg.data.dataset_name == "imagenet-c"
                    else cfg.data.dataset_name
                )
                model = load_model(
                    cfg.data.model_name,
                    cfg.user.ckpt_dir,
                    dataset_name,
                    ThreatModel.corruptions,
                )

                orig_model = copy.deepcopy(model)
                model = model.cuda()

                if cfg.data.dataset_name == "cifar10":
                    tune_params_dict = {
                        "all": [model.parameters()],
                        "first_two_block": [
                            model.conv1.parameters(),
                            model.block1.parameters(),
                        ],
                        "second_block": [
                            model.block2.parameters(),
                        ],
                        "third_block": [
                            model.block3.parameters(),
                        ],
                        "last": [model.fc.parameters()],
                    }
                elif cfg.data.dataset_name == "imagenet-c":
                    tune_params_dict = {
                        "all": [model.model.parameters()],
                        "first_second": [
                            model.model.conv1.parameters(),
                            model.model.layer1.parameters(),
                            model.model.layer2.parameters(),
                        ],
                        "first_two_block": [
                            model.model.conv1.parameters(),
                            model.model.layer1.parameters(),
                        ],
                        "second_block": [
                            model.model.layer2.parameters(),
                        ],
                        "third_block": [
                            model.model.layer3.parameters(),
                        ],
                        "fourth_block": [
                            model.model.layer4.parameters(),
                        ],
                        "last": [model.model.fc.parameters()],
                    }

                params_list = list(itertools.chain(*tune_params_dict[tune_option]))
                
                opt = optim.Adam(params_list, lr=lr, weight_decay=wd)
                N = sum(p.numel() for p in params_list if p.requires_grad)

                print(
                    f"\nTrain mode={cfg.args.train_mode}, using {cfg.args.train_n} corrupted images for training"
                )
                print(
                    f"Re-training {tune_option} ({N} params). lr={lr}, wd={wd}. Corruption {corruption_type}"
                )

                criterion = F.cross_entropy
                layer_weights = [0 for layer, _ in model.named_parameters() if 'bn' not in layer]
                layer_names = [layer for layer, _ in model.named_parameters() if 'bn' not in layer]
                for epoch in range(1, cfg.args.epochs + 1):
                    if cfg.args.train_mode == "train":
                        model.train()
                    if cfg.args.auto_tune != 'none': 
                        if cfg.args.auto_tune == 'RGN':
                            weights = get_lr_weights(model, loaders["train"], cfg)
                            max_weight = max(weights.values())
                            for k, v in weights.items(): 
                                weights[k] = v / max_weight
                            layer_weights = [sum(x) for x in zip(layer_weights, weights.values())]
                            tune_metrics['layer_weights'] = layer_weights
                            params = defaultdict()
                            for n, p in model.named_parameters():
                                if "bn" not in n:
                                    params[n] = p 
                            params_weights = []
                            for param, weight in weights.items():
                                params_weights.append({"params": params[param], "lr": weight*lr})
                            opt = optim.Adam(params_weights, lr=lr, weight_decay=wd)
                        elif cfg.args.auto_tune == 'eb-criterion':
                            # Go by individual layers
                            weights = get_lr_weights(model, loaders["train"], cfg)
                            print(f"Epoch {epoch}, autotuning weights {min(weights.values()), max(weights.values())}")
                            tune_metrics['max_weight'].append(max(weights.values()))
                            tune_metrics['min_weight'].append(min(weights.values()))
                            print(weights.values())
                            for k, v in weights.items(): 
                                weights[k] = 0.0 if v < 0.95 else 1.0
                            print("weight values", weights.values())
                            layer_weights = [sum(x) for x in zip(layer_weights, weights.values())]
                            tune_metrics['layer_weights'] = layer_weights
                            params = defaultdict()
                            for n, p in model.named_parameters():
                                if "bn" not in n:
                                    params[n] = p 
                            params_weights = []
                            for k, v in params.items():
                                if k in weights.keys():
                                    params_weights.append({"params": params[k], "lr": weights[k]*lr})
                                else:
                                    params_weights.append({"params": params[k], "lr": 0.0})
                            opt = optim.Adam(params_weights, lr=lr, weight_decay=wd)
                            
                        else:
                            # Log rough fraction of parameters being tuned
                            no_weight = 0
                            for elt in params_weights:
                                if elt['lr'] == 0.:
                                    no_weight += elt['params'][0].flatten().shape[0]
                            total_params = sum(p.numel() for p in model.parameters())
                            tune_metrics['frac_params'].append((total_params-no_weight)/total_params)
                            print(f"Tuning {(total_params-no_weight)} out of {total_params} total")
                        
                    acc_tr, loss_tr, grad_magnitudes = train(
                        model, loaders["train"], criterion, opt, cfg, orig_model=orig_model
                    )
                    acc_te, loss_te = test(model, loaders["test"], criterion, cfg)
                    acc_val, loss_val = test(model, loaders["val"], criterion, cfg)
                    tune_metrics["acc_train"].append(acc_tr)
                    tune_metrics["acc_val"].append(acc_val)
                    tune_metrics["acc_te"].append(acc_te)
                    log_dict = {
                        f"{tune_option}/train/acc": acc_tr,
                        f"{tune_option}/train/loss": loss_tr,
                        f"{tune_option}/val/acc": acc_val,
                        f"{tune_option}/val/loss": loss_val,
                        f"{tune_option}/test/acc": acc_te,
                        f"{tune_option}/test/loss": loss_te,
                    }
                    print(f"Epoch {epoch:2d} Train acc: {acc_tr:.4f}, Val acc: {acc_val:.4f}")

                    if cfg.wandb.use:
                        wandb.log(log_dict)

                tune_metrics["lr_tested"].append(lr)
                tune_metrics["wd_tested"].append(wd)
            
            # Get test acc according to best val acc
            best_run_idx = np.argmax(np.array(tune_metrics["acc_val"]))
            best_testacc = tune_metrics["acc_te"][best_run_idx]
            best_lr_wd = best_run_idx // (cfg.args.epochs)

            print(
                f"Best epoch: {best_run_idx % (cfg.args.epochs)}, Test Acc: {best_testacc}"
            )

            data = {
                "corruption_type": corruption_type,
                "train_mode": cfg.args.train_mode,
                "tune_option": tune_option,
                "auto_tune": cfg.args.auto_tune,
                "train_n": cfg.args.train_n,
                "seed": cfg.args.seed,
                "lr": tune_metrics["lr_tested"][best_lr_wd],
                "wd": tune_metrics["wd_tested"][best_lr_wd],
                "val_acc": tune_metrics["acc_val"][best_run_idx],
                "best_testacc": best_testacc,
            }

            recorded = False
            fieldnames = data.keys()
            csv_file_name = f"{cfg.args.log_dir}/results_seed{cfg.args.seed}.csv"
            write_header = True if not os.path.exists(csv_file_name) else False
            while not recorded:
                try:
                    with open(csv_file_name, "a") as f:
                        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0.0)
                        if write_header:
                            csv_writer.writeheader()
                        csv_writer.writerow(data)
                    recorded = True
                except:
                    time.sleep(5)


if __name__ == "__main__":
    main()
