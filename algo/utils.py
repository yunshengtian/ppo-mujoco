import yaml
import glob
import os
import torch.nn as nn
import torch
import argparse
import logging
from algo.envs import VecNormalize


def save_model(save_path, agent, epoch, is_best: bool = False):
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    file_name = "checkpoint.pt"

    if is_best:
        file_name = 'best_' + file_name

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": agent.actor_critic.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        },
        os.path.join(save_path, file_name))


def get_logger(cfg):
    name = cfg['id']
    save_path = os.path.join(
        "./logging", cfg['task'], cfg["algorithm"], cfg["id"], str(cfg['seed']))
    logger = logging.getLogger(name)

    try:
        os.makedirs(save_path)
    except OSError:
        pass

    file_path = os.path.join(save_path, 'logger.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def get_config():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/mujoco/clean.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    if not cfg["id"]:
        raise ValueError('"id" should not be none in config yaml')

    return cfg

# Get a render function


def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
