import os
import time
from collections import deque

import numpy as np
import torch

from algo import PPO, utils
from algo.arguments import get_args
from algo.model import Policy
from algo.storage import RolloutStorage
from algo.utils import get_vec_normalize, get_config, get_logger
from evaluation import evaluate

from env import get_env


def main(cfg: dict):
    seed = cfg['seed']
    num_workers = cfg['num_workers']
    num_steps = cfg['train']['num_steps']
    num_env_steps = cfg['train']['num_env_steps']
    save_interval = cfg["train"]["save_interval"]
    log_interval = cfg["log_interval"]
    save_path = os.path.join(
        "./checkpoints", cfg["algorithm"], cfg["id"], str(seed))
    algo_args = cfg['train']['algorithm_params']
    # args = get_args()

    # log_dir = os.path.expanduser(args.log_dir)
    # eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir)
    # if args.eval_interval is not None:
    #     utils.cleanup_log_dir(eval_log_dir)

    torch.manual_seed(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = get_env(cfg=cfg, num_workers=num_workers, device=device)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)
    actor_critic.to(device)

    # if args.load_dir is None:
    #     actor_critic = Policy(
    #         envs.observation_space.shape,
    #         envs.action_space,
    #         base_kwargs={'recurrent': args.recurrent_policy})
    #     actor_critic.to(device)
    # else:
    #     load_path = args.load_dir if args.load_dir.endswith('.pt') else os.path.join(args.load_dir, args.env_name + '.pt')
    #     actor_critic, ob_rms = torch.load(load_path)
    #     vec_norm = get_vec_normalize(envs)
    #     if vec_norm is not None:
    #         vec_norm.eval()
    #         vec_norm.ob_rms = ob_rms

    agent = PPO(actor_critic=actor_critic, **algo_args)

    rollouts = RolloutStorage(num_steps=num_steps, num_processes=num_workers,
                              obs_shape=envs.observation_space.shape, action_space=envs.action_space,
                              recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(num_env_steps) // num_steps // num_workers
    for j in range(num_updates):

        if not cfg['train']['disable_linear_lr_decay']:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                optimizer=agent.optimizer, epoch=j, total_num_epochs=num_updates, initial_lr=algo_args['lr'])

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(
            next_value=next_value, **cfg['train']['compute_returns'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % save_interval == 0 or j == num_updates - 1):
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, "checkpoint.pt"))

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_workers * num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     ob_rms = utils.get_vec_normalize(envs).ob_rms
        #     evaluate(actor_critic, ob_rms, args.env_name, seed,
        #              num_workers, eval_log_dir, device)


if __name__ == "__main__":
    cfg = get_config()

    if cfg['device_id'] is not None:
        with torch.cuda.device(cfg['device_id']):
            main(cfg)
    else:
        main(cfg)
