import numpy as np
import torch
from env import get_env


def evaluate(actor_critic, cfg, num_processes, writer,
             device, eval_step, logger):
    eval_envs = get_env(cfg=cfg, num_workers=num_processes,
                        device=device, seed=cfg['seed'] + 1000)
    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    mean_reward = np.mean(eval_episode_rewards).item()
    median_reward = np.median(eval_episode_rewards).item()
    min_reward = np.min(eval_episode_rewards).item()
    max_reward = np.max(eval_episode_rewards).item()
    writer.add_scalar(tag="Eval Mean Reward Of Last 10 Episode Rewards",
                      scalar_value=mean_reward, global_step=eval_step)
    writer.add_scalar(tag="Eval Median Reward Of Last 10 Episode Rewards",
                      scalar_value=median_reward, global_step=eval_step)
    writer.add_scalar(tag="Eval Min Reward Of Last 10 Episode Rewards",
                      scalar_value=min_reward, global_step=eval_step)
    writer.add_scalar(tag="Eval Max Reward Of Last 10 Episode Rewards",
                      scalar_value=max_reward, global_step=eval_step)

    logger.info(
        f'Eval Number:{eval_step}, mean reward: {mean_reward}, median reward: {median_reward}, min reward: {min_reward}, max_reward: {max_reward}')
    
    return mean_reward
