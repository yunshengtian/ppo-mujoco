# ppo-mujoco

This repository is a minimal version of the [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) repository, which only includes the PyTorch implementation of [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf), and is designed to be friendly and flexible for [MuJoCo environments](https://gym.openai.com/envs/#mujoco).

## Key Features

Differences compared to the original [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) repository:

- Minimal code for PPO training and simplified installation process
- Using local environments in ```envs/``` for environment customization
- Support fine-tuning policies (i.e. training starts on the loaded policy)
- Support enjoy without learned policies (zero actions or random actions)
- The default hyperparameters are adjusted according to the suggestions from the original repository

## Requirements

```bash
pip install -r requirements.txt
```

## Training

```bash
# training from scratch
python train.py --env-name HalfCheetah-v2 --num-env-steps 1000000

# fine-tuning
python train.py --env-name HalfCheetah-v2 --num-env-steps 1000000 --load-dir trained_models
```

## Enjoy

```bash
# with learned policy
python enjoy.py --env-name HalfCheetah-v2 --load-dir trained_models

# without learned policy (zero actions)
python enjoy.py --env-name HalfCheetah-v2

# without learned policy (random actions)
python enjoy.py --env-name HalfCheetah-v2 --random
```

## Visualization

In order to visualize the training curves use ```visualize.ipynb```.

## Results

![halfcheetah](imgs/ppo_halfcheetah.png)

![hopper](imgs/ppo_hopper.png)

![reacher](imgs/ppo_reacher.png)

![walker](imgs/ppo_walker.png)
