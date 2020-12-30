# mujoco-ppo

This repository is a minimal version of the [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) repository, which only includes the PyTorch implementation of [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf), and is designed to be friendly and flexible for [mujoco environments](https://gym.openai.com/envs/#mujoco). The default hyperparameters are adjusted according to the suggestions in the original [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) repository.

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Training

```bash
python main.py --env-name "HalfCheetah-v2"
```

## Enjoy

```bash
python enjoy.py --load-dir trained_models --env-name "HalfCheetah-v2"
```

## Visualization

In order to visualize the results use ```visualize.ipynb```.

## Results

![halfcheetah](imgs/ppo_halfcheetah.png)

![hopper](imgs/ppo_hopper.png)

![reacher](imgs/ppo_reacher.png)

![walker](imgs/ppo_walker.png)
