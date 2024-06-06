"""Uses Stable-Baselines3 to train agents to play the custom Ecosystem environment using SuperSuit vector envs.

Based on Waterworld algorithm by Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time
from copy import copy
import json
from typing import Callable

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.a2c import MlpPolicy

import ecosystem_sa
from pettingzoo.utils.env import ParallelEnv
from supersuit.vector import MarkovVectorEnv

import torch
print("CUDA: ", torch.cuda.is_available())

def pettingzoo_env_to_vec_env_v1_black_death(parallel_env):
    assert isinstance(
        parallel_env, ParallelEnv
    ), "pettingzoo_env_to_vec_env takes in a pettingzoo ParallelEnv. Can create a parallel_env with pistonball.parallel_env() or convert it from an AEC env with `from pettingzoo.utils.conversions import aec_to_parallel; aec_to_parallel(env)``"
    assert hasattr(
        parallel_env, "possible_agents"
    ), "environment passed to pettingzoo_env_to_vec_env must have possible_agents attribute."
    return MarkovVectorEnv(parallel_env, black_death=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def train(env_fn,
          g=0.99,
          lr=1e-3,
          net_arch=[64, 64, 32, 32],
          steps: int = 1_000_000,
          algorithm="ppo",
          leaky=True,
          n_steps = 1600,
          batch_size=32,
          lr_schedule=False,
          seed: int | None = 0,
          **env_kwargs):
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    env = ss.black_death_v3(env=env)
    env = pettingzoo_env_to_vec_env_v1_black_death(env)
    env = ss.concat_vec_envs_v1(env, 10, num_cpus=1, base_class="stable_baselines3")
    lr_text = str(lr).replace('.', '-')
    if leaky == True:
        policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
                        net_arch=dict(pi=net_arch, vf=net_arch))
        act = "LReLU"
    else:
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=net_arch, vf=net_arch))
        act = "ReLU"
    net_string = ""
    if lr_schedule:
        lrs = "LRS"
        lr = linear_schedule(lr_schedule)
    else:
        lrs = "CLR"
    print(policy_kwargs)
    for layer in net_arch:
        net_string += str(layer) + "_"
    print(f"Starting training on {str(env.metadata['name'])}.")
    log_file = f"{env.unwrapped.metadata.get('name')}_{algorithm}_{act}_{net_string}g{int(g*1000)}_lr{lr_text}_nsteps{n_steps}_batch{batch_size}_{lrs}_{steps}_{time.strftime('%Y%m%d-%H%M%S')}"
    if algorithm == "ppo":
        model = PPO(
            MlpPolicy,
            env,
            n_steps= n_steps,
            verbose=3,
            gamma=g,
            learning_rate=lr,
            device="cpu",
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_file
        )
    elif algorithm == "a2c":
        model = A2C(
            MlpPolicy,
            env,
            verbose=3,
            gamma=g,
            device="cpu",
            learning_rate=lr,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_file)

    model.learn(total_timesteps=steps)
    model.save(log_file)
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()

def eval(env_fn, num_games: int = 100, policy="latest", render_mode: str | None = None, **env_kwargs):
    """
    Before evaluation, specific policies should be placed in the working directory
    """
    env = env_fn.env(render_mode=render_mode)
    act_dict = {}
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])}, policy {policy}, (num_games={num_games}, render_mode={render_mode})"
    )
    if policy == "latest":
        try:
            latest_policy = max(
                glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
            )
        except ValueError:
            print("No policy found.")
            exit(0)
    elif policy == "ppo":
        try:
            latest_policy = max(
                glob.glob(f"{env.metadata['name']}*ppo*.zip"), key=os.path.getctime
            )
        except ValueError:
            print("PPO Policy not found.")
            exit(0)
    elif policy == "a2c":
        try:
            latest_policy = max(
                glob.glob(f"{env.metadata['name']}*a2c*.zip"), key=os.path.getctime
            )
        except ValueError:
            print("A2C Policy not found.")
            exit(0)
    if "ppo" in latest_policy:
        model = PPO.load(latest_policy)
    elif "a2c" in latest_policy:
        model = A2C.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}
    actions = {}
    for i in range(num_games):
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]
                
            if str(agent) in actions:
                actions[str(agent)].append(act.tolist())
            else:
                actions[str(agent)] = [act.tolist()]
            if str(act) in act_dict:
                act_dict[str(act)] += 1
            else:
                act_dict[str(act)] = 1
            env.step(act)
            
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    print(act_dict)
    result = {
        "rewards": rewards,
        "avg_reward": avg_reward,
        "action_dict": act_dict,
        "actions": actions
    }
    with open(f"{str(latest_policy)}.json", "w") as outfile: 
        json.dump(result, outfile)
    return result

if __name__ == "__main__":
    env_fn = ecosystem_sa
    pars = {
        "gamma": [0.95, 0, 0.25, 0.5, 0.75, 0.9, 0.99],
        "lr": [1e-3, 1e-2, 1e-4, 1e-5, 4e-3]
    }
    costs_dict = {
    }
    env_kwargs = {"render_mode": "none",
                  "observation_mode": "single",
                  "params": costs_dict
                  }
    train(
        env_fn,
        g=0.99,
        lr=0.001,
        net_arch=[128, 64, 64, 32],
        steps=5_000_000,
        algorithm="ppo",
        leaky=True,
        n_steps=3200,
        batch_size=16,
        lr_schedule=False,
        seed=0,
        **env_kwargs)
    eval(env_fn, num_games=100, policy="ppo", render_mode="None")
    eval(env_fn, num_games=2, policy="ppo", render_mode="human")