import os
import random
import time
from dataclasses import dataclass
import imageio
import cv2
import wandb

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "DDQN_CartPole_3"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True

    wandb_project_name: str = "CartPole"
    wandb_entity: str = None
    use_wandb: bool = True
    capture_video: bool = False
    save_model: bool = True
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    total_timesteps: int = 300000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10


def make_env(env_id, seed, capture_video, run_name,eval_mode=False):

    if capture_video :
        env = gym.make(env_id, render_mode="rgb_array")
        if eval_mode:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}/eval")
        else:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}/train")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)

    return env


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class LinearEpsilonDecay(nn.Module):
    def __init__(self, initial_eps, end_eps, total_timesteps):
        super(LinearEpsilonDecay, self).__init__()
        self.initial_eps = initial_eps
        # self.decay_factor = decay_factor
        self.total_timesteps = total_timesteps
        self.end_eps = end_eps
        
        
    def forward(self, current_timestep, decay_factor):
        slope = (self.end_eps - self.initial_eps) / (self.total_timesteps * decay_factor)
        return max(slope * current_timestep + self.initial_eps, self.end_eps)


def setup_wandb(args, run_name):
   

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=False,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

def evaluate(model, device, run_name, num_eval_eps=10, record=False):
    """Evaluate model for num_eval_eps episodes. Optionally record frames."""
    env = make_env(Args.env_id, Args.seed, record, run_name, True)
    env.action_space.seed(Args.seed)
    model = model.to(device)
    model.eval()
    returns, frames = [], []
    for _ in range(num_eval_eps):
        obs, _ = env.reset(); done = False; ep_r = 0.0
        while not done:
            if record:
                if ep_r > 500: break
                frames.append(env.render())
            with torch.no_grad():
                a = model(torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)).argmax().item()
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc; ep_r += r
        returns.append(ep_r)
    env.close()
    return returns, frames

def set_seed(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


args = tyro.cli(Args)

run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
if args.track:
    setup_wandb(args, run_name)
os.makedirs(f"videos/{run_name}/train", exist_ok=True)
os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
set_seed(args.seed, args.torch_deterministic)

# env setup
env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)

#intitialize agent
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
q_network = QNetwork(env).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
target_network = QNetwork(env).to(device)
target_network.load_state_dict(q_network.state_dict())
epsilon_decay = LinearEpsilonDecay(args.start_e, args.end_e, args.total_timesteps)
q_network.train()
target_network.train()

rb = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device=device, handle_timeout_termination=False)


# TRY NOT TO MODIFY: start the game
obs, _ = env.reset(seed=args.seed)
start_time = time.time()

for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = epsilon_decay(global_step, args.exploration_fraction)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        q_values = q_network(torch.Tensor(obs).to(device))
        action = torch.argmax(q_values).cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, rewards, terminated, truncated, info = env.step(action)

    
    #save data to reply buffer
    real_next_obs = next_obs.copy()
    done = terminated or truncated
    rb.add(obs, real_next_obs, action, rewards, terminated, info)


    if "episode" in info:
        print(f"Step={global_step}, Return={info['episode']['r']}")
        
        # WandB logging
        if args.use_wandb:
            wandb.log({
                "episodic_return": info['episode']['r'],
                "episodic_length": info['episode']['l'],
                "epsilon": epsilon,
                "global_step": global_step
            })

    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        data = rb.sample(args.batch_size)
        with torch.no_grad():

                        # Get the actions with the highest Q-value from the main network
            next_q_values = q_network(data.next_observations)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)
            
            # Use the target network to evaluate those actions
            target_q_values = target_network(data.next_observations)
            target_max = target_q_values.gather(1, next_actions).squeeze(1)
            td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
        old_val = q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        if global_step % 100 == 0:
            if args.use_wandb:
                wandb.log({
                    "losses/td_loss": loss.item(),
                    "mean_q_values": old_val.mean().item(),
                    "SPS": int(global_step / (time.time() - start_time)),
                    "global_step": global_step
                })
            print("SPS:", int(global_step / (time.time() - start_time)))

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update target network
        if global_step % args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                )

    if args.save_model and global_step %1000 == 0:
        
        #model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        #torch.save(q_network.state_dict(), model_path)
        #print(f"model saved to {model_path}")
        
        episodic_returns, eval_frames = evaluate(q_network, device, run_name)
        avg_return = np.mean(episodic_returns)

        if args.use_wandb:
            wandb.log({
                # "val_episodic_returns": episodic_returns,
                "avg_return": avg_return,
                "val_step": global_step
            })
        print(f"Evaluation returns: {episodic_returns}")

    if done:
        obs, _ = env.reset()
    else:
        obs = next_obs

"""envs.close()
writer.close()"""

if args.use_wandb:
    train_video_path = f"videos/final.mp4"
    returns, frames = evaluate(q_network, device, run_name, record=True)
    # if os.path.exists(train_video_path) and os.listdir(train_video_path):
        # wandb.log({"train_video": wandb.Video(f"{train_video_path}/rl-video-episode-0.mp4")})
    imageio.mimsave(train_video_path, frames, fps=30)
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()

if args.capture_video:
    cv2.destroyAllWindows()
