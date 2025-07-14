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
from torch.distributions import Categorical


@dataclass
class Args:
    exp_name: str = "lunarlander"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True

    wandb_project_name: str = "cleanrl-lunarlander"
    wandb_entity: str = "kris_reddy"
    use_wandb: bool = True
    capture_video: bool = False
    save_model: bool = True
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    episodes: int = 5000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    rollout: int = 10


def make_env(env_id, seed, capture_video, run_name,eval_mode=False):

    if capture_video :
        env = gym.make(env_id, render_mode="rgb_array")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)

    return env


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

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
                action,_,_,_ = choose_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; ep_r += r
        returns.append(ep_r)
    env.close()
    return returns, frames

def set_seed(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

def choose_action(state):
    state = torch.FloatTensor(state).to(device)
    probs = actor(state)
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    value = critic(state)
    return action.item(), log_prob, value, dist.entropy()

def compute_returns(rewards, dones, last_value):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + args.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

def update( trajectory, last_state):
        states, actions, log_probs, values, rewards, dones, entropies = zip(*trajectory)

        log_probs = torch.stack(log_probs).to(device)
        values = torch.stack(values).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        entropies = torch.stack(entropies).to(device)

        with torch.no_grad():
            last_state = torch.FloatTensor(last_state).to(device)
            last_value = critic(last_state)

        returns = compute_returns(rewards, dones, last_value)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        advantages = returns - values.detach()

        # Losses
        actor_loss = -(log_probs * advantages).mean()- 0.001 * entropies.mean()
        critic_loss = nn.MSELoss()(values, returns)

        # Backprop
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        return actor_loss.item(), critic_loss.item(), advantages.mean().item(), values.mean().item(), entropies.mean().item()
        
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
actor = Actor(env.observation_space.shape[0], env.action_space.n ).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
critic = Critic(env.observation_space.shape[0]).to(device)
critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate)
actor.train()
critic.train()

# TRY NOT TO MODIFY: start the game
obs, _ = env.reset(seed=args.seed)
start_time = time.time()
trajectory = []

done = False
for global_step in range(args.episodes):
    # ALGO LOGIC: put action logic here
    obs, _ = env.reset()  # Reset the environment
    done = False
    count = 0
    while not done:
        count += 1
        action, log_prob, value, entropy  = choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        real_next_obs = next_obs.copy()
        done = terminated or truncated
        trajectory.append((obs, action, log_prob, value, reward, done,entropy))
        
        obs = next_obs

        if count % args.rollout == 0:
            actor_loss, critic_loss, advantage_mean, Value_mean, entropy_mean  = update(trajectory, real_next_obs)
            trajectory.clear()
            #print(f"Step {global_step}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Advantage Mean: {advantage_mean:.4f}, Value Mean: {Value_mean:.4f}, Entropy Mean: {entropy_mean:.4f}")
                
            if global_step % 100 == 0 and args.use_wandb:
                wandb.log({"actor_loss":actor_loss , "critic_loss": critic_loss, "advantage_mean": advantage_mean,"value_mean": Value_mean,  "entropy_mean": entropy_mean,"global_step": global_step})
        if done:
            done  = False
            break

    if "episode" in info:
            if args.use_wandb:
                wandb.log({
                    "episodic_return": info['episode']['r'],
                    "episodic_length": info['episode']['l'],
                    "entropy": entropy.mean().item()
                })
            #print(f"count {count}, Return: {info['episode']['r']:.2f}, Length: {info['episode']['l']}, Time: {time.time() - start_time:.2f}s")

    if args.save_model and global_step %500 == 0:
        
        #model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        #torch.save(q_network.state_dict(), model_path)
        #print(f"model saved to {model_path}")
        
        episodic_returns, eval_frames = evaluate(actor, device,run_name,num_eval_eps=4,record=False)
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
    train_video_path = f"videos/lunarlander/final.mp4"
    returns, frames = evaluate(actor, device, run_name,num_eval_eps=1, record=True)
    # if os.path.exists(train_video_path) and os.listdir(train_video_path):
        # wandb.log({"train_video": wandb.Video(f"{train_video_path}/rl-video-episode-0.mp4")})
    imageio.mimsave(train_video_path, frames, fps=30)
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()

if args.capture_video:
    cv2.destroyAllWindows()
