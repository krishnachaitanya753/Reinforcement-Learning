import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import tyro
from tqdm import tqdm
import time
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

class Args:

    exp_name: str = "PPO"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True

    wandb_project_name: str = "PPO"
    wandb_entity: str = None
    use_wandb: bool = True
    capture_video: bool = False
    save_model: bool = True
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 500000
    num_steps: int = 128
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    update_epochs: int = 4
    buffer_size: int = 100000
    gae: bool = True
    gamma: float = 0.99
    Lambda: float = 0.95
    tau: float = 1
    target_network_frequency: int = 500
    anneal_lr: bool = True
    batch_size: int = num_envs * num_steps
    learning_starts: int = 10
    train_frequency: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    norm_adv: bool = True
    clip_vloss: bool = True


def make_env(env_id, seed, capture_video):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env
    return thunk

def layer_init(layer,std = np.sqrt(2), bias_const=0.0 ):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# load tha previously saved model and continue training fromppo.cleanrl_model

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
        layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor = nn.Sequential(
        layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        )

    def value(self,x):
        return self.critic(x)
    
    def get_action_and_value(self,x,action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.critic(x)
        return action,probs.log_prob(action) ,probs.entropy(), value  

def evaluate(model, device, run_name, num_eval_eps=10, record=False):
    """Evaluate model for num_eval_eps episodes. Optionally record frames."""
    env = gym.make(args.env_id, render_mode="rgb_array" if record else None)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_eval_eps)
    env.observation_space.seed(Args.seed)
    env.action_space.seed(Args.seed)
    model = model.to(device)
    model.eval()
    returns, frames = [], []
    for _ in range(num_eval_eps):
        obs, _ = env.reset(); done = False; ep_r = 0.0
        
        while not done:
            obs = torch.tensor(obs, device=device, dtype=torch.float32)
            if record:
                if ep_r > 500: break
                frames.append(env.render())
            with torch.no_grad():
                action,_,_,_ = model.get_action_and_value(obs)
            obs, r, term, trunc, _ = env.step(action.cpu().numpy())
            done = term or trunc; ep_r += r
        returns.append(ep_r)
    env.close()
    return returns, frames


if __name__ == "__main__":
    
    args = Args()
    run_name = f"{args.exp_name}_seed{args.seed}"

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config = vars(args),name=run_name, monitor_gym=True, save_code=True)

    start_time = time.time()

    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    writer.add_text("hyperparamters", "|parameter|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in args.__dict__.items()]))

    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed+i,  args.capture_video ) for i in range(args.num_envs)])
    observations,_ = envs.reset()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action spaces are supported in this env."


    agent = Agent(envs).to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)

    #storage set-up
    obs = torch.zeros((args.num_steps,args.num_envs) + envs.single_observation_space.shape, device=device, dtype=torch.float32   ) 
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs,_ = envs.reset()
    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros((args.num_envs), device=device, dtype=torch.float32)
    num_updates = args.total_timesteps // (args.batch_size)

    #print(agent.get_action_and_value(next_obs))

    global_step = 0
    for update in tqdm(range(1,num_updates+1)):

        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lrnow = args.learning_rate * frac
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(args.num_steps):
            global_step += 1*args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action,log_prob, entropies, value = agent.get_action_and_value(next_obs)
                value = value.flatten()
            actions[step] = action
            logprobs[step] = log_prob
            values[step] = value 

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32).view(-1)
            next_obs,next_done = torch.tensor(next_obs, device=device, dtype=torch.float32), torch.tensor(terminations | truncations, device=device, dtype=torch.float32)
            
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.Lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.num_envs):
                end = start + args.num_envs
                mb_inds = b_inds[start:end]
                mb_advantages = b_advantages[mb_inds]
                

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  
                #print(f"pg_loss: {pg_loss.item()}")
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()              

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if eval and update%1000 == 0:
            returns, frames = evaluate(model=agent, device=device, run_name=run_name, num_eval_eps=10, record=args.capture_video)
            writer.add_scalar("charts/episodic_return", np.mean(returns), global_step)
            writer.add_scalar("charts/episodic_length", np.mean([len(frame) for frame in frames]), global_step)
        if args.save_model and update%((update//5+1)):
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")



    envs.close()
    writer.close()


returns, frames   = evaluate(model=agent, device=device, run_name=run_name, num_eval_eps=1, record=True)

import imageio
if frames:  
    video_path = f"videos/{run_name}.mp4"
    imageio.mimwrite(video_path, frames, fps=30, quality=8)
    print(f"Video saved to {video_path}")

        

    


