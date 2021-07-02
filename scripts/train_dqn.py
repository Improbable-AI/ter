import argparse
from argparse import Namespace

import logging
import os
import ruamel.yaml
import gin
import sys
import datetime

from gym import spaces

import torch
import pfrl
from pfrl import experiments
from pfrl import utils

import numpy as np

import backtracking
import backtracking.agents
from backtracking.envs.wrappers import wrap_env
from backtracking.replay_buffers import EfficientReverseSweepReplayBuffer

from backtracking.experiments.train_agent import train_agent_with_evaluation, Monitor
from backtracking.experiments.helpers import get_explorer, get_agent_config

    
def make_agent(args, obs_space, action_space):
    explorer = get_explorer(args, action_space)
    
    # This line is task dependent. Make sure your task has a corresponding agent_config
    make_q_func, phi, make_error_func = get_agent_config(args)
   
    q_func = make_q_func(args.env, obs_space, action_space)
    optimizer = torch.optim.Adam(q_func.parameters(), lr=args.lr)
    
    if args.algo == 'DISCOR-DQN' or args.algo == 'DISCOR-DDQN':
        error_func = make_error_func(args.env, obs_space, action_space)
        error_model_optimizer = torch.optim.Adam(error_func.parameters(), lr=args.lr)
    else:
        error_func = None
        error_model_optimizer = None
    
    if args.replay == 'TER':
        betasteps = args.steps / args.update_interval
        replay_buffer = EfficientReverseSweepReplayBuffer(
            obs_space=obs_space,
            capacity=args.replay_capacity,
            alpha=0.6,
            beta0=0.4,
            betasteps=betasteps,
            num_steps=1
        ) # TODO: account for n-step later
    elif args.replay == 'PER':
        betasteps = args.steps / args.update_interval
        replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(
            args.replay_capacity,
            alpha=0.6,
            beta0=0.4,
            betasteps=betasteps,
            num_steps=1, # TODO: account for n-step later
        )
    elif args.replay == 'UER':
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=args.replay_capacity)
    elif args.replay in ['EBU']:
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=args.replay_capacity)
    else:
        raise NotImplemented()

    if args.algo == 'DDQN':
        if args.replay == 'EBU':
            algo_cls = backtracking.agents.DoubleEBUDQN
        else:
            algo_cls = backtracking.agents.DoubleDQN 
    elif args.algo == 'DISCOR-DDQN':
        algo_cls = backtracking.agents.DoubleDisCorDQN

    agent = algo_cls(
        q_function=q_func,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=args.gamma,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        minibatch_size=args.batch_size,
        update_interval=args.update_interval,
        n_times_update=args.n_times_update,
        target_update_interval=args.target_update_interval,        
        target_update_method= 'hard' if args.soft_update_tau == 1.0 else 'soft',
        soft_update_tau=args.soft_update_tau,
        gpu=args.gpu,
        phi=phi,
        max_grad_norm=args.max_grad_norm,
        replay_buffer_snapshot_path=args.outdir if args.store_replay_buffer_snapshot else None,
        replay_buffer_snapshot_interval=args.replay_buffer_snapshot_interval,
        batch_accumulator=args.batch_accumulator,
        error_func=error_func,
        error_model_optimizer=error_model_optimizer,
    )
    
    
    return agent

def main(args):
     # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    time_format = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.datetime.now().strftime(time_format)
    exp_id = '{}-{}-{}'.format(args.algo, args.replay, timestamp)
    args.outdir = experiments.prepare_output_dir(args, args.outdir,
                                                 exp_id=exp_id,
                                                 argv=sys.argv)
    logging.basicConfig(level=args.log_level)
    print("Output files are saved in {}".format(args.outdir))    
    
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(idx=0, test=False):        
        env = wrap_env(args.env, test)
        
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        utils.set_random_seed(env_seed)
        
        if args.monitor:
            env = Monitor(env, args.outdir)
        
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, args.noise_eval)
            
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
            
        if args.vis:
            env = pfrl.wrappers.Render(env)
            
        return env
    
    env = make_env()
    eval_env = make_env(test=True)

    from gym.wrappers import TimeLimit
    if hasattr(env, 'spec') and env.spec is not None:
        timestep_limit = env.spec.max_episode_steps
    else:
        timestep_limit = env.unwrapped.max_steps
    obs_space = env.observation_space
    action_space = env.action_space 
  
    agent = make_agent(args, obs_space, action_space)
    print("train")
    train_agent_with_evaluation(
                agent=agent,
                env=env,
                steps=args.steps,
                eval_n_steps=None,
                eval_n_episodes=args.eval_n_runs,
                eval_interval=args.eval_interval,
                eval_start_steps=args.replay_start_size,
                outdir=args.outdir,
                eval_env=eval_env,
                train_max_episode_len=timestep_limit,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str,  #required=True,
                        choices=['DDQN', 'DISCOR-DDQN'])
    parser.add_argument('--replay', type=str, # required=True,
                        help='Type of experience replay',
                        default='UER')
    
    parser.add_argument('--explorer', type=str, # required=True,
                        choices=['linear-decay', 'decay'])
    
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--env", type=str, default="Acrobot-v1")

    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use, set to -1 if no GPU.")
    parser.add_argument("--steps", type=int, default=10 ** 5, help="Total number of timesteps to train the agent.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-n-runs", type=int, default=20, help="Number of episodes run for each evaluation.")
    parser.add_argument("--eval-interval", type=int, default=500, help="Interval in timesteps between evaluations.",)
    parser.add_argument("--update-interval", type=int, default=250, help="Interval in timesteps between update.")
    parser.add_argument("--soft-update-tau", type=float, default=1.0, help="Tau for soft update, 1.0 is hard.")
    parser.add_argument("--final-exploration-steps", type=int, default=50000)
    parser.add_argument("--start-epsilon-steps", type=int, default=0)
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--end-epsilon", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-factor", type=float, default=0.995)
    parser.add_argument("--batch-accumulator", type=str, default='mean')
    
    parser.add_argument("--replay-start-size", type=int, default=20000)
    
    parser.add_argument("--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor.")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--store-replay-buffer-snapshot", action="store_true")
    parser.add_argument("--replay-buffer-snapshot-interval", type=int, default=100000)
    parser.add_argument("--noise-eval", type=float, default=0.0)
    parser.add_argument("--log-level", type=int, default=logging.INFO, help="Level of the root logger.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--reward-scale-factor", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    
    # DQN only
    parser.add_argument("--batch-size", type=int, default=100, help="Minibatch size")
    parser.add_argument("--n-times-update", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update-interval", type=int, default=250, 
                        help="Interval in timesteps between update.")

    # RS only
    parser.add_argument("--rs-config-dir", type=str, 
                    default='rs_configs')
    parser.add_argument("--rs-gin-files", action='append', type=str, default=[])
    parser.add_argument("--rs-gin-bindings", action='append', type=str, default=[])

    # YAML config
    parser.add_argument("--config", action='append', type=str, default=[])
    args = parser.parse_args()

    # TODO: replace args by yaml config (for backward compatabililty, we preserve argparse)
    if len(args.config) > 0:
        args_dict = vars(args)
        for cfg_file in args.config:
            with open(cfg_file) as file:
                config = ruamel.yaml.load(file)
            for cfg_key, cfg_val in config.items():
                if cfg_key in args_dict:
                    print("Overwrite arg={}: {} by {}".format(cfg_key, args_dict[cfg_key], cfg_val))
                args_dict[cfg_key] = cfg_val
        args = Namespace(**args_dict)

    if args.seed == -1:
        import random        
        args.seed = random.randint(0, 1000)
        print('Use randomly generated seed: {}'.format(args.seed))

    rs_gin_files = [os.path.join(args.rs_config_dir, f) for f in args.rs_gin_files]
    print('RS Gin files:', rs_gin_files)
    gin.parse_config_files_and_bindings(rs_gin_files,
                                      bindings=args.rs_gin_bindings,
                                      skip_unknown=False)
    
    main(args)
