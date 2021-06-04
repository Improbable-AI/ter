import itertools
import logging
import numpy as np
from backtracking.common.custom_explorer import LinearDecayEpsilonGreedy, DecayEpsilonGreedy

def get_explorer(args, action_space):
    if args.explorer == 'linear-decay':
        explorer = LinearDecayEpsilonGreedy(
            start_epsilon=args.start_epsilon,
            end_epsilon=args.end_epsilon,
            decay_steps=args.final_exploration_steps,
            start_steps=args.start_epsilon_steps,
            random_action_func=action_space.sample,
        )
    elif args.explorer == 'decay':
        explorer = DecayEpsilonGreedy(epsilon=args.start_epsilon,
                                end_epsilon=args.end_epsilon,
                                decay_factor=args.epsilon_decay_factor,
                                random_action_func=action_space.sample)
    return explorer

def get_agent_config(args):
    import importlib
    if args.env.startswith('MiniGrid'):
        from backtracking.agent_config.minigrid import make_large_atari_q_func as make_q_func
        from backtracking.agent_config.minigrid import make_large_atari_error_func as make_error_func
        from backtracking.agent_config.minigrid import phi
        pass
    elif args.env.startswith('Sokoban'):     
        from backtracking.agent_config.sokoban import make_large_atari_q_func as make_q_func
        from backtracking.agent_config.sokoban import make_large_atari_error_func as make_error_func
        from backtracking.agent_config.sokoban import phi
    else:
        raise NotImplemented()
        
    return make_q_func, phi, make_error_func