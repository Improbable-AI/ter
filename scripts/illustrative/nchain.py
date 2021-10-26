import random
import collections
from collections import deque
from collections import OrderedDict

import copy

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class NChainEnv(gym.Env):
    def __init__(self, n=5, slip=0.2, small=2, large=10):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        if self.np_random.rand() < self.slip:
            action = not action  # agent slipped, reverse action taken

        if action:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state = max(self.state - 1, 0)
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            self.state += 1
            reward = self.small
            if self.state == self.n - 1:
              reward = self.large
              done = True
        
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state


class BaseQLearning(object):
    def __init__(self, observation_space, action_space, epislon_greedy_decay=0.995,
                 lr=0.98, discount_factor=0.99, init='zero', 
                 record_td_errors=False,
                 record_q_values=False,
                 q_snapshot_interval=10):
        self.lr = lr
        self.observation_space = observation_space
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.epislon_greedy_decay = epislon_greedy_decay
        self.record_td_errors = record_td_errors
        self.record_q_values = record_q_values
        self.q_snapshot_interval = q_snapshot_interval
        
        if init == 'zero':
            self.Q = np.zeros((observation_space.n, action_space.n))
        elif init == 'uniform':
            self.Q = np.random.random((observation_space.n, action_space.n)) * 0.1
        self._init_memory()
        self.eps = 1.0

        self.td_errors_record = {s: {a: [] for a in range(self.Q.shape[1])} for s in range(self.Q.shape[0])}
        self.q_record = []
        self.select_sa_record = []
        self.n_updates = 0    

    def _init_memory(self):
        raise NotImplemented()
            
    def reset(self):
        pass
    
    def memorize(self, s_t, a_t, r_t, s_tp1, d_t, info_t):
        raise NotImplemented()

    def select_action(self, s_t, eval_mode=False):
        if np.random.random() < self.eps and not eval_mode:
            a_t = self.action_space.sample()
        else:
            a_t = self.greedy_action(s_t)
        
        if not eval_mode:
            self.eps *= self.epislon_greedy_decay

        return a_t
        
    def greedy_action(self, s_t):
        return np.argmax(self.Q[s_t])
    
    def _update_step(self, s_t, a_t, r_t, s_tp1, d_t):
        not_done = not d_t
        delta = (r_t + not_done * self.discount_factor * np.max(self.Q[s_tp1, :]) - self.Q[s_t, a_t])
        self.Q[s_t, a_t] += self.lr * delta

        self.n_updates += 1
        if self.record_td_errors:
            self.td_errors_record[s_t][a_t].append(delta)
        if self.record_q_values and self.n_updates % self.q_snapshot_interval == 0:
            self.q_record.append(copy.copy(self.Q))
            self.select_sa_record.append((s_t, a_t, s_tp1))

        return delta
        
    def update(self):
        raise NotImplemented()

class UERQLearning(BaseQLearning):

    def _init_memory(self):
        self.replay_buffer = []

    def memorize(self, s_t, a_t, r_t, s_tp1, d_t, info_t):
        self.replay_buffer.append((s_t, a_t, r_t, s_tp1, d_t))

    def update(self):
        batch = random.sample(list(self.replay_buffer), 1)
        for s_t, a_t, r_t, s_tp1, d_t in batch:
            self._update_step(s_t, a_t, r_t, s_tp1, d_t)

class PERQLearning(UERQLearning):

    def _init_memory(self):
        super()._init_memory()
        self.priorities = []

    def memorize(self, s_t, a_t, r_t, s_tp1, d_t, info_t):
        super().memorize(s_t, a_t, r_t, s_tp1, d_t, info_t)
        self.priorities.append(1)        
    
    def update(self):
        # idx = np.argmax(self.priorities)
        idx = np.random.choice(np.flatnonzero(self.priorities == np.max(self.priorities)))
        s_t, a_t, r_t, s_tp1, d_t = self.replay_buffer[idx]
        td_error = self._update_step(s_t, a_t, r_t, s_tp1, d_t)
        self.priorities[idx] = np.abs(td_error)  

class EBUQLearning(BaseQLearning):
    def _init_memory(self):
        self.episodes = []
        self.episode_buffer = []
        self.episode_iterator = None
        self.next_sample = None

    def memorize(self, s_t, a_t, r_t, s_tp1, d_t, info_t):
        self.episode_buffer.append((s_t, a_t, r_t, s_tp1, d_t))
        if d_t: #and not info_t.get('TimeLimit.truncated', False) :
            self.episodes.append(copy.copy(self.episode_buffer))
            self.episode_buffer = []

    def update(self):
        if len(self.episodes) == 0:
            batch = random.sample(self.episode_buffer, 1)
            for s_t, a_t, r_t, s_tp1, d_t in batch:
                self._update_step(s_t, a_t, r_t, s_tp1, d_t)
        else:
            while self.next_sample is None:    
                episode = random.sample(self.episodes, 1)[0]
                self.episode_iterator = iter(reversed(episode))                      
                self.next_sample = next(self.episode_iterator, None)
            
            self._update_step(*self.next_sample)        
            self.next_sample = next(self.episode_iterator, None)

class RSQLearning(BaseQLearning):

    def _init_memory(self):
        assert isinstance(self.observation_space, gym.spaces.Discrete)
        self.n_states = n_states = self.observation_space.n
        self.n_actions = n_actions = self.action_space.n
        self.preds = [ [ [[] for s in range(n_states)] for a in range(n_actions) ] for s in range(n_states)]
        self.replay_buffer = []
        self.terminal_state_set = set()
        self.search_queue = deque()
        self.transition_queue = deque()
        self.visited = list([False for s in range(n_states)])
        self.node_expand_iterator = None

    def memorize(self, s_t, a_t, r_t, s_tp1, d_t, info_t): 
        self.preds[s_tp1][a_t][s_t].append((s_t, a_t, r_t, s_tp1, d_t))
        self.replay_buffer.append((s_t, a_t, r_t, s_tp1, d_t))

        if d_t and not info_t.get('TimeLimit.truncated', False):
            # print(s_tp1, d_t, info_t.get('TimeLimit.truncated', False))
            self.terminal_state_set.add(s_tp1)

    def update(self):
        if len(self.terminal_state_set) == 0:
            batch = random.sample(list(self.replay_buffer), 1)
            for s_t, a_t, r_t, s_tp1, d_t in batch:
                self._update_step(s_t, a_t, r_t, s_tp1, d_t)
        else:
            if len(self.transition_queue) > 0:
                transition = self.transition_queue.popleft()
                self._update_step(*transition)
            else:
                while len(self.transition_queue) == 0:
                    if len(self.search_queue) == 0:
                        for s in range(self.n_states):
                            self.visited[s] = False
                        for terminal_state in self.terminal_state_set:
                            self.search_queue.append(terminal_state)
                    
                    sp = self.search_queue.popleft()
                    # import ipdb; ipdb.set_trace()
                    self.visited[sp] = True

                    for a, pred_a in enumerate(self.preds[sp]):
                        for s, pred_s in enumerate(pred_a):
                            transition_list = pred_s
                            if len(transition_list) > 0:
                                sample_transitions = random.sample(transition_list , 1)
                                for transition in sample_transitions:
                                    self.transition_queue.append(transition)
                                if not self.visited[s]:
                                    self.search_queue.append(s)
                # import ipdb; ipdb.set_trace()
                transition = self.transition_queue.popleft()
                self._update_step(*transition)

def make_env(env_name):
    env_type, env_sizes = env_name.split('-')
    env_size = int(env_sizes.split('x')[0])
    max_episode_steps = env_size * env_size

    if env_type == 'Safe':
        env = FrozenLakeEnv(generate_n_by_n_safe_map(env_size), is_slippery=False, 
                    step_penalty=-1.0 / max_episode_steps, 
                    max_r=1)
    elif env_type == 'NChain':
        env = NChainEnv(n=env_size, slip=0, small=0, large=1)
        max_episode_steps = env_size * env_size * 2
    elif env_type == 'ShortcutChain':
        max_episode_steps = env_size
        env = ShortcutChainEnv(n=env_size, slip=0, small=0, large=1)
    else:
        env =  FrozenLakeEnv(generate_random_map(env_size), is_slippery=False, 
                    step_penalty=-1.0 / max_episode_steps, 
                    max_r=env_size**2)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def _eval(env, agent, n_eval_episodes=1):
    ep_rets, ep_lens = [], []
    for ep in range(n_eval_episodes):
        s_t = env.reset()
        d_t = False
        ep_ret = 0
        t = 0

        while not d_t:
            a_t = agent.select_action(s_t, eval_mode=True)
            s_tp1, r_t, d_t, info_t = env.step(a_t)
            ep_ret += r_t
            s_t = s_tp1
            t += 1

        ep_rets.append(ep_ret)
        ep_lens.append(t)

    return np.mean(ep_rets), np.mean(ep_lens)

def train(env, agent, steps, eval_env, eval_interval):

    s_t = env.reset()
    d_t = False

    ep_ret = 0
    ep_rets = []
    ep_lens = []
    eval_mean_ep_rets = []
    eval_mean_ep_lens = []
    ep_start_t = 0
    t = 0
    while t < steps:
        a_t = agent.select_action(s_t)
        s_tp1, r_t, d_t, info_t = env.step(a_t)
        agent.memorize(s_t, a_t, r_t, s_tp1, d_t, info_t)
        agent.update()
        ep_ret += r_t        

        if d_t:
            s_t = env.reset()
            ep_rets.append(ep_ret)
            ep_lens.append(t - ep_start_t)
            ep_ret = 0
            ep_start_t = t
        else:
            s_t = s_tp1
        
        t += 1

        if t % eval_interval == 0:
            eval_mean_ep_ret, eval_mean_ep_len = _eval(eval_env, agent)
            eval_mean_ep_rets.append(eval_mean_ep_ret)
            eval_mean_ep_lens.append(eval_mean_ep_len)
    
    return ep_rets, ep_lens, eval_mean_ep_rets, eval_mean_ep_lens

def train_offline(env, agent, experiences, steps, eval_env, eval_interval):

    for s_t, a_t, r_t, s_tp1, d_t, info_t in experiences:
        agent.memorize(s_t, a_t, r_t, s_tp1, d_t, info_t)
    
    eval_mean_ep_rets = []
    eval_mean_ep_lens = []
    for t in range(steps):
        agent.update()
        
        if t % eval_interval == 0:
            eval_mean_ep_ret, eval_mean_ep_len = _eval(eval_env, agent)
            eval_mean_ep_rets.append(eval_mean_ep_ret)
            eval_mean_ep_lens.append(eval_mean_ep_len)
    
    return eval_mean_ep_rets, eval_mean_ep_lens

def collect_dataset(env, steps):
    experiences = []
    s_t = env.reset()
    d_t = False
    for t in range(steps):
        a_t = env.action_space.sample()
        s_tp1, r_t, d_t, info_t = env.step(a_t)
        experiences.append((s_t, a_t, r_t, s_tp1, d_t, info_t))
        if d_t:
            s_t = env.reset()
        else:
            s_t = s_tp1

    return experiences

def run_offline(env_name, experiences, Agent, label, steps=100, eval_interval=1, running_window=50, init='uniform', verbose=True):
    env, eval_env = make_env(env_name), make_env(env_name)
    agent = Agent(env.observation_space, env.action_space, init=init)
    eval_ep_rets, eval_ep_lens = train_offline(env=env, experiences=experiences, agent=agent, steps=steps, eval_env=eval_env, eval_interval=eval_interval)  
    agent.eval_ep_rets = eval_ep_rets
    agent.eval_ep_lens = eval_ep_lens
    agent.eval_stat = pd.DataFrame({'iter': range(len(eval_ep_rets)), 'ret': eval_ep_rets, 'len': eval_ep_lens, 'label': label})
    agent.eval_stat['len'] /= env._max_episode_steps
    if verbose:
      print('Env:', env_name)
      print('Agent:', agent)
      print('EvalEpRet:', np.mean(eval_ep_rets[-running_window:]), max(eval_ep_rets))
      print('EvalEpLen:', np.mean(eval_ep_lens[-running_window:]), min(eval_ep_lens))
      print('========')
    return agent

env_size = 20
env_name = 'NChain-{}x{}'.format(env_size, env_size)
experiences = collect_dataset(make_env(env_name), steps=1000)
if len(list(filter(lambda e: e[2] > 0, experiences))) == 0:
   assert 0 
print('Num. success:', len(list(filter(lambda e: e[2] > 0, experiences))))

all_stats = []
for i in range(5):
  uer = run_offline(env_name, experiences, UERQLearning, 'UER', steps=100, verbose=False)
  per = run_offline(env_name, experiences, PERQLearning, 'PER', steps=100, verbose=False)
  ebu = run_offline(env_name, experiences, EBUQLearning, 'EBU', steps=100, verbose=False)
  rs = run_offline(env_name, experiences, RSQLearning, 'TER', steps=100, verbose=False)  
  stats = pd.concat([uer.eval_stat, per.eval_stat, ebu.eval_stat, rs.eval_stat])
  all_stats.append(stats)
nchain_all_stats = pd.concat(all_stats)

nchain_all_stats['normalized_mean_return'] = 1.0 - nchain_all_stats['len']
nchain_all_stats = nchain_all_stats.reset_index()

sns.lineplot(x='iter', y='normalized_mean_return', hue='label', data=nchain_all_stats, linewidth=2.5)
plt.title('NChain (N=20)')
plt.savefig('fig_nchain_20.png')