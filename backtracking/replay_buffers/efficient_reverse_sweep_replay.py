import os
import gin
import dill
import random

from collections import deque

import numpy as np

from scipy.special import softmax
from absl import logging

# from dopamine.replay_memory.prioritized_replay_buffer import WrappedPrioritizedReplayBuffer
from pfrl.replay_buffers import PrioritizedReplayBuffer

from backtracking.replay_buffers.efficient_graphical_memory import EfficientGraph
from backtracking.replay_buffers.efficient_graphical_memory import EfficientBreadthFirstGraphSearcher

from backtracking.common.utils import RunningStat, IntRunningStat, timed_stat
from backtracking.common.encoder import RandomProjectionEncoder

def obs2img(obs):
  from PIL import Image
  img = Image.fromarray(obs, 'L')
  img.save('obs.png')
  img.show()

class TerminalNodeList(object):
  def __init__(self):
    self.terminal_node_ids = []
    self.terminal_node_indices = []
    self.terminal_node_returns = []
  
  def append(self, node_id, buffer_index, episode_ret):
    self.terminal_node_ids.append(node_id)
    self.terminal_node_indices.append(buffer_index)
    self.terminal_node_returns.append(episode_ret)

  def update_node_probs(self):
    pass

  def remove_outdated_nodes(self, latest_index, capacity):
    n_removes = 0
    for i in range(len(self.terminal_node_indices)):
      if self.terminal_node_indices[i] + capacity <= latest_index:
        n_removes += 1

    self.terminal_node_ids = self.terminal_node_ids[n_removes:] 
    self.terminal_node_indices = self.terminal_node_indices[n_removes:]
    self.terminal_node_returns = self.terminal_node_returns[n_removes:]

  def sample(self, n):
    sampled_node_ids = random.sample(self.terminal_node_ids, n)
    return sampled_node_ids

  def __len__(self):
    return len(self.terminal_node_ids)

class TerminalSet(object):
  def __init__(self):
    self.terminal_node_set = dict()
    self.terminal_node_counts = dict()
    self.terminal_node_latest_index = dict()
    self.terminal_node_probs = None

  def __len__(self):
    return len(self.terminal_node_set)

  def remove(self, node_id):
    if node_id in self.terminal_node_set:
      del self.terminal_node_set[node_id]
      del self.terminal_node_counts[node_id]
      del self.terminal_node_latest_index[node_id]

  def sample(self, n):
    return random.sample(list(self.terminal_node_set.keys()), n)

  def append(self, node_id, buffer_index, episode_ret):
    # New terminal node
    if node_id not in self.terminal_node_set:
      self.terminal_node_set[node_id] = episode_ret
      self.terminal_node_counts[node_id] = 1
      self.terminal_node_latest_index[node_id] = buffer_index
    # Existing terminal node
    else:
      mean = self.terminal_node_set[node_id]
      delta = episode_ret - mean
      count = self.terminal_node_counts[node_id]
      self.terminal_node_set[node_id] = mean + delta / (count + 1)
      self.terminal_node_latest_index[node_id] = max(buffer_index, self.terminal_node_latest_index[node_id])

  def remove_outdated_nodes(self, latest_index, capacity):
    # TODO: might slow down the training, be aware of the speed
    sorted_terminal_node_latest_index = sorted(self.terminal_node_latest_index.items(), 
                                            key=lambda item: item[1])
    # print(sorted_terminal_node_latest_index, latest_index, capacity)                                                  
    while len(self.terminal_node_latest_index) > 0 and \
              sorted_terminal_node_latest_index[0][1] + capacity <= latest_index:
      # import ipdb; ipdb.set_trace()
      node_id = sorted_terminal_node_latest_index[0][0]
      sorted_terminal_node_latest_index.pop(0)
      logging.info('The latest visited index {} of node {} is too old.'.format(sorted_terminal_node_latest_index[0][1], node_id))
      # print('Before remove {}'.format(node_id), len(self.terminal_node_set))
      del self.terminal_node_set[node_id]
      del self.terminal_node_latest_index[node_id]
      del self.terminal_node_counts[node_id]
      # print('After remove {}'.format(node_id), len(self.terminal_node_set))

  def update_node_probs(self):
    pass

class RewardPrioritizedTerminalSet(TerminalSet):
  def __init__(self, reward_prioritize_temperature):
    super().__init__()
    self.reward_prioritize_temperature = reward_prioritize_temperature
  
  def update_node_probs(self):
    pass

  def sample(self, n):
    ep_rets = np.asarray(list(self.terminal_node_set.values()))
    terminal_node_probs = softmax(ep_rets * self.reward_prioritize_temperature)
    terminal_node_ids = list(self.terminal_node_set.keys())
    selected_indices = np.random.choice(list(range(len(terminal_node_ids))),
                                size=n,
                                replace=False,
                                p=terminal_node_probs)
    selected_node_ids = [terminal_node_ids[idx] for idx in selected_indices]
    return selected_node_ids

class WrappedPrioritizedReplayBuffer(PrioritizedReplayBuffer):

    def __init__(
        self,
        capacity=None,
        alpha=0.6,
        beta0=0.4,
        betasteps=2e5,
        eps=0.01,
        normalize_by_max=True,
        error_min=0,
        error_max=1,
        num_steps=1,
    ):
      super().__init__(capacity,
              alpha,
              beta0,
              betasteps,
              eps,
              normalize_by_max,
              error_min,
              error_max,
              num_steps)
      self.add_count = 0

    def append(
        self,
        state,
        action,
        reward,
        next_state=None,
        next_action=None,
        is_state_terminal=False,
        env_id=0,
        **kwargs
    ):
      super().append(state, action, reward, next_state, next_action, is_state_terminal, **kwargs)
      self.add_count += 1

@gin.configurable()
class EfficientReverseSweepReplayBuffer(WrappedPrioritizedReplayBuffer):
  """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.
  Usage:
    To add a transition:  call the add function.
    To sample a batch:    Construct operations that depend on any of the
                          tensors is the transition dictionary. Every sess.run
                          that requires any of these tensors will sample a new
                          transition.
  """

  def __init__(self,
            obs_space,
            capacity=None,
            alpha=0.6,
            beta0=0.4,
            betasteps=2e5,
            eps=0.01,
            normalize_by_max=True,
            error_min=0,
            error_max=1,
            num_steps=1,
            node_phi_cls=RandomProjectionEncoder, # Mapping from state to node
            transition_id_sample_ratio=1.0, # Number of real transitions sampled from an edge      
            max_predecessor_expand_per_node=3,
            graph_searcher_reset_type='episode', # Reset method for graph searcher
            n_initial_nodes=4, # Number of initial nodes per sweep
            reward_prioritize_temperature=0.1,
            graph_capacity=1000000,
            terminal_state_criteria='game_over-or-reset',
            sampling_strategy='backtrack', # backtrack, backtrack-iso-0.5, backtrack-er-0.5
            bias_correction='none',
            terminal_node_remove_timing='append',
            debug_options=[''],
      ):
      super().__init__(capacity,
            alpha,
            beta0,
            betasteps,
            eps,
            normalize_by_max,
            error_min,
            error_max,
            num_steps)
      self.obs_space = obs_space      
      self.graph_searcher_reset_type = graph_searcher_reset_type
      self.n_initial_nodes = n_initial_nodes
      self.graph_capacity = graph_capacity
      self.reward_prioritize_temperature = reward_prioritize_temperature
      self.terminal_state_criteria = terminal_state_criteria
      self.bias_correction = bias_correction
      self.terminal_node_remove_timing = terminal_node_remove_timing
      self.sampling_strategy = sampling_strategy
      self.transition_id_sample_ratio = transition_id_sample_ratio
      self.max_predecessor_expand_per_node = max_predecessor_expand_per_node
      self._debug_options = debug_options

      self.node_phi = node_phi_cls(obs_space.shape)
      self.graph = EfficientGraph(graph_capacity)
      self.graph_searcher = EfficientBreadthFirstGraphSearcher(graph=self.graph,
                              predecessor_expand_ratio=self.transition_id_sample_ratio,
                              max_predecessor_expand_per_node=self.max_predecessor_expand_per_node,
                              check_transition_is_valid=lambda tid: self._is_transition_id_valid(tid),
                              need_propagate_error=(self.graph_searcher_reset_type.startswith('cp-state')),
                              need_propagate_sampling_prob=self.bias_correction == 'importance',
                              need_keep_removed_terminal_nodes=self.terminal_node_remove_timing == 'sample')    

      # Store terminal node at each episode
      self.sample_queue = deque()
      if self.bias_correction == 'importance':
        self.sample_probs_queue = deque()
      else:
        self.sample_probs_queue = None

      if self.graph_searcher_reset_type == 'nu-episode':
        self.terminal_set = TerminalNodeList()
      elif self.graph_searcher_reset_type == 'rp-episode':
        self.terminal_set = RewardPrioritizedTerminalSet(self.reward_prioritize_temperature)
      else:
        self.terminal_set = TerminalSet()

      self._accumulated_reward = 0
      self._n_fallbacks = 0
      self._n_uniform_sampling = 0
      self._n_resets = 0

      # To debugging
      # To use this, make sure the capacity is unbounded
      self._debug_replay_indices = False
      if 'replay_indices' in debug_options:
        self._replay_indices_counts = np.zeros(capacity)
        self._debug_replay_indices = True
      
      self._debug_profile = False
      if 'profile' in debug_options:
        self._debug_profile = True
      
      self._debug_graph_searcher = False
      if 'debug_graph_searcher' in debug_options:
        self._debug_graph_searcher = True

      # To benchmark each operation
      self._add_time_stat = RunningStat()
      self._sample_time_stat = RunningStat()
      self._reset_time_stat = RunningStat()
      self._num_node_expanded_per_sweep = IntRunningStat()
      self._num_transitions_used_per_sweep = IntRunningStat()

      logging.info(
        'Creating a %s replay wrapper with the following parameters:',
        self.__class__.__name__)
      logging.info('\t replay_capacity: %s', str(self.capacity))
      logging.info('\t graph_searcher_reset_type: %s', str(self.graph_searcher_reset_type))
      logging.info('\t bias_correction: %s', str(self.bias_correction))
      logging.info('\t terminal_state_criteria: %s', str(self.terminal_state_criteria))
      logging.info('\t n_initial_nodes: %s', str(self.n_initial_nodes))
      logging.info('\t reward_prioritize_temperature: %s', str(self.reward_prioritize_temperature))
      logging.info('\t transition_id_sample_ratio: {}'.format(transition_id_sample_ratio))
      logging.info('\t max_predecessor_expand_per_node: {}'.format(max_predecessor_expand_per_node))
      logging.info('\t debug_options: {}'.format(debug_options))

  def save(self, checkpoint_dir, iteration_number):
    super().save(checkpoint_dir, iteration_number)
    if 'save-graph' in self._debug_options:
      logging.info('Save the EfficientReverseSweepReplayWrapper')
      with open(os.path.join(checkpoint_dir, 
                'node_phi-{}'.format(iteration_number)), 'wb') as f:
        dill.dump(self.node_phi, f)
      with open(os.path.join(checkpoint_dir, 
                'graph-{}'.format(iteration_number)), 'wb') as f:
        dill.dump(self.graph, f)
      # with open('graph_searcher-{}'.format(iteration_number), 'wb') as f:
      #   dill.dump(self.graph_searcher, f)
      # with open('sample_queue-{}'.format(iteration_number), 'wb') as f:
      #   dill.dump(self.sample_queue, f)    
      with open(os.path.join(checkpoint_dir, 
                'termainal_set-{}'.format(iteration_number)), 'wb') as f:
        dill.dump(self.terminal_set, f)

  def load(self, checkpoint_dir, suffix):
    super().load(checkpoint_dir, suffix)
    if 'save-graph' in self._debug_options:
      logging.info('Load the EfficientReverseSweepReplayWrapper')
      iteration_number = suffix
      with open(os.path.join(checkpoint_dir, 
                'node_phi-{}'.format(iteration_number), 'rb')) as f:
        self.node_phi = dill.load(self.node_phi, f)
      with open(os.path.join(checkpoint_dir, 
                'graph-{}'.format(iteration_number)), 'rb') as f:
        self.graph = dill.load(self.graph, f)
      with open(os.path.join(checkpoint_dir, 
                'termainal_set-{}'.format(iteration_number)), 'rb') as f:
        self.terminal_set = dill.load(self.terminal_set, f)
      self.graph_searcher = EfficientBreadthFirstGraphSearcher(graph=self.graph,
                                predecessor_expand_ratio=self.transition_id_sample_ratio,
                                max_predecessor_expand_per_node=self.max_predecessor_expand_per_node,
                                check_transition_is_valid=lambda tid: self._is_transition_id_valid(tid),
                                need_propagate_error=(self.graph_searcher_reset_type.startswith('cp-state')))  

  def reset_time_stat(self):
    self._add_time_stat.reset()
    self._sample_time_stat.reset()
    self._reset_time_stat.reset()
    self._n_fallbacks = 0
    self._n_uniform_sampling = 0

  def get_add_time_stat(self):
    return self._add_time_stat

  def get_sample_time_stat(self):
    return self._sample_time_stat

  def get_statistics(self):
    if self._debug_profile:
      stat = {'num_nodes': len(self.graph),
              **self.graph_searcher.graph.get_node_stat_dict(),           
              'num_terminals': len(self.terminal_set),
              'num_fallback': self._n_fallbacks,
              'num_uniform_sample': self._n_uniform_sampling,
              'num_resets': self._n_resets,
              **self._num_node_expanded_per_sweep.get_dict('num_nodes_expanded_per_sweep'),
              **self._num_transitions_used_per_sweep.get_dict('num_transitions_used_per_sweep'),
              **self._add_time_stat.get_dict('_add_time'),
              **self._reset_time_stat.get_dict('_reset_time'),
              **self._sample_time_stat.get_dict('_sample_time'),                   
            }

      if self._debug_replay_indices:
        T = int(self.add_count)
        stat['max_replay'] = np.max(self._replay_indices_counts[:T])
        stat['median_replay'] = np.median(self._replay_indices_counts[:T])
        stat['min_replay'] = np.min(self._replay_indices_counts[:T])
    
      return list(stat.items())
    elif self._debug_graph_searcher:
      return list(self.graph_searcher.get_statistics().items())
    else:
      return []
    
  def _update_terminal_node_probs(self):
    self.terminal_set.update_node_probs()

  def _reset_graph_by_random_node(self):
    n_nodes = len(self.graph.nodes.keys())
    n = min(self.n_initial_nodes, n_nodes)
    if self.bias_correction == 'importance':
      terminal_node_probs = [1. / n_nodes] * n
    else:
      terminal_node_probs = None
    self.graph_searcher.reset(random.sample(list(self.graph.nodes.keys()), n), 
                initial_nodes_probs=terminal_node_probs)

  def _reset_graph_by_episodic_terminal_states(self):
    n = min(self.n_initial_nodes, len(self.terminal_set))

    # If there is no terminal states, fall back to use
    if n == 0:
      self._reset_graph_by_random_node()
      return

    if self.bias_correction == 'importance':
      terminal_node_probs = [1. / len(self.terminal_set)] * n
    else:
      terminal_node_probs = None

    self.graph_searcher.reset(self.terminal_set.sample(n), 
                initial_nodes_probs=terminal_node_probs)

  def _reset_graph_by_return_prioritized_episodic_terminal_states(self):
    n = min(self.n_initial_nodes, len(self.terminal_set))

    if n == 0:
      self._reset_graph_by_random_node()
      return

    self.graph_searcher.reset(self.terminal_set.sample(n))

  def _reset_graph_by_tderror_prioritized_states(self, n_non_terminal):
    assert int(self.add_count) >= n_non_terminal
    n_terminal = min(self.n_initial_nodes - n_non_terminal, len(self.terminal_set))
    terminal_start_node_ids = self.terminal_set.sample(n_terminal)
    
    final_n_non_terminal = (self.n_initial_nodes - n_terminal)
    non_terminal_starts = super().sample(final_n_non_terminal)
    non_terminal_starts = [e[0]['state'] for e in non_terminal_starts]
    self.memory.flag_wait_priority = False    
    # Get the last stacked elements, [0] is observation
    non_terminal_start_node_ids = [self.node_phi(non_terminal_starts[i]) for i in range(final_n_non_terminal)]

    self.graph_searcher.reset(terminal_start_node_ids + non_terminal_start_node_ids)
      
  def _reset_graph_by_successor_certainty_states(self):
    _, pr_temp_str = self.graph_searcher_reset_type.split('_')
    pr_temp = float(pr_temp_str)
    node_ids = [node_id for node_id in self.graph.nodes.keys()]
    node_successor_losses = [-np.sqrt(node.running_successor_loss) * pr_temp for node in self.graph.nodes.values()]
    node_probs = softmax(node_successor_losses)
    selected_indices = np.random.choice(list(range(len(node_ids))), 
                                size=self.n_initial_nodes, 
                                replace=True,
                                p=node_probs)
    selected_node_ids = [node_ids[idx] for idx in selected_indices]
    self.graph_searcher.reset(selected_node_ids)

  def _reset_graph_searcher(self):
    with timed_stat(self._reset_time_stat):
      # Book the statistics
      sweep_stat = self.graph_searcher.get_statistics()
      self._num_node_expanded_per_sweep.update(sweep_stat['num_node_expanded'])
      self._num_transitions_used_per_sweep.update(sweep_stat['num_transitions_used'])

      # episode: choose terminal states from episodes as initial nodes
      if self.graph_searcher_reset_type == 'episode' or self.graph_searcher_reset_type == 'nu-episode':
        self._reset_graph_by_episodic_terminal_states()
      # rp-episode: prioritize terminal states at high-value episodes
      elif self.graph_searcher_reset_type == 'rp-episode':
        self._reset_graph_by_return_prioritized_episodic_terminal_states()
      # tdp-state: td error prioritized states as starts of backtracking
      elif self.graph_searcher_reset_type.startswith('tdp-state'):
        non_terminal_ratio = float(self.graph_searcher_reset_type.split('_')[1])
        n_non_terminal = int(self.n_initial_nodes * non_terminal_ratio)
        assert n_non_terminal <= self.n_initial_nodes
        self._reset_graph_by_tderror_prioritized_states(n_non_terminal)
      # cp-state: certainty prioritized states as starts of backtracking
      elif self.graph_searcher_reset_type.startswith('cp-state'):
        self._reset_graph_by_successor_certainty_states()
      self._n_resets += 1    
  
  def _get_shifted_index(self, tid):
    if self.add_count < self.capacity:
      return tid
    min_valid_idx = self.add_count - self.capacity
    offset_min_valid_idx = tid - min_valid_idx
    return offset_min_valid_idx

  def _is_transition_id_valid(self, tid):
    shifted_tid = self._get_shifted_index(tid)
    return shifted_tid >= 0
  
  def _is_transition_id_outdated(self, tid):
    return self._get_shifted_index(tid) < 0

  def _fetch_samples(self, batch_size):
    max_retries = 3
    while len(self.sample_queue) < batch_size:
      sample_indices, sample_probs, need_reset = self.graph_searcher.step(current_timestep=self.add_count,
                            valid_id_interval=self.capacity)  
      self.sample_queue.extend(sample_indices)
      if self.bias_correction == 'importance':
        assert sample_probs is not None, 'importance bias-correction needs sampling probs returned by the graph searcher.'
        self.sample_probs_queue.extend(sample_probs)

      if need_reset:
        self._reset_graph_searcher()

      # No new elements added and the curren sample queue is empty
      if len(sample_indices) == 0 and len(self.sample_queue) == 0:
        return False
    
    # After many graph_searcher steps, we will know some outdated nodes. If some are terminal states, then kill them
    if self.terminal_node_remove_timing == 'sample':
      for terminal_node_to_delete in self.graph_searcher.terminal_nodes_to_delete:
        self.terminal_set.remove(terminal_node_to_delete)
        logging.info('Remove terminal node {} due to empty transition list'.format(terminal_node_to_delete))
      self.graph_searcher.clear_nodes_to_delete()

    return True

  def _sample_from_backtrack(self, n):
    with timed_stat(self._sample_time_stat):      
      batch_size = n
   
      sample_indices = list()
      sample_probs = list()
      while len(sample_indices) < batch_size:
        if len(self.sample_queue) < batch_size:
          is_success_to_fetch = self._fetch_samples(batch_size)
          if not is_success_to_fetch:
            sample_indices, probabilities, min_prob = self.memory._sample_indices_and_probabilities(n, uniform_ratio=0)
            sample_weights = self.weights_from_probabilities(probabilities, min_prob)
            return sample_indices, sample_weights

        transition_idx = self.sample_queue.popleft()
        if self._is_transition_id_valid(transition_idx):
          sample_indices.append(self._get_shifted_index(transition_idx))
        if self.bias_correction == "importance":
          sample_probs.append(self.sample_probs_queue.popleft())
        else:
          sample_probs.append(1.0)
    
    if self._debug_replay_indices:
      for idx in sample_indices:
        self._replay_indices_counts[idx] += 1
    
    if self.bias_correction == 'importance':
      min_prob = min(sample_probs)
      # logging.info("Sample probs: {}".format(sample_probs))
      sample_weights = self.weights_from_probabilities(sample_probs, min_prob)
    else:
      sample_weights = sample_probs

    return sample_indices, sample_weights

  def _sample_from_iso(self, n):
    iso_list = list(self.graph_searcher.last_non_expanded_node_set)
    
    if len(iso_list) == 0:
      return []

    # print('Sample from isolated nodes: n={}, len={}'.format(n, len(iso_list)))
    n = min(n, len(iso_list))    
    sampled_indices = []
    while len(sampled_indices) < n:
      node_id = random.sample(iso_list, 1)[0]   
      node = self.graph[node_id]

      if len(node) == 0:
        continue

      pred_transition_ids, pred_node_ids = node.sample_predecssors(self.transition_id_sample_ratio,
                                                        max_per_node=self.max_predecessor_expand_per_node)
      sampled_indices += pred_transition_ids

    if len(sampled_indices) > n:
      return sampled_indices[:n]
    return sampled_indices                                                

  def sample(self, n):
    if self.sampling_strategy == 'backtrack':
      sample_indices, weights = self._sample_from_backtrack(n)
    elif self.sampling_strategy.startswith('backtrack-iso-mixin'):
      assert 0, "iso is abandonded"
      _, iso_ratio = self.sampling_strategy.split('_')
      n_iso = int(n * float(iso_ratio))
      iso_sample_indices = self._sample_from_iso(n_iso)
      n_backtrack = n - len(iso_sample_indices)
      backtrack_sample_indices, backtrack_sample_weights = self._sample_from_backtrack(n_backtrack)
      sample_indices = iso_sample_indices + backtrack_sample_indices

    elif self.sampling_strategy.startswith('backtrack-iso'):
      assert 0, "iso is abandonded"
      _, iso_prob = self.sampling_strategy.split('_')
      if np.random.random() < float(iso_prob):
        # print('Use sample from isolated nodes')
        sample_indices = self._sample_from_iso(n)
        if len(sample_indices) == 0:
          # print('No isolated nodes, use backtracking')
          sample_indices = self._sample_from_backtrack(n)
      else:
        # print('Use sample from backtracking nodes')
        sample_indices = self._sample_from_backtrack(n)

    elif self.sampling_strategy.startswith('backtrack-er-mixin'):
      _, er_ratio = self.sampling_strategy.split('_')
      n_er = int(n * float(er_ratio))
      er_sampled_indices, probabilities, min_prob = self.memory._sample_indices_and_probabilities(n_er, uniform_ratio=0)
      er_weights = self.weights_from_probabilities(probabilities, min_prob)
      n_bt = n - n_er
      bt_sampled_indices, bt_weights = self._sample_from_backtrack(n_bt)
      sample_indices = er_sampled_indices + bt_sampled_indices
      weights = er_weights + bt_weights
      # print('Mixin ER {} and {}'.format(n_er, n_bt))
    elif self.sampling_strategy.startswith('backtrack-er'):
      _, er_prob = self.sampling_strategy.split('_')
      if np.random.random() < float(er_prob):
        # print('Use sample from ER')
        sample_indices, probabilities, min_prob = self.memory._sample_indices_and_probabilities(n, uniform_ratio=0)
        weights = self.weights_from_probabilities(probabilities, min_prob)
      else:
        # print('Use sample from BT')
        sample_indices, weights = self._sample_from_backtrack(n)
    else:
      raise NotImplemented()

    sampled = [self.memory.data[i] for i in sample_indices]

    # For PER buffer
    self.memory.sampled_indices = sample_indices
    self.memory.flag_wait_priority = True

    if self.bias_correction == 'importance' or \
        self.sampling_strategy.startswith('backtrack-er') or \
        self.graph_searcher_reset_type.startswith('cp-state') or \
        self.graph_searcher_reset_type.startswith('tdp-state'):      
      for e, w in zip(sampled, weights):
          e[0]["weight"] = w

    return sampled

  def update_errors(self, errors):
    if self.graph_searcher_reset_type.startswith('cp-state'):
      indices = self.memory.sampled_indices
      losses = errors

      for index, loss in zip(indices, losses):
        transition = self.memory.data[index]
        s = transition[0]['state']
        node_id = self.node_phi(s)
        if node_id in self.graph.nodes:
          self.graph[node_id].update_running_loss(loss)
        else:
          logging.warning('Warning: node id {} is missing in the graph. Check the node encoder'.format(node_id))

    super().update_errors(errors)

  def append(
        self,
        state,
        action,
        reward,
        next_state=None,
        next_action=None,
        is_state_terminal=False,
        env_id=0,
        **kwargs
    ):
    with timed_stat(self._add_time_stat):
      buffer_index = int(self.add_count)
      node_id = self.node_phi(state)
    
      next_node_id = self.node_phi(next_state)    
      
      if self.terminal_state_criteria == 'game_over':
        # Timeout can be done=True && reset=True, but the game_over must done=True && reset=False
        is_time_to_add_terminal_state = is_state_terminal and (not kwargs['reset'])
      elif self.terminal_state_criteria == 'game_over-or-reset':
        # Either reset or done are terminal states
        is_time_to_add_terminal_state = is_state_terminal or kwargs['reset']
      else:
        raise NotImplemented()

      pred_node, suc_node = self.graph.add_transition_id(s_t_node_id=node_id,
                                    s_tp1_node_id=next_node_id,
                                    transition_id=buffer_index,
                                    is_terminal=is_time_to_add_terminal_state)

      # if is_state_terminal or kwargs['reset']:
      #   print('done: {}, reset:{}, is_terminal_state: {}, num_terminals: {}'.format(is_state_terminal, kwargs['reset'], is_time_to_add_terminal_state, len(self.terminal_set)))
      
      if is_time_to_add_terminal_state:
        # TODO: usually if done_t==True, s_{t+1} is all-zeros. We cannot distinguish terminal states by that
        # Maybe we should use `node_id` as terminl node
        self.terminal_set.append(next_node_id, buffer_index, self._accumulated_reward)        
        self._accumulated_reward = 0
        # Will make update if the terminal set has a method
        if self.terminal_node_remove_timing == 'append':
          self.terminal_set.remove_outdated_nodes(buffer_index, self.capacity)
        self._update_terminal_node_probs()
      else:        
        self._accumulated_reward += reward
      
      super().append(state, action, reward, next_state, next_action, is_state_terminal, **kwargs)

