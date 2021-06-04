from absl import logging

import gin
import bisect
import numpy as np
import random

from collections import defaultdict, deque

class EfficientNode(object):
  
  def __init__(self, discount_factor=0.5, is_terminal=False):
    self.predecessor_transition_ids = list()
    self.predecessor_node_ids = list()
    self.running_loss = 0 # Value prediction error at s
    self.running_successor_loss = 0 # Value prediction errors of s'
    self.is_terminal = is_terminal
    self.discount_factor = discount_factor
  
  def update_running_loss(self, loss):
    self.running_loss = loss + self.discount_factor * self.running_loss
  
  def update_running_successor_loss(self, loss):
    self.running_successor_loss = loss + self.discount_factor * self.running_successor_loss

  def add_transition_id(self, transition_id, node_id):
    self.predecessor_transition_ids.append(int(transition_id))
    self.predecessor_node_ids.append(node_id)

  def remove_outdated_transition_ids(self, min_valid_timestep):   
    min_valid_idx = bisect.bisect_left(self.predecessor_transition_ids, min_valid_timestep)
    self.predecessor_transition_ids = self.predecessor_transition_ids[min_valid_idx:]
    self.predecessor_node_ids = self.predecessor_node_ids[min_valid_idx:]    

  def sample_predecssors(self, ratio, max_per_node=int(1e6)):
    assert len(self.predecessor_transition_ids) > 0

    size = len(self.predecessor_transition_ids)
    sample_size = min(max(int(size * ratio), 1), max_per_node)
    sampled_indices = random.sample(list(range(size)), sample_size)
    sampled_transition_ids = [self.predecessor_transition_ids[i] for i in sampled_indices]
    sampled_node_ids = [self.predecessor_node_ids[i] for i in sampled_indices]

    return sampled_transition_ids, sampled_node_ids

  def predecessors(self):
    return self.predecessor_transition_ids, self.predecessor_node_ids

  def __len__(self):
    return len(self.predecessor_node_ids)

  def num_predecessors(self):
    raise NotImplemented()

class DictEfficientNode(object):
  def __init__(self, discount_factor=0.5, is_terminal=False):
    self.running_loss = 0 # Value prediction error at s
    self.running_successor_loss = 0 # Value prediction errors of s'
    self.discount_factor = discount_factor
    self.predecessors_dict = defaultdict(list)
    self.successors_probs = dict()
    self.is_terminal = is_terminal
    self.num_transitions = 0

  def clear_nodes_to_remove(self):
    self.nodes_to_remove = []

  def prob(self):
    return min(sum(self.successors_probs.values()), 1.0)

  def update_successor_probs(self, successor_node_id, successor_prob):
    self.successors_probs[successor_node_id] = successor_prob

  def update_running_loss(self, loss):
    self.running_loss = loss + self.discount_factor * self.running_loss
  
  def update_running_successor_loss(self, loss):
    self.running_successor_loss = loss + self.discount_factor * self.running_successor_loss
  
  def add_transition_id(self, transition_id, node_id):
    self.predecessors_dict[node_id].append(int(transition_id))
    self.num_transitions += 1

  def remove_outdated_transition_ids(self, min_valid_timestep):
    nodes_to_remove = []
    for pred_id, predecessor_transition_ids in self.predecessors_dict.items():
      before_len = len(predecessor_transition_ids)
      min_valid_idx = bisect.bisect_left(predecessor_transition_ids, min_valid_timestep)
      new_transition_ids = predecessor_transition_ids[min_valid_idx:]
      after_len = len(new_transition_ids)

      self.num_transitions -= (before_len - after_len)
      assert self.num_transitions >= 0

      self.predecessors_dict[pred_id] = new_transition_ids
      if len(new_transition_ids) == 0:
        # del self.predecessors_dict[pred_id]
        nodes_to_remove.append(pred_id)
        
    for node_id in nodes_to_remove:
      del self.predecessors_dict[node_id]

  def sample_predecssors(self, ratio, max_per_node=int(1e6)):
    assert self.num_transitions > 0
  
    pred_node_ids = list(self.predecessors_dict.keys())
    n_preds = len(pred_node_ids)
    sample_size = min(max(int(n_preds * ratio), 1), max_per_node)
    shuffled_indices = random.sample(range(n_preds), n_preds)

    sampled_transition_ids = []
    sampled_node_ids = []
    it = 0
    while len(sampled_node_ids) < sample_size:
      sampled_pred_node_id = pred_node_ids[shuffled_indices[it]]
      transition_ids = self.predecessors_dict[sampled_pred_node_id]
      if len(transition_ids) > 0:
        sampled_transition_id = random.sample(transition_ids, 1)[0]
        sampled_transition_ids.append(sampled_transition_id)
        sampled_node_ids.append(sampled_pred_node_id)
      it += 1

    return sampled_transition_ids, sampled_node_ids

  def __len__(self):
    return self.num_transitions

  def num_predecessors(self):
    return len(self.predecessors_dict)

  def predecessors(self):
    transition_ids = list(reduce(lambda x, y: x + y, [ts for ts in self.predecessors_dict.values()]))
    node_ids = list(reduce(lambda x, y: x + y, [[node_id for i in range(len(ts))] for node_id, ts in self.predecessors_dict.items()]))
    return transition_ids, node_ids
    
@gin.configurable()
class EfficientGraph(object):
    
    '''
    - Only remove node when it's empty
    '''

    def __init__(self, node_capacity, node_cls=DictEfficientNode):
      self.node_capacity = node_capacity
      self.node_cls = node_cls
      self.nodes = defaultdict(node_cls)

      logging.info('Node class: {}'.format(node_cls))

    def get_node_stat_dict(self):
      node_sizes = [len(node) for node in self.nodes.values()]
      if len(node_sizes) > 0:
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        mean_size = sum(node_sizes) / len(self.nodes)
        median_size = np.median(node_sizes)
      else:
        max_size = 0
        min_size = 0
        mean_size = 0
        median_size = 0
      # unique, counts = np.unique(node_sizes, return_counts=True)
      return {'num_preds_per_node(max)': max_size,
              'num_preds_per_node(min)': min_size,
              'num_preds_per_node(mean)': mean_size,
              'num_preds_per_node(median)': median_size,
              # 'num_preds_per_node(dist)': list(zip(unique, counts))
              }
      
    def add_transition_id(self, s_t_node_id, s_tp1_node_id, transition_id, is_terminal):

      suc_node = self.nodes[s_tp1_node_id]
      suc_node.add_transition_id(transition_id, s_t_node_id)
      suc_node.is_terminal = is_terminal
      pred_node = self.nodes[s_t_node_id]      
    
      return pred_node, suc_node

    def delete_node(self, node_id):
      del self.nodes[node_id]

    def __len__(self):
      return len(self.nodes)

    def __getitem__(self, key):
      if key not in self.nodes:
        return None
      return self.nodes[key]

    def __iter__(self):
      for node_id, node in self.nodes.items():
        yield node_id, node

    def __str__(self):
      ret = ''
      for node_id, node in self.nodes.items():
        ret += '\tNode: {}\n'.format(node_id)              
        predecessor_transition_ids, predecessor_node_ids = node.predecessors()
        for tid, nid in zip(predecessor_transition_ids, predecessor_node_ids):
            ret += '\t\tTid={}, Nid={}\n'.format(tid, nid)
      return ret

class EfficientBreadthFirstGraphSearcher(object):

  def __init__(self, graph, 
      predecessor_expand_ratio, # Ratio of predecessor to expand
      max_predecessor_expand_per_node, # To prevent repeated nodes
      check_transition_is_valid,
      need_propagate_error=False,
      need_propagate_sampling_prob=False,
      need_keep_removed_terminal_nodes=False):
    self.graph = graph
    self.predecessor_expand_ratio = predecessor_expand_ratio
    self.max_predecessor_expand_per_node = max_predecessor_expand_per_node    
    self.check_transition_is_valid = check_transition_is_valid
    self.need_propagate_error = need_propagate_error
    self.need_propagate_sampling_prob = need_propagate_sampling_prob
    self.need_keep_removed_terminal_nodes = need_keep_removed_terminal_nodes

    self.nodes_to_expand_queue = deque()
    if self.need_propagate_sampling_prob:
      self.nodes_to_expand_probs_queue = deque()
    self.expanded_node_set = set()
    self.last_non_expanded_node_set = set()
    self.visited_edge_set = set()
    self.terminal_nodes_to_delete = []
    
    self._num_node_expanded = 0
    self._num_transitions_used = 0
    self._pred_probs = deque(maxlen=1000)

  def clear_nodes_to_delete(self):
    self.terminal_nodes_to_delete = []

  def get_statistics(self):
    stats = {'num_node_expanded': self._num_node_expanded,
            'num_transitions_used': self._num_transitions_used}

    if self.need_propagate_sampling_prob:
      if len(self._pred_probs) == 0:
        stats['min_pred_prob'] = np.inf
        stats['mean_pred_prob'] = 0
        stats['median_pred_prob'] = 0
        stats['max_pred_prob'] = -np.inf 
      else:
        stats['min_pred_prob'] = np.min(self._pred_probs)
        stats['mean_pred_prob'] = np.mean(self._pred_probs)
        stats['median_pred_prob'] = np.median(self._pred_probs)
        stats['max_pred_prob'] = np.max(self._pred_probs)      

    return stats

  def _update_last_non_expanded_node_set(self):
    all_node_ids = set(self.graph.nodes.keys())
    self.last_non_expanded_node_set = all_node_ids - self.expanded_node_set

  def reset(self, initial_nodes, initial_nodes_probs=None):
    self._update_last_non_expanded_node_set()
    self.expanded_node_set.clear()
    self.visited_edge_set.clear()
    self.nodes_to_expand_queue.clear()
    self._num_node_expanded = 0
    self._num_transitions_used = 0
      
    for idx, node_id in enumerate(initial_nodes):
      self.nodes_to_expand_queue.append(node_id)
      if self.need_propagate_sampling_prob:
        assert initial_nodes_probs is not None, "Propagating sampling probs needs reset node probs"
        if node_id in self.graph.nodes:
          self.graph[node_id].update_successor_probs(None, initial_nodes_probs[idx])

  def _pop_node_to_expand(self, min_valid_timestep):
    # Expand nodes until a non-zero node is found
    while len(self.nodes_to_expand_queue) > 0:     
      node_to_expand_id = self.nodes_to_expand_queue.popleft()

      # Before accessing the node, we have to check
      # 1. Prevent expanding the same node again
      if node_to_expand_id in self.expanded_node_set:     
        continue
      # 2. Prevent outdated node
      if node_to_expand_id not in self.graph.nodes:
        continue

      node_to_expand = self.graph[node_to_expand_id]

      # After accessing the node, we have to do
      self._num_node_expanded += 1
      # 1. Mark as expanded (checked)
      self.expanded_node_set.add(node_to_expand_id)
      # 2. Remove outdated transitions
      node_to_expand.remove_outdated_transition_ids(min_valid_timestep)
      # 3. If the node is empty, then remove it
      if len(node_to_expand) == 0:
        self.graph.delete_node(node_to_expand_id)
        # 3.1. Keep the node if it's a terminal node so that we can remove it outside
        if self.need_keep_removed_terminal_nodes and node_to_expand.is_terminal:
          self.terminal_nodes_to_delete.append(node_to_expand_id)
        continue
      return node_to_expand_id, node_to_expand
    return None, None

  def _push_node_to_expand(self, node_ids):
    self.nodes_to_expand_queue.extend(node_ids)

  def step(self, current_timestep, valid_id_interval):
    selected_transition_ids = list()
    selected_transition_probs = list() if self.need_propagate_sampling_prob else None

    while len(selected_transition_ids) == 0:
      node_to_expand_id, node_to_expand = self._pop_node_to_expand(current_timestep - valid_id_interval)

      # Reset is needed is there is no remaining nodes
      if node_to_expand is None:
        return selected_transition_ids, selected_transition_probs, True
            
      # We've ensure that there is no outdated transitions in `pop`
      pred_transition_ids, pred_node_ids = self._expand_node(node_to_expand)      
      selected_transition_ids.extend(pred_transition_ids)      

      # Propagate sampling prob
      if self.need_propagate_sampling_prob:
        branching_prob = min(self.max_predecessor_expand_per_node, node_to_expand.num_predecessors()) / node_to_expand.num_predecessors()
        suc_prob = node_to_expand.prob() * branching_prob       
        for pred_node_id in pred_node_ids:
          pred_node = self.graph[pred_node_id]
          if pred_node is not None:
            pred_node.update_successor_probs(node_to_expand_id, suc_prob)
            pred_prob = pred_node.prob()            
          else:
            pred_prob = suc_prob # TODO: find out why there is missing nodes
          self._pred_probs.append(pred_prob)
          selected_transition_probs.append(pred_prob)  

      self._push_node_to_expand(pred_node_ids)

    self._num_transitions_used += len(selected_transition_ids)

    return selected_transition_ids, selected_transition_probs, False

  def _expand_node(self, node):
    pred_transition_ids, pred_node_ids = node.sample_predecssors(self.predecessor_expand_ratio,
                                                        max_per_node=self.max_predecessor_expand_per_node)
    # NOTE: must use list
    # Check if the transition index valid
    filtered = list(filter(lambda x: self.check_transition_is_valid(x[0]), 
                      zip(pred_transition_ids, pred_node_ids)))
    filtered_pred_transition_ids = [f[0] for f in filtered]
    filtered_pred_node_ids = [f[1] for f in filtered]

    # Propagate the successor node's running loss to
    if self.need_propagate_error:
      for pred_node_id in filtered_pred_node_ids:
        self.graph[pred_node_id].update_running_successor_loss(node.running_loss)

    assert len(filtered_pred_transition_ids) == len(filtered_pred_node_ids)
    return filtered_pred_transition_ids, filtered_pred_node_ids

if __name__ == '__main__':
  import gym
  from common import RandomProjectionEncoder, IdentityEncoder

  #env = gym.make('PongNoFrameskip-v4')
  env = gym.make('FrozenLake-v0')
  mem_size = 100
  graph = EfficientGraph(node_capacity=100)
  graph_searcher = EfficientBreadthFirstGraphSearcher(graph, 1.0, 3,
                        lambda tid: True)
  #phi = RandomProjectionEncoder(env.observation_space.shape)
  phi = IdentityEncoder(env.observation_space.shape)
  rbuf = list()
  t = 0
  
  s = env.reset()
  while len(rbuf) < 10000:
    a = env.action_space.sample()
    sp, r, done, _ = env.step(a)
    rbuf.append((s, a, sp, r, done))
    phi_s = phi(s)
    phi_sp = phi(sp)
    graph.add_transition_id(s_t_node_id=phi_s, s_tp1_node_id=phi_sp, transition_id=t)
    t += 1
    s = sp
    if done:
      s = env.reset()
   
  print('Graph stat:', graph.get_node_stat_dict())
  # print('Len:', len(rbuf))
  # for node_id, node in graph:
  #   print('Node_id={}'.format(node_id))
  #   for tid, nid in zip(*node.predecessors()):
  #     if tid < t - mem_size:
  #       transition = '(removed)'
  #     else:
  #       transition = rbuf[tid]
  #     print('\tpred_node_id={}; tid={}; transtion={}'.format(nid, tid, transition))

  graph_searcher.reset([8,])
  for i in range(20):
    batch, need_reset = graph_searcher.step(t, mem_size)
    print(batch)
    if need_reset:
      graph_searcher.reset([8,])
      print('\t\tReset')
      

  
