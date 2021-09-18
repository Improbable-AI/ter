from hash_graph import HashGraph, BFS

import numpy as np

class RandomProjectionEncoder:
    
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.phi = np.random.normal(loc=0, scale=1./ np.sqrt(latent_dim), 
                      size=(latent_dim, input_dim))
        
    def __call__(self, states: np.ndarray):
        x = states.reshape(-1, self.input_dim) if len(states.shape) >= 2 else states
        return tuple(*np.dot(x, self.phi.T))

if __name__ == '__main__':
    from visualize import plot_graph
    import matplotlib.pyplot as plt

    '''
    Collect transitions data from minigrid
    '''
    import gym
    import gym_minigrid
    from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

    env = gym.make('MiniGrid-Empty-5x5-v0')
    env = ImgObsWrapper(RGBImgObsWrapper(env))

    replay_buffer = []
    
    obs = env.reset()
    done = False

    while not done:
      action = env.action_space.sample()
      obs_next, rew, done, info = env.step(action)

      replay_buffer.append((obs.copy(), action, obs_next.copy(), done))

      obs = obs_next
    
    '''
    Build a graph from these data
    '''
    phi = RandomProjectionEncoder(np.prod(obs.shape), 2)
    g = HashGraph()
    for idx, transition in enumerate(replay_buffer):
      g.insert(phi(transition[0]), phi(transition[2]), idx)
    print('nV:', g.nE)
    print('nE:', g.nV)
    print('n_transitions:', len(replay_buffer))

    # Perform BFS over the graph    
    searcher = BFS()
    
    # For example, we could select the last state at an episode as the destination
    sample_obs = replay_buffer[-1][0]
    dst_vertex_id = phi(sample_obs)
    shortest_paths = (searcher.search(g, dst_vertex_id, True))

    # Retrieve data from the graph by the shortest path in a reverse order
    # We should see the numbers we insert when creating the graph
    for path_src, path in shortest_paths.items():
      print(f'The "reversed" shortest path from {path_src} to {dst_vertex_id} (len={len(path)}):')
      for src, dst in reversed(path):
        transition_indices = g.edge(src, dst)
        print(f'\t{src}->{dst}:', transition_indices)