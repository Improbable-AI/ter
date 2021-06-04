import json
import os

import numpy as np

import gym
from gym.wrappers import TimeLimit
from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper, FlatObsWrapper

    
class EqualScaleMinigridStepPenaltyReward(gym.Wrapper):
    
    def __init__(self, env, test=False):
      super().__init__(env)
      self.test = test

    def step(self, a):
        obs, rew, done, info = self.env.step(a)
        orig_rew = rew
        rew = self.env.unwrapped.max_steps if rew else -1
        
        # Bad early termination
        if done and orig_rew == 0 and self.env.unwrapped.step_count < self.env.unwrapped.max_steps:          
            rew = -self.env.unwrapped.max_steps
                
        return obs, rew, done, info
    
class MinigridActionCompress(gym.Wrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        print(env_name.split('-', 1))
        domain, task = env_name.split('-', 1)
        if task.startswith('Empty') or task.startswith('FourRooms') \
            or task.startswith('LavaGap') \
            or task.startswith('SimpleCrossing') \
            or task.startswith('LavaCrossing') \
            or task.startswith('LavaCrossing') \
            or task.startswith('Dynamic-Obstacles'):
            self.action_space = gym.spaces.Discrete(3)

class MiniGridImageResize(gym.ObservationWrapper):

    def __init__(self, env, max_size=(40, 40)):
        super().__init__(env)
        self.max_size = max_size
        orig_size = self.observation_space.shape[:2]
        h, w = orig_size

        self.size = orig_size
        self.need_resize = False
        # if h > max_size[0] or w > max_size[0]:
        self.size = max_size
        self.need_resize = True
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.size + (3,), dtype=np.uint8)
        print('MiniGridImageResize: orig_size={} size={}'.format(env.observation_space, self.observation_space))
    
    def observation(self, obs):  
        if self.need_resize:
            obs = cv2.resize(obs,
                    self.size,
                    #interpolation=cv2.INTER_AREA
                    interpolation=cv2.INTER_NEAREST
                    )

            obs = np.asarray(obs, dtype=np.uint8)
        return obs

class SokobanWrapper(gym.Wrapper):
    
    def __init__(self, env, screen_size=(84, 84)):
        super().__init__(env)
        self.screen_size = screen_size
        self.action_space = gym.spaces.Discrete(self.env.action_space.n - 1)
        self.observation_space = gym.spaces.Box(low=0, high=255, 
            shape=self.screen_size + (3,), dtype=np.uint8)

    def _process_obs(self, obs):
        transformed_image = cv2.resize(obs,
                                    self.screen_size,
                                    interpolation=cv2.INTER_NEAREST)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return int_image

    def reset(self):
        image = self.env.reset(render_mode='tiny_rgb_array')
        processed_image = self._process_obs(image)
        return processed_image

    def step(self, action):
        action = action + 1 # To remove noop
        observation, reward, is_terminal, info = self.env.step(action, 'tiny_rgb_array')
        # print('Max steps?', self.environment._check_if_maxsteps(), self.environment.max_steps, self.environment.num_env_steps, is_terminal)
        game_over = info.get("all_boxes_on_target", False) # Only send termination when all boxes are on the targets
        processed_image = self._process_obs(observation)
        return processed_image, reward, game_over, info

def wrap_minigrid(env, env_id, im_size=(40, 40), test=False):
    wrapped_env = MiniGridImageResize(
        MinigridActionCompress(
            EqualScaleMinigridStepPenaltyReward(
                ImgObsWrapper(
                    RGBImgObsWrapper(env)
                ), test=test
            ), env_id
        ), im_size
    )
    return TimeLimit(wrapped_env, env.max_steps)

def wrap_env(env_name, env_wrapper, test=False):
    if env_name.startswith('MiniGrid'):
        env = gym.make(env_name)
        return pfrl.wrappers.CastObservationToFloat32(wrap_minigrid(env, env_name, im_size=(40, 40), test=test))
    elif env_name.startswith('Sokoban'): # Sokoban_10x10_1_120
        import gym_sokoban
        from gym_sokoban.envs.sokoban_env import SokobanEnv
        from gym_sokoban.envs.sokoban_env_pull import PushAndPullSokobanEnv
        size_strs, num_boxes, max_steps = env_name.split('_')[1:]
        sizes = tuple(map(lambda size_str: int(size_str), size_strs.split('x')))
        max_steps = int(max_steps)
        assert sizes[0] >= 5, 'Size must be greater than 4.'

        num_boxes = int(num_boxes)
        # max_steps = 120 * (sizes[0] - 5 + 1) + (num_boxes - 1) * 120
                
        if env_name.startswith('Sokoban-Push'):
            print('Use PushOnly Sokoban, max_steps={}'.format(max_steps))
            env = SokobanWrapper(SokobanEnv(dim_room=sizes, num_boxes=int(num_boxes), max_steps=max_steps))
        else:
            raise NotImplemented()
        return pfrl.wrappers.CastObservationToFloat32(TimeLimit(env, env.max_steps))
    else:
        raise NotImplemented()
