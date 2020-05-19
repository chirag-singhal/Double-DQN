from collections import deque

import numpy as np

import torch
import torchvision.transforms as T

import gym
from gym.wrappers import AtariPreprocessing


class Game():

    def __init__(self, game_name, frameskip=4, device='cpu', grayscale_obs=True, scale_obs=False):
        self.frameskip = frameskip
        self.device = device

        self.buffer = deque([], self.frameskip)

        self.env = gym.make(game_name)
        self.envWrapped = AtariPreprocessing(self.env, frame_skip=1, grayscale_obs=grayscale_obs, scale_obs=scale_obs)
        self.envWrapped.reset()

        self.n_actions = self.env.action_space.n

        init_screen = self.get_screen()
        # Screen dimension is represented as (CHW) for PyTorch
        self.scr_dims = tuple([self.frameskip] + list(init_screen.shape))

        for _ in range(self.frameskip):
            self.buffer.append(init_screen.clone())

    def get_screen(self):
        screen = self.envWrapped._get_obs()
        screen = torch.from_numpy(screen).to(self.device)
        return screen

    def get_input(self):
        # Each element in buffer is a tensor of 84x84 dimensions.
        # This function returns tensor of 4x84x84 dimensions.
        return torch.stack(tuple(self.buffer), dim=0).type(torch.float)
    
    def get_n_actions(self):
        # return number of actions
        return self.n_actions
    
    def reset_env(self):
        # reset the gym environment
        self.env.reset()

    def get_screen_dims(self):
        # return the screen dimensions
        return self.scr_dims

    def step(self, action):
        reward = 0
        done = False
        for step in range(self.frameskip):
            _, _reward, done, _ = self.envWrapped.step(action)
            self.buffer.append(self.get_screen())
            reward += _reward
            if done:
                break
        return reward, done


# Actions in OpenAI Gym ALE
# -------------------------
# ACTION_MEANING = {
#     0: "NOOP",
#     1: "FIRE",
#     2: "UP",
#     3: "RIGHT",
#     4: "LEFT",
#     5: "DOWN",
#     6: "UPRIGHT",
#     7: "UPLEFT",
#     8: "DOWNRIGHT",
#     9: "DOWNLEFT",
#     10: "UPFIRE",
#     11: "RIGHTFIRE",
#     12: "LEFTFIRE",
#     13: "DOWNFIRE",
#     14: "UPRIGHTFIRE",
#     15: "UPLEFTFIRE",
#     16: "DOWNRIGHTFIRE",
#     17: "DOWNLEFTFIRE",
# }
