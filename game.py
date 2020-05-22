from collections import deque

import numpy as np

import gym
from gym.wrappers import AtariPreprocessing


class Game():

    def __init__(self, game_name, last_n_frames=4, frameskip=4, grayscale_obs=True, scale_obs=False):
        self.last_n_frames = last_n_frames
        self.frameskip = frameskip

        self.buffer = deque([], self.last_n_frames)

        self.env = gym.make(game_name)

        # Hacks to make environment compatible with Atari Preprocessing
        self.env.unwrapped.frameskip = 1
        self.env.spec.id += '-NoFrameskip'


        self.envWrapped = AtariPreprocessing(self.env, frame_skip=self.frameskip, grayscale_obs=grayscale_obs, scale_obs=scale_obs)
        self.envWrapped.reset()

        self.n_actions = self.env.action_space.n

        init_screen = self.get_screen()
        # Screen dimension is represented as (CHW) for PyTorch
        self.scr_dims = tuple([self.last_n_frames] + list(init_screen.shape))

        # for _ in range(self.frameskip):
        #     self.buffer.append(init_screen.copy())
        self.start_game()

    def start_game(self):
        # input No-Ops till buffer gets filled
        # This would mean 'last_n_frames' * 'frameskip' number of single frame No-Op actions
        while len(self.buffer) < self.buffer.maxlen:
            # 0 corresponds to No-Op action
            self.step(0)

    def get_screen(self):
        screen = self.envWrapped._get_obs()
        return screen

    def get_input(self):
        # Each element in buffer is a tensor of 84x84 dimensions.
        # This function returns tensor of 4x84x84 dimensions.
        return np.stack(tuple(self.buffer), axis=0).astype(np.float32)
    
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
        screen, reward, done, _ = self.envWrapped.step(action)

        # ALE takes care of the max pooling of the last 2 frames
        # Refer: "https://danieltakeshi.github.io/2016/11/25/
        # frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/"
        self.buffer.append(screen)

        # reward is clipped between -1 and 1
        reward = np.clip(reward, -1.0, 1.0)

        return reward, done


"""
Actions in OpenAI Gym ALE
-------------------------
ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
"""
