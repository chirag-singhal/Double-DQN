from collections import deque

import numpy as np

import torch
import torch.transforms as T

import gym


class Game():

    # The transform grayscales and resizes the image to 84x84
    transform = T.compose([
        T.ToPILImage(),
        T.Resize((84, 84), interpolation=Image.BILINEAR),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()])

    def __init__(self, game_name, num_frames_per_step=4, device='cpu'):
        self.num_frames_per_step = num_frames_per_step
        self.buffer = deque([], self.num_frames_per_step)
        
        self.env = gym.make(game_name)
        self.env.reset()
        
        self.n_actions = self.env.action_space.n

        init_screen = self.get_screen()
        self.scr_h, self.scr_w = init_screen.shape

        for _ in range(len(self.num_frames_per_step)):
            self.buffer.append(init_screen.clone())

    def get_screen():
        # Gym returns screen as HWC numpy array, convert it to CHW
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Convert to float, rescale
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
        screen = torch.from_numpy(screen)
        screen = transform(screen).to(device)
        # The output is now 84x84
        return screen

    def get_input():
        # Each element in buffer is a tensor of 84x84 dimensions.
        # This function returns tensor of 84x84x4 dimensions.
        return torch.stack(torch.tuple(self.buffer), dim=2)


    def step(action):
        reward = 0
        done = False
        for step in range(self.num_frames_per_step):
            _, _reward, done, _ = self.env.step(action)
            self.buffer.append(self.get_screen())
            reward += _reward
            if done:
                break
        return reward, done
