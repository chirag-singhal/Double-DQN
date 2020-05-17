from collections import deque

import numpy as np

import torch
import torch.transforms as T

import gym

# The following code can be simplified by using AtariPreprocessing (TODO: Harshan)
# from gym.wrappers import AtariPreprocessing


class Game():

    # The transform grayscales and resizes the image to 84x84
    # Resize resizes to 110x84
    # Center crop makes it into 84x84
    transform = T.compose([
        T.ToPILImage(),
        T.Resize(84, interpolation=Image.BILINEAR),
        T.CenterCrop(84),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()])

    def __init__(self, game_name, num_frames_per_step=4, device='cpu'):
        self.num_frames_per_step = num_frames_per_step
        self.buffer = deque([], self.num_frames_per_step)
        
        # Most games in ALE have frameskip.
        # This makes it difficult to obtain continuous video feed
        # while performing repeated actions.
        # For now, environment can be unwrapped and frameskip can
        # be manually overriden to be 1
        self.env = gym.make(game_name).unwrapped
        self.env.frameskip = 1
        self.env.reset()
        
        self.n_actions = self.env.action_space.n

        init_screen = self.get_screen()
        self.scr_h, self.scr_w = init_screen.shape

        for _ in range(len(self.num_frames_per_step)):
            self.buffer.append(init_screen.clone())

    def get_screen():
        # Gym returns screen as HWC numpy array, convert it to CHW
        # Gym returns (210, 160, 3)
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
