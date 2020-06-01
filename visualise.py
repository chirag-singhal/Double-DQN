import numpy as np
import matplotlib.pyplot as plt

from agent import Agent

# game_name = 'BreakoutNoFrameskip-v4' # 'Breakout-v0' # 'VideoPinball-v0'
# device = 'cuda:0'
# chkpnt_name = 'BreakoutNoFrameskip-v4_10000' # 'breakoutv4_100_test'
# pretrained_name = 'BreakoutNoFrameskip-v4_10000' # 'breakoutv4_100_test'
# verbosity = 0

# agent = Agent(game_name, device, chkpnt_name, pretrained_name, verbosity)

# game_name = 'VideoPinballNoFrameskip-v4' # 'Breakout-v0' # 'VideoPinball-v0'
# device = 'cuda:0'
# chkpnt_name = 'VideoPinballNoFrameskip-v4_10000'
# pretrained_name = 'VideoPinballNoFrameskip-v4_10000' # 'VideoPinballNoFrameskip-v4_10000'
# verbosity = 0

# agent = Agent(game_name, device, chkpnt_name, pretrained_name, verbosity)

# plt.imshow(agent.sanity_check_screen())
# plt.plot()

print(agent.evaluate(visualise=True))
