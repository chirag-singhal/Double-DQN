# Deep Reinforcement Learning with Double Q-learning
## Hado van Hasselt and Arthur Guez and David Silver
### Google DeepMind
The research paper can be found [here](https://arxiv.org/pdf/1509.06461.pdf).
***
#### Goal of the project: To analyze the overestimations of DQN and show that Double DQN improves over DQN both in terms of value accuracy and in terms of policy quality.

This _repository_ contains the code for implementation of the same.

Our testbed consists of Atari 2600 games, using the OpenAI Gym Atari environments.
_(Gym is a toolkit for developing and comparing reinforcement learning algorithms. The gym open source library gives access to a standardized set of environments.)_

### Learning Outcomes: 
* Q-learning often results in overestimations because of its inclination towards an upward bias caused due to various estimation errors and noise, leading to a suboptimal learned policy.
* Double Q-learning produces a significant improvement over Q-learning resulting in values that are much closer to the ground truth.
* Double DQN gives more accurate value estimates and better learned policies producing state-of-the-art result on Atari 2600 domain using the already existing DQN architecture. 
***
### What is Double DQN Algorithm?
