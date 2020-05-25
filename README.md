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
### An Insight into Double DQN Algorithm
* Two key concepts of DQN algorithm are the use of _target network_ and _experience replay_. 
* However it uses the same values for both the selection and evaluation of an action which leads to unwanted overestimations. 
* Double DQN decouples this selection and evaluation by using a second set of weights for the target network used for the evaluation of the current greedy policy. 
* This target network is a periodic copy of the online network and is updated after a fixed number of episodes. 
* This way Double DQN produces more accurate value estimates and better learned policies.
***
### Network Architecture and Hyper-parameters used
For this project, we have used a similar _Convolution Network_ as given in the research paper, details of which are as follows:
* Input to the network is a 4x84x84 tensor containing a rescaled and grey scale version of the last four frames. The network consists of 3 convolutional layers followed by two fully connected layers.
* The two fully connected layers, the hidden and the output layer, have a size of _512_ and the size of _number of actions_ respectively.
* The first convolutional layer has 32 filters of size 8 (stride 4), second layer has 64 layers of size 4(stride 2) and the final convolutional layer has 64 filters of size 3 (stride 1).
* All the layers, except the last one, are separated by ReLU.
* The optmization technique used is RMSProp with momentum paameter set as 0.95
* A discount factor of 0.99 and a batch size of 32 have been used.
* For the epsilon-greedy policy, epsilon varies lineraly from 1 to 0.1 over 1,000,000 game steps.
* However, we use a learning rate of 0.00001 for our purposes instead of 0.00025 as mentioned in the paper. Due to resource constraints, a buffer length of 15,000 steps has been used against the suggested value of 1M steps. 

***
### Challenging areas
* Training the model has been one of the major areas of concern. Due to limited resources we could not train the agent for all the 49 games as mentioned in the paper.

_-add frameskip and other things-

***
### Inferences Drawn
<img align="centre" src="https://github.com/chirag-singhal/Double-DQN/blob/master/Rewards%20And%20Losses/pinball.gif">

Here we analyse the results for **VideoPinball**. The above is a gif of the mentioned game. The results are more or less the same for other Atari 2600 games as well.


<img align="left" src="https://github.com/chirag-singhal/Double-DQN/blob/master/Rewards%20And%20Losses/pinball_avg_reward_dqn.png">
<img align = "centre" src = "https://github.com/chirag-singhal/Double-DQN/blob/master/Rewards%20And%20Losses/pinball_avg_reward.png">

* The above two graphs depict the average rewards received by the model for the game **VideoPinball** for DQN and Double DQN respectively.
* Using evaluation reward as a comparison metric, we can say that Double DQN outperforms DQN.
* Rewards achieved by Double DQN are higher than those of DQN as we can be clearly seen from the 2 graphs.
* Double Q-learning leads to more consistent results as compared to Q-learing. As we can see that average rewards are fluctuating in the case of DQN whereas they are more consistent in the case of Double-DQN.
* This also shows that Double DQN not only leads to better rewards but also gives better learned policies. Thus, Double DQN is a significant improvement over DQN making minimalistic changes to the existing network architecture.

***
### Code Deployment
_-Prerequisites-_

_-How to run the code-_
***
