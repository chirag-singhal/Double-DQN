from itertools import count
import random

import numpy as np

import torch as nn
import torch.optim as optim

from game import Game
from experience_replay import experienceReplay
from dqn import DQN


class Agent():
    
    def __init__(self, game_name):
        
        #Set hyperparameters
        self.discount = 0.99
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.eps_start = 1
        self.eps_end = 0.1
        self.eps_decay = 1000000
        self.target_update = 10000
        self.num_steps = 50000000
        self.max_episodes = 10000
        
        #Game
        self.game = Game(game_name)
        self.num_actions = self.game.get_n_actions()
        
        #Experience Replay Memory
        self.memory_size = 10000 # 10000000
        self.memory = experienceReplay(self.memory_size, self.game.get_screen_dims(), self.num_actions)
        
        #Double Deep Q Network
        self.primary_network = DQN(self.num_actions)
        self.target_network = DQN(self.num_actions)
        self.target_network.load_state_dict(self.primary_network.state_dict())
        self.target_network.eval()
        
        
        #Optimiser
        self.momentum = 0.95
        self.optimizer = optim.RMSprop(self.primary_network.parameters(), 
                            lr=self.learning_rate, alpha=0.99, eps=1e-08, 
                            weight_decay=0, momentum=self.momentum
                        )
        #clear gradients
        self.optimizer.zero_grad()
        
        
    def select_action(self, steps, state):
        """
            Selects next action to perform using greedy policy. Target network is used to 
            estimate q-values.
            
            Arguments:
                steps - Number of steps performed till now
                state - Current state of atari game, contains last 4 frames. 
        """
        #linear decay of epsilon value
        epsilon = self.eps_start + (self.eps_end - self.eps_start) * (steps / self.eps_decay)
        if random.random() < epsilon:
            #exploration
            return nn.tensor(np.random.choice(np.arange(self.num_actions)))
        else:
            #exploitation
            #use target_network to estimate q-values of actions
            return nn.argmax(self.target_network(state))
        
    
    def batch_train(self):
        """
            Performs batch training on the network. Implements Double Q learning on network. 
            It evaluates greedy policy using primary network but its value is 
            estimated using target network.
            Loss funtion used is Mean Squared Error.
            Uses RMSprop for gradient based optimisation.
        """
        if(self.memory.number_of_experiences() < batch_size):
            #Not enough experiences for batch training
            return
        
        #Sample batch from replay memory
        batch_states, batch_actions, batch_rewards, batch_next_states, done = self.memory.selectBatch(self.batch_size)
        
        batch_states = nn.from_numpy(batch_states).type(nn.float32)
        batch_actions = nn.from_numpy(batch_actions).type(nn.float32)
        batch_rewards = nn.from_numpy(batch_rewards).type(nn.float32)
        batch_next_states = nn.from_numpy(batch_next_state).type(nn.float32)
        not_done = nn.from_numpy(1 - done).type(nn.int32)
        
        Q_t_values = self.target_network(batch_states)[:, batch_actions]
        
        next_Q_t_primary_values = not_done * self.target_network(batch_next_states)
        next_Q_t_target_values = not_done * self.target_network(batch_next_states)
        
        next_Q_t_values_max = next_Q_t_target_values[:, np.argmax(next_Q_t_primary_values, axis=0)]
        
        #Double Q-Learning
        expected_Q_values = batch_rewards + (self.discount * next_Q_t_values_max)
        
        #Calulating loss
        loss = np.mean(np.square(Q_t_values - expected_Q_values))
        
        #Clear gradients from last backward pass
        self.optimizer.zero_grad()
        
        #Run backward pass and calculate gradients
        loss.backward()
        
        #Update weights from calculated gradients
        self.optimizer.step()
        
        
    def train(self):
        steps = 0
        total_reward = 0
        record_rewards = []
        for i in range(self.max_episodes):
            self.game.reset_env()
            state = self.game.get_input()
            for j in count():
                #Update counters
                steps += 1
                
                #Select action using greedy policy
                action = self.select_action(steps, state)
                reward, done = self.game.step(action)
                
                total_reward += reward
                
                if not done:
                        #get the next state
                        next_state = self.game.get_input()
                else:
                    next_state = None
                
                # Convert everything to one format (CPU PyTorch Tensor)
                state = state.cpu()
                next_state = next_state.cpu()
                action = action.cpu()
                reward = nn.tensor(reward)
                done = nn.tensor(done)

                #Store experiences in replay memory for batch training
                self.memory.storeExperience(state, action, reward, next_state, done)
                
                if(done):
                        #Batch Train from experiences if final state is reached
                        self.batch_train()
                        record_rewards.append(total_reward)
                        total_reward = 0
                        break
                        
                #next state assigned to current state
                state = next_state
                
                if(steps % self.target_update == 0):
                    #Update the target_network
                    self.target_network.load_state_dict(self.primary_network.state_dict())
                    self.target_network.eval()
                
                if(steps == self.num_steps):
                    print("Training Done\n")
                    break
            
            if(steps == self.num_steps):
                break
                
        return record_rewards
                
                
