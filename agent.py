from itertools import count
import random

import numpy as np

import torch
import torch.optim as optim

from game import Game
from experience_replay import experienceReplay
from dqn import DQN


class Agent():
    
    def __init__(self, game_name, device='cpu'):
        
        #Set hyperparameters
        self.discount = 0.99
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.eps_start = 1
        self.eps_end = 0.1
        self.eps_decay = 1000000
        self.target_update = 10000
        self.num_steps = 50000000
        self.max_episodes = 1000 # 10000
        
        #Device
        self.device = torch.device(device)
        print('Using device: ', device)

        #Game
        self.game = Game(game_name)
        self.num_actions = self.game.get_n_actions()
        
        #Experience Replay Memory
        self.memory_size = 10000 # 10000000
        self.memory = experienceReplay(self.memory_size)
        
        #Double Deep Q Network
        self.primary_network = DQN(self.num_actions).to(self.device)
        self.target_network = DQN(self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.primary_network.state_dict())
        self.target_network.eval()
        
        #Loss function
        self.loss_func = torch.nn.MSELoss()
        
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
            return np.random.choice(np.arange(self.num_actions))
        else:
            #exploitation
            #use primary_network to estimate q-values of actions
            state = torch.as_tensor(state).to(self.device)
            return torch.argmax(self.primary_network(state.unsqueeze(0))).detach().cpu().numpy()
        
    
    def batch_train(self):
        """
            Performs batch training on the network. Implements Double Q learning on network. 
            It evaluates greedy policy using primary network but its value is 
            estimated using target network.
            Loss funtion used is Mean Squared Error.
            Uses RMSprop for gradient based optimisation.
        """
        if(self.memory.number_of_experiences() < self.batch_size):
            #Not enough experiences for batch training
            return
        
        #Sample batch from replay memory
        batch_data = self.memory.selectBatch(self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, done = list(zip(*batch_data))

        batch_states = torch.as_tensor(np.stack(batch_states, axis=0)).to(self.device)
        batch_next_states = torch.as_tensor(np.stack(batch_next_states, axis=0)).to(self.device)
        batch_actions = torch.as_tensor(np.stack(batch_actions)).to(self.device)
        batch_rewards = torch.tensor(batch_rewards).to(self.device)
        not_done = (~torch.tensor(done).unsqueeze(1)).to(self.device)
        
        Q_t_values = self.primary_network(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze()
        
        next_Q_t_primary_values = self.primary_network(batch_next_states).detach()
        next_Q_t_target_values = not_done * self.target_network(batch_next_states).detach()
        
        next_Q_t_values_max = next_Q_t_target_values.gather(1, torch.argmax(next_Q_t_primary_values, dim=1).unsqueeze(1)).detach().squeeze()
        
        #Double Q-Learning
        expected_Q_values = (batch_rewards + (self.discount * next_Q_t_values_max))
        
        #Calulating loss
        loss = self.loss_func(Q_t_values, expected_Q_values)
        
        #Clear gradients from last backward pass
        self.optimizer.zero_grad()
        
        #Run backward pass and calculate gradients
        loss.backward()
        
        #Update weights from calculated gradients
        self.optimizer.step()

        loss_item = loss.detach().item()

        #Delete GPU tensors to free up GPU memory
        del batch_states
        del batch_next_states
        del batch_actions
        del batch_rewards
        del not_done
        del Q_t_values
        del next_Q_t_primary_values
        del next_Q_t_target_values
        del next_Q_t_values_max
        del expected_Q_values
        del loss
        # if self.device.type == 'cuda':
        #     torch.cuda.empty_cache()
        #     # print(torch.cuda.memory_allocated(device=self.device))

        return loss_item
        
        
    def train(self):
        steps = 0
        total_reward = 0
        record_rewards = []
        record_losses = []
        record_steps = []
        for i in range(self.max_episodes):
            print("Episode ", i)
            self.game.reset_env()
            state = self.game.get_input()
            for j in count():
                #Update counters
                steps += 1
                
                #Select action using greedy policy
                action = self.select_action(steps, state)
                reward, done = self.game.step(action)
                
                total_reward += reward
                
                # if not done:
                #         #get the next state
                #         next_state = self.game.get_input()
                # else:
                #     next_state = None
                next_state = self.game.get_input()
                
                # # Convert all arrays to CPU Torch tensor
                # state = state.cpu()
                # next_state = next_state.cpu()
                # action = action.cpu()
                # reward = torch.tensor(reward)  # 'reward' is left as float
                # done = torch.tensor(done)  # 'done' is left as boolean

                #Store experiences in replay memory for batch training
                self.memory.storeExperience(state, action, reward, next_state, done)
                
                if done:
                    #Batch Train from experiences if final state is reached
                    loss = self.batch_train()
                    record_steps.append(steps)
                    record_losses.append(loss)
                    record_rewards.append(total_reward)
                    total_reward = 0
                    break

                loss = self.batch_train()
                # record_steps.append(steps)
                # record_losses.append(loss)
                # record_rewards.append(total_reward)
                
                        
                #next state assigned to current state
                state = next_state
                
                if(steps % self.target_update == 0):
                    #Update the target_network
                    with torch.no_grad():
                        self.target_network.load_state_dict(self.primary_network.state_dict())
                        self.target_network.eval()
                
                if(steps == self.num_steps):
                    print("Training Done\n")
                    break
            
            if(steps == self.num_steps):
                break
                
        return record_rewards, record_losses, record_steps
                
                
