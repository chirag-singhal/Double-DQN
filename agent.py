from itertools import count
import random
import pickle

import numpy as np

import torch
import torch.optim as optim

from game import Game
from experience_replay import experienceReplay
from dqn import DQN


class Agent():
    
    def __init__(self, game_name, device='cpu', chkpnt_name=None, pretrained_name=None, verbosity=0):
        
        #Set hyperparameters
        self.discount = 0.99
        self.learning_rate = 0.00001
        self.batch_size = 32
        self.eps_start = 1
        self.eps_end = 0.1
        self.eps_decay = 1000000 # 1000000
        self.primary_update = 4
        self.target_update = 10000
        self.num_steps = 50000000
        self.max_episodes = 10000 # 10000
        self.episodes_per_chkpnt = 50
        self.evaluation_steps = 10000 # 1000000
        
        #Model Checkpointing
        if chkpnt_name == None:
            chkpnt_name = game_name + '_' + str(self.max_episodes)
        self.chkpnt_path = 'models/' + chkpnt_name
        
        #Metrics
        self.metrics = {
            'rewards': [],
            'losses': [],
            'steps': [],
            'cum_steps': [],
            'evaluation': []
        }

        #Device
        self.device = torch.device(device)
        print('Using device: ', device)

        #Game
        self.game = Game(game_name)
        self.num_actions = self.game.get_n_actions()
        
        #Experience Replay Memory
        self.memory_size = 25000 # 10000000
        self.memory = experienceReplay(self.memory_size)
        
        #Double Deep Q Network
        self.primary_network = DQN(self.num_actions).to(self.device).float()
        self.target_network = DQN(self.num_actions).to(self.device).float()
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

        #Open pretrained model
        if pretrained_name != None:
            with open('models/' + pretrained_name + '.metrics', 'rb') as metrics_file:
                self.metrics = pickle.load(metrics_file)
            self.primary_network.load_state_dict(torch.load('models/' + pretrained_name + '.pth'))
            self.primary_network.train()
            self.optimizer.load_state_dict(torch.load('models/' + pretrained_name + '.opt'))
            with torch.no_grad():
                self.target_network.load_state_dict(self.primary_network.state_dict())
                self.target_network.eval()
            print('Using pretrained model: ' + pretrained_name)

        #Verbosity
        # 0 - No info
        # 1 - Prints metrics per episode
        # 2 - Prints training batch information (BLOCKS EXECUTION)
        # 3 - Prints model weights and saves state plot (BLOCKS EXECUTION)
        self.verbosity = verbosity
        
        
    def sanity_check_screen(self):
        #Returns the opening screen of game for sanity checks
        #The opening screen should be 84x84 grayscaled image
        self.game.reset_env()
        state = self.game.get_screen()
        print('Dimension of screen: ', state.shape)
        return state


    def select_action(self, steps, state):
        """
            Selects next action to perform using greedy policy. Target network is used to 
            estimate q-values.
            
            Arguments:
                steps - Number of steps performed till now
                state - Current state of atari game, contains last 4 frames. 
        """
        #linear decay of epsilon value
        epsilon = self.eps_start + (self.eps_end - self.eps_start) * (min(steps, self.eps_decay) / self.eps_decay)
        if random.random() < epsilon:
            #exploration
            return np.random.choice(np.arange(self.num_actions))
        else:
            #exploitation
            #use primary_network to estimate q-values of actions
            state = torch.as_tensor(state, dtype=torch.float).to(self.device) / 255.
            return torch.argmax(self.primary_network(state.unsqueeze(0))).detach().cpu().numpy()
        
    
    def evaluate(self, visualise=False):
        total_reward = 0
        done = False

        self.game.reset_env()

        if visualise:
            self.game.env.render()
            import time
            time.sleep(0.03)

        while not done:
            if visualise:
                self.game.env.render()
                time.sleep(0.03)
            state = self.game.get_input()
            action = self.select_action(self.eps_decay, state)
            reward, done = self.game.step(action)            
            total_reward += reward

        self.game.env.close()
        return total_reward


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

        #Convert the batch information into PyTorch tensors
        batch_states = torch.as_tensor(np.stack(batch_states, axis=0), dtype=torch.float).to(self.device) / 255.
        batch_next_states = torch.as_tensor(np.stack(batch_next_states, axis=0), dtype=torch.float).to(self.device) / 255.
        batch_actions = torch.as_tensor(np.stack(batch_actions)).to(self.device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float).to(self.device)
        not_done = (~torch.tensor(done).unsqueeze(1)).to(self.device)
        
        #Prediction
        Q_t_values = self.primary_network(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze()
        
        #Ground-truth
        next_Q_t_primary_values = self.primary_network(batch_next_states)
        next_Q_t_target_values = not_done * self.target_network(batch_next_states)
        
        next_Q_t_values_max = next_Q_t_target_values.gather(1, torch.argmax(next_Q_t_primary_values, dim=1).unsqueeze(1)).detach().squeeze()
        
        #Double Q-Learning
        expected_Q_values = (batch_rewards + (self.discount * next_Q_t_values_max))
        
        #Calulating loss
        loss = self.loss_func(Q_t_values, expected_Q_values)
 
        # DEBUG
        if self.verbosity >= 2 and loss.detach().item() < 0.1:
            print('BATCH_ACTION: ', batch_actions)
            print('BATCH_REWARD: ', batch_rewards)
            print('LOSS: ', loss.detach().item())
            print('PRIMARY: ', self.primary_network(batch_states).detach())
            print('Q: ', Q_t_values)
            print('T: ', expected_Q_values)
            
            if self.verbosity >= 3:
                import matplotlib.pyplot as plt
                plt.imshow(batch_states[0][0].cpu().numpy())
                plt.plot()
                plt.savefig('tmp_img.png')
                
                for name, param in self.primary_network.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)
            print('\n')
            input()
        # DEBUG

        #Clear gradients from last backward pass
        self.optimizer.zero_grad()
        
        #Run backward pass and calculate gradients
        loss.backward()
        
        # #Clip loss gradient between -1 and 1
        # for param in self.primary_network.parameters():
        #     param.grad.data.clamp_(-1, 1)

        #Update weights from calculated gradients
        self.optimizer.step()

        return loss.detach().item()
        
        
    def train(self):

        def save_model():
            print('Saving model and metrics ...')
            torch.save(self.primary_network.state_dict(), self.chkpnt_path + '.pth')
            torch.save(self.optimizer.state_dict(), self.chkpnt_path + '.opt')
            with open(self.chkpnt_path + '.metrics', 'wb') as metrics_file:
                pickle.dump(self.metrics, metrics_file)

        steps = 0
        
        for i in range(self.max_episodes):
            
            if self.verbosity >= 1:
                print('\n', '-'*40)
            print('Episode ', i)
            
            total_reward = 0
            
            self.game.reset_env()
            state = self.game.get_input()

            for steps_delta in count():
                #Update counters
                steps += 1
                
                #Select action using greedy policy
                action = self.select_action(steps, state)
                reward, done = self.game.step(action)
                
                total_reward += reward
                
                if done:
                    #Reset game screen if terminal state reached
                    self.game.reset_env()
                
                next_state = self.game.get_input()
                
                #Store experiences in replay memory for batch training
                self.memory.storeExperience(state, action, reward, next_state, done)
                
                #Train primary network every k steps
                if steps % self.primary_update == 0:
                    loss = self.batch_train()

                #next state assigned to current state
                state = next_state
                
                if steps % self.target_update == 0:
                    #Update the target_network
                    with torch.no_grad():
                        self.target_network.load_state_dict(self.primary_network.state_dict())
                        self.target_network.eval()
                
                if done:
                    if self.verbosity >= 1:
                        print('Steps taken: ', steps_delta)
                        print('Cumulative Steps taken: ', steps)
                        print('Loss: ', loss)
                        print('Reward: ', total_reward)
                    #Record the metrics after an episode
                    self.metrics['steps'].append(steps_delta)
                    self.metrics['cum_steps'].append(steps)
                    self.metrics['losses'].append(loss)
                    self.metrics['rewards'].append(total_reward)
                    break
                                
                if steps == self.num_steps:
                    print("Training Done\n")
                    break
            
            #Model checkpointing
            if i % self.episodes_per_chkpnt == self.episodes_per_chkpnt-1:
                save_model()

            #Model evaluation
            if steps % self.evaluation_steps == 0:
                eval_reward = self.evaluate()
                if self.verbosity >= 1:
                    print('Evaluation reward: ', eval_reward)
                self.metrics['evaluation'].append((steps, eval_reward))
            
            #Maximum training steps reached
            if steps == self.num_steps:
                save_model()
                return self.metrics

        #Save final trained model
        save_model()

        return self.metrics
                
                
