import numpy as np

import torch as nn

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
        self.num_actions = self.game.n_actions
        
        #Experience Replay Memory
        self.memory_size = 10000000
        self.memory = experienceReplay(self.memory_size)
        
        #Double Deep Q Network
        self.primary_network = DQN(self.num_actions)
        self.target_network = DQN(self.num_actions)
        self.target_network.load_state_dict(self.primary_network.state_dict())
        self.target_network.eval()
        
        
        #Optimiser
        self.momentum = 0.95
        self.optimizer = optim.RMSprop(
                            self.primary_net.parameters(), 
                            lr=self.learning_rate, alpha=0.99, eps=1e-08, 
                            weight_decay=0, momentum=self.momentum
                        )
        
    def select_action(self, steps, state):
        #linear decay of epsilon value
        epsilon = self.eps_start + (self.eps_end - self.eps_start) * (steps / self.eps_decay)
        if random.random() < epsilon:
            #exploration
            return np.random.choice(np.arrange(self.num_actions))
        else:
            #exploitation
            #use target_network to estimate q-values of actions
            return nn.argmax(self.target_network(state))
    
    def train(self):
        steps = 0
        total_reward = 0
        record_rewards = []
        for i in range(self.max_episodes):
            self.game.env.reset()
            state = self.game.get_input()
            for j in count():
                #Update counters
                steps += 1
                
                #Select action using greedy policy
                action = self.primary_network.select_action(steps, state)
                reward, done = self.game.step(action)
                
                total_reward += reward
                
                if not done:
                        #get the next state
                        next_state = self.game.get_input()
                else:
                    next_state = None
                
                self.memory.storeExperience(state, action, reward, next_state, done)
                
                if(done || steps == self.num_steps):
                        #Batch Train from experiences if final state is reached
                        #or if the total steps is reached
                        #Not Implemented Yet
                        self.primary_network.batch_train()
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
                
                
