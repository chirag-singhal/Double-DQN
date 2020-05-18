import torch as nn
import random


class experienceReplay(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.size = max_size
        self.count = 0
        """
        State, action, reward and next state are stored in memory
        to train the network. 

        terminal_memory helps to identify if it's the end of an episode.

        """
        self.state_memory = nn.zeros((self.size, *input_shape), dtype = nn.float32)
        self.next_state_memory = nn.zeros((self.size, *input_shape), dtype = nn.float32)
        self.action_memory = nn.zeros(self.size, dtype = nn.float32)
        self.reward_memory = nn.zeros(self.size, dtype = nn.float32)
        self.terminal_memory = nn.zeros(self.size, dtype = nn.int32)

    def storeExperience(self, state, action, reward, next_state, is_done):
        """
        Some old samples are removed and new ones are stored in the memory.

        """
        i = self.count % self.size
        self.state_memory[i] = state
        self.next_state_memory[i] = next_state
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.terminal_memory[i] = is_done
        self.count += 1

    def selectBatch(self, batch_size):
        """
        A random batch of size batch_size is chosen from amongst the
        stored experiences to train the primary network.
        """
        max_mem = min(self.count, self.size)
        
        #batch has the random indices selected from max_mem
        batch = random.sample(max_mem, batch_size, replace = False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_state = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_state, terminal
    
    def number_of_experiences():
        #Returns total number of experiences stored in memory
        if(count >= self.size):
            return self.size
        else:
            return count
        
