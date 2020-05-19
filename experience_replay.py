import torch
import random
import collections as c


class experienceReplay(object):

    experience = c.namedtuple('experience', 'state, action, reward, next_state, is_done')

    def __init__(self, max_size):
        self.size = max_size
        self.count = 0
        self.store = c.deque([]);
        self.store1 = c.deque();
        """
        State, action, reward and next state are stored in memory
        to train the network. 

        terminal_memory helps to identify if it's the end of an episode.

        """
        """
        self.state_memory = torch.zeros((self.size, *itorchut_shape), dtype = torch.float32)
        self.next_state_memory = torch.zeros((self.size, *itorchut_shape), dtype = torch.float32)
        self.action_memory = torch.zeros(self.size, dtype = torch.float32)
        self.reward_memory = torch.zeros(self.size, dtype = torch.float32)
        self.terminal_memory = torch.zeros(self.size, dtype = torch.int32)
        """

    def storeExperience(self, state, action, reward, next_state, is_done):
        """
        Some old samples are removed and new ones are stored in the memory.

        """
        """
        i = self.count % self.size
        self.state_memory[i] = state.transpose(0, 2)
        self.next_state_memory[i] = next_state.transpose(0, 2)
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.terminal_memory[i] = is_done
        self.count += 1
        """
        t = experienceReplay.experience(state, action, reward, next_state, is_done)
        self.store.append(t)
        self.count += 1

    def selectBatch(self, batch_size):
        """
        A random batch of size batch_size is chosen from amongst the
        stored experiences to train the primary network.
        """
        max_mem = min(self.count, self.size)
        
        #batch has the random indices selected from max_mem
        # NOTE:
        # For now, there is a bug in the implementation that newer memories
        # (memories after max_size) never get used
        batch = random.sample(range(max_mem), batch_size)
        """
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_state = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_state, terminal
        """
        for i in range(batch_size):
            t = self.store[batch[i]]
            self.store1.append(t)
        return self.store1
    
    def number_of_experiences(self):
        #Returns total number of experiences stored in memory
        if(self.count >= self.size):
            return self.size
        else:
            return self.count
        
