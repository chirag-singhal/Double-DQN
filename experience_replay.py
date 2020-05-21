import torch
import random
import collections as c


class experienceReplay(object):

    experience = c.namedtuple('experience', 'state, action, reward, next_state, is_done')

    def __init__(self, max_size):
        """
        State, action, reward and next state are stored in memory
        to train the network. 

        terminal_memory helps to identify if it's the end of an episode.

        """
        self.store = c.deque([], max_size);

    def storeExperience(self, state, action, reward, next_state, is_done):
        """
        Some old samples are removed and new ones are stored in the memory.

        """
        t = experienceReplay.experience(state, action, reward, next_state, is_done)
        self.store.append(t)

    def selectBatch(self, batch_size):
        """
        A random batch of size batch_size is chosen from amongst the
        stored experiences to train the primary network.
        """
        
        #batch has the random indices selected from max_mem
        batch = random.sample(self.store, batch_size)
        return batch
    
    def number_of_experiences(self):
        """
        Returns total number of experiences stored in memory
        """
        return len(self.store)
        
