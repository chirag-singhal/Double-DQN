

class Agent():
    
    def __init__(self, env):
        
        #Set hyperparameters
        self.discount = 0.99
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.eps_start = 1
        self.eps_end = 0.1
        self.eps_decay = 1000000
        self.target_update = 10000
        self.num_episodes = 50000000
        
        #Environment
        self.env = env
        self.num_actions = env.action_space.n
        
        #Experience Replay Memory
        self.memory_size = 10000000
        self.memory = ExperienceReplay(self.memory_size)
        
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
        