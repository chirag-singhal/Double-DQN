

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
        self.num_steps = 50000000
        self.max_episodes = 10000
        
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
    def train(self):
        steps = 0
        primary_net_counter = 0
        for i in range(self.max_episodes):
            self.env.reset()
            for k in range(4):
                state[:,:,k] = get_screen() 
            for j in count():
                #Update counters
                steps += 1
                primary_net_counter = (primary_net_counter) % 4
                
                #Select action using greedy policy
                #Not Implemented Yet
                action = self.primary_network.select_action(state)
                _, reward, done, _ = env.step(action)
                
                if not done:
                        #get the next state
                        for k in range(4):
                            next_state[:,:,k] = get_screen() 
                else:
                    next_state = None
                
                self.memory.storeExperience(state, action, reward, next_state, done)
                
                if(primary_net_counter == 0 || done || steps == self.num_steps):
                        #Batch Train from experiences after every four steps 
                        #or if final state is reached
                        #or if the total steps is reached
                        #Not Implemented Yet
                        self.primary_network.batch_train()
                
                if(counter % self.target_update == 0):
                    #Update the target_network
                    self.target_network.load_state_dict(self.primary_network.state_dict())
                    self.target_network.eval()
                
                if(steps == self.num_steps):
                    print("Training Done\n")
                    break
            
            if(steps == self.num_steps):
                break
                
                