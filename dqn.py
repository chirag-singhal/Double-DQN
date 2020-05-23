import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    
    def __init__(self, outputs, h = 84, w = 84):
        """
            The input to the network is a 84 * 84 * 4 tensor containing a rescaled,
            and gray scale version of last four frames.
            
            The network consists of three convolutional layers followed by two 
            fully connected layers.
            
            Arguments:
                h: Height of input
                w: Width of input
                otputs: Number of total actions possible
        """
        
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        
        def conv_2d_size_output(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        #Calculating the output size of all the three convolution layers
        convw1 = conv_2d_size_output(w, 8, 4)
        convw2 = conv_2d_size_output(convw1, 4, 2)
        convw3 = conv_2d_size_output(convw2, 3, 1)
        convh1 = conv_2d_size_output(h, 8, 4)
        convh2 = conv_2d_size_output(convh1, 4, 2)
        convh3 = conv_2d_size_output(convh2, 3, 1)
        self.linear_input_size = convw3 * convh3 * 64
        
        self.fc4 = nn.Linear(self.linear_input_size, 512)
        self.fc5 = nn.Linear(512, outputs)

        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        
    def forward(self, x):
        """
            Perform forward propagation of the network. Activation function used is ReLU.
            
            Arguments:
                x: Input to the network
        """
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        #Flatten the output from Convolution Layers
        x = x.view(-1, self.linear_input_size)
        
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x
        