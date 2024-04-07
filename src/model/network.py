import torch

class SeqNeuralNetwork(torch.nn.Module):
    '''
    Neural Netowrk with sequential architecture.
    '''
    def __init__(self, arc: torch.nn.Sequential):
        super().__init__()
        self.arc = arc

    def forward(self, x):
        pred = self.arc(x)
        return pred