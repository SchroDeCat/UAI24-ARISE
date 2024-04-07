
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPE = torch.float

class Random():
    """
    Random baseline for Nash Quilibrium Pipeline. Note the objective is to be minimized
    """
    @staticmethod
    def query( test_x:torch.Tensor,  **kwargs)->torch.Tensor:
        '''
        Rando for Nash Equilibrium
        Input:
            @test_x: discrete search space
        Output:
            candidate: to be evaluated (with minimum gap)
        '''
        max_pts = np.random.choice(np.arange(test_x.size(0)),size=1)      

        candidate = test_x[max_pts]
        return candidate


