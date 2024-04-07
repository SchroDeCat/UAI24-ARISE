from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize

def sample_pts(lb, ub, n_pts:int=10, dim:int=2, seed:int=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    x = sobol.draw(n=n_pts)
    return unnormalize(x, (lb, ub))