from abc import ABC, abstractmethod
from .general import check_resutls_dir, find_nearest_points
import numpy as np
import itertools
import os
import tqdm
from joblib import Parallel, delayed
from typing import List


def cartesian_product_transpose(*arrays)->np.ndarray:
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T

def point_distance(x1, y1, x2, y2, x, y):
    distance1 = (x - x1) ** 2 + (y - y1) ** 2
    distance2 = (x - x2) ** 2 + (y - y2) ** 2
    return distance1 < distance2

def find_nearest_hotel(x1, y1, x2, y2, x3, y3, x, y):
    distance1 = (x - x1) ** 2 + (y - y1) ** 2
    distance2 = (x - x2) ** 2 + (y - y2) ** 2
    distance3 = (x - x3) ** 2 + (y - y3) ** 2
    if distance1 < distance2 and distance1 < distance3:
        return 1
    elif distance2 < distance1 and distance2 < distance3:
        return 2
    else:
        return 3

def area_ratio3(x1, y1, x2, y2, x3, y3):
    
    s1 = 0
    s2 = 0
    s3 = 0

    num_samples = 1000  # Adjust this for accuracy

    for _ in range(num_samples):
        x = np.random.random()
        y = np.random.random()
        
        # if point_distance(x1, y1, x2, y2, x, y):
        #     s1 += 1
        # else:
        #     s2 += 1
        closest = find_nearest_hotel(x1, y1, x2, y2, x3, y3, x, y)
        if closest == 1:
            s1 += 1
        elif closest == 2:
            s2 += 1
        else:
            s3 += 1

    s1 = s1 / num_samples  # Normalize to the unit square area
    s2 = s2 / num_samples  # Normalize to the unit square area
    s3 = s3 / num_samples  # Normalize to the unit square area
    return s1, s2, s3

def area_ratio(x1, y1, x2, y2):
    
    s1 = 0
    s2 = 0

    num_samples = 1000  # Adjust this for accuracy

    for _ in range(num_samples):
        x = np.random.random()
        y = np.random.random()
        
        if point_distance(x1, y1, x2, y2, x, y):
            s1 += 1
        else:
            s2 += 1

    s1 = s1 / num_samples  # Normalize to the unit square area
    s2 = s2 / num_samples  # Normalize to the unit square area
    return s1, s2

class problem(ABC):
    @abstractmethod
    def utilities(self, x):
        '''
        return the utilities of the two players
        '''
        pass

    @abstractmethod
    def evaluate_regret(self, x):
        '''
        return the regret of the two players
        '''
        pass


    @staticmethod
    def load_mem(task_inst, mem:str='util')->bool:
        '''
        Load the utility / regret memory for a given task instance
        Return if successfully loaded
        '''
        assert mem in ['util', 'reg']
        name = task_inst.__name__
        size = task_inst.x_space.shape[0]
        dir = f"tmp/{name}-{size}"

        # load memory
        _file_name = f'{dir}_{mem}.npy'
        if os.path.exists(_file_name):
            data = np.load(_file_name, allow_pickle=True)
            if mem == 'util':
                task_inst.util_mem = data
            else:
                task_inst.reg_mem = data
            return True
        return False


    @staticmethod
    def construct_mem(task_inst, mem:str='util', force=False):
        '''
        Construct the utility memory for a given task instance
        '''
        assert mem in ['util', 'reg']
        name = task_inst.__name__
        size = task_inst.x_space.shape[0]
        dir = f"tmp/{name}-{size}"

        load_flag = problem.load_mem(task_inst=task_inst, mem=mem)
        if load_flag and not force:
            return
        
        train_x = task_inst.train_x

        # check if the folder exists
        check_resutls_dir(folder_path='./tmp')
        if mem == 'util':
            task_inst.util_mem = [None for _ in range(len(train_x))]
            for idx, x in tqdm.auto.tqdm(enumerate(train_x), total=len(train_x), desc=f"Running {task_inst.__name__.upper()} {mem.upper()}"):
                task_inst.util_mem[idx] = task_inst.utilities(x)
            # multi-dim ndarray
            task_inst.util_mem = np.array(task_inst.util_mem)
            np.save(f'{dir}_{mem}.npy', arr=task_inst.util_mem)
        elif mem == 'reg':
            task_inst.reg_mem = np.zeros(len(train_x))
            task_inst.reg_mem = Parallel(n_jobs=8)(delayed(task_inst.evaluate_regret)(x) for x in tqdm.auto.tqdm(train_x, total=len(train_x), desc=f"Running {task_inst.__name__.upper()} {mem.upper()}"))
            # for idx, x in tqdm.auto.tqdm(enumerate(train_x), total=len(train_x), desc=f"Running {task_inst.__name__.upper()} {mem.upper()}"):
                # task_inst.reg_mem[idx] = task_inst.evaluate_regret(x)
            np.save(f'{dir}_{mem}.npy', arr=task_inst.reg_mem)
        else:
            raise NotImplementedError(f"Memory {mem} not implemented")

    @staticmethod
    def construct_general_mem(task_inst, force=False):
        problem.construct_mem(task_inst, mem='util', force=force)
        problem.construct_mem(task_inst, mem='reg', force=force)

class hotelling3(problem):
    def __init__(self, dim=2, x_opt=0.5):
        self.xs = []
        self.fs = []
        self.x_ne = np.array([[x_opt]*dim, [x_opt]*dim, [x_opt]*dim])
        self.dim = dim
        # self.x_space = cartesian_product_transpose(*([np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]*dim))
        self.x_space = cartesian_product_transpose(*([np.array([0. , 0.25, 0.5, 0.75, 1.0])]*dim))
        self.train_x = np.array(list(itertools.product(*[self.x_space]*3)))
        self.__name__ = f'hotelling_3agents_{dim}_{x_opt}_{self.train_x.shape[0]}'
        problem.construct_general_mem(self)


    def evaluate_regret(self, x):
        if len(x) > 2:
            x = x.reshape([3, 2])
        if hasattr(self, 'util_mem'):
            val1, val2, val3 = self.query_utilities(x)
        else:
            val1, val2, val3 = self.utilities(x)
        res = 0.0
        rgt1, rgt2, rgt3 = val1, val2, val3
        for mis_x in self.x_space:
            rgt1 = max(rgt1, self.utilities(np.array([mis_x, x[1], x[2]]))[0])
            rgt2 = max(rgt2, self.utilities(np.array([x[0], mis_x, x[2]]))[1])
            rgt3 = max(rgt3, self.utilities(np.array([x[0], x[1], mis_x]))[2])
        return rgt1 - val1 + rgt2 - val2 + rgt3 - val3

    def compute_utilities(self, x):
        return self.utilities(x)

    def utilities(self, x):
        if len(x) > 2:
            x = x.reshape([3, 2])
        val0, val1, val2 = area_ratio3(x[0][0], x[0][1], x[1][0], x[1][1], x[2][0], x[2][1])
        val0 = np.array(val0)
        val1 = np.array(val1)
        val2 = np.array(val2)
        return val0, val1, val2
    

    def regrets(self, x):
        return self.query_regrets(x)

    def query_regrets(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.reg_mem[idx]

    def query_utilities(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.util_mem[idx]

class hotelling(problem):
    def __init__(self, dim=2, x_opt=0.5):
        self.xs = []
        self.fs = []
        self.x_ne = np.array([[x_opt]*dim, [x_opt]*dim])
        self.dim = dim
        self.x_space = cartesian_product_transpose(*([np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]*dim))
        self.train_x = np.array(list(itertools.product(*[self.x_space]*2)))
        self.__name__ = f'hotelling_{dim}_{x_opt}_{self.train_x.shape[0]}'
        problem.construct_general_mem(self)


    def evaluate_regret(self, x):
        if len(x) > 2:
            x = x.reshape([2, 2])
        if hasattr(self, 'util_mem'):
            val1, val2 = self.query_utilities(x)
        else:
            val1, val2 = self.utilities(x)
        res = 0.0
        rgt1, rgt2 = val1, val2
        for mis_x in self.x_space:
            rgt1 = max(rgt1, self.utilities(np.array([mis_x, x[1]]))[0])
            rgt2 = max(rgt2, self.utilities(np.array([x[0], mis_x]))[1])
        return rgt1 - val1 + rgt2 - val2

    def compute_utilities(self, x):
        return self.utilities(x)

    def utilities(self, x):
        if len(x) > 2:
            x = x.reshape([2,  2])
        val0, val1 = area_ratio(x[0][0], x[0][1], x[1][0], x[1][1])
        val0 = np.array(val0)
        val1 = np.array(val1)
        return val0, val1
    

    def regrets(self, x):
        return self.query_regrets(x)

    def query_regrets(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.reg_mem[idx]

    def query_utilities(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.util_mem[idx]
    
class BudgetAllocation(problem):
    def __init__(self, n_agents: int=2, n_channels: int=4, n_customers: int=12, seed: int=10):#, budget: int=11, unit_cost: List[int]=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], channel_capacity: List[int]=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], activation_prob: List[List[float]]=[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]):
        """
        n_agents: number of agents
        n_channels: number of channels
        n_customers: number of customers
        budget: budget
        unit_cost: unit cost for each channel
        channel_capacity: capacity for each channel
        activation_prob: n_channels x n_customers - activation probability for each channel and each customer
        """
        self.n_agents = n_agents
        self.n_channels = n_channels
        self.n_customers = n_customers
        self.seed = seed
        np.random.seed(seed)
        budget = n_channels + np.random.randint(1, 10)
        unit_cost = np.random.randint(1, 3, n_channels)
        channel_capacity = np.random.randint(1, 5, n_channels)
        activation_prob = np.random.rand(n_channels, n_customers)
        self.budget = budget
        self.unit_cost = unit_cost
        self.channel_capacity = channel_capacity
        self.activation_prob = activation_prob
        self.x_space = self.construct_strategy_space(self.n_channels, self.budget, self.unit_cost, self.channel_capacity)
        self.dim = self.n_channels
        self.train_x = np.array(list(itertools.product(*[self.x_space]*self.n_agents)))
        print(f"length of train_x: {len(self.train_x)}; length of x_space: {len(self.x_space)}")
        self.__name__ = f'BudgetAllocation_{self.n_agents}_{self.n_channels}_{self.n_customers}'
        problem.construct_general_mem(self)

    def construct_strategy_space(self, n_channels, budget, unit_cost, channel_capacity):
        vectors = []

        def generate_valid_vectors(current_vector, remaining_channels):
            if remaining_channels == 0:
                if np.dot(current_vector, unit_cost) < budget and all(current_vector_i <= channel_capacity_i for current_vector_i, channel_capacity_i in zip(current_vector, channel_capacity)):
                    vectors.append(current_vector.copy())
                return

            for value in range(min(budget, max(channel_capacity)) + 1):
                current_vector.append(value)
                generate_valid_vectors(current_vector, remaining_channels - 1)
                current_vector.pop()

        generate_valid_vectors([], n_channels)
        return np.array(vectors)

    def evaluate_regret(self, x):
        if len(x) > 2:
            x = x.reshape([self.n_agents,  self.dim])
        if hasattr(self, 'util_mem'):
            vals = self.query_utilities(x)
        else:
            vals = self.utilities(x)
        res = 0.0
        rgts = vals.copy()
        for i in range(self.n_agents):
            for mis_x in self.x_space:
                x_copy = x.copy()
                x_copy[i] = mis_x
                rgts[i] = max(rgts[i], self.utilities(x_copy)[i])
            res += rgts[i] - vals[i]
        return res

    def compute_utilities(self, x):
        return self.utilities(x)

    def utilities(self, x):
        if len(x) > 2:
            x = x.reshape([self.n_agents,  self.dim])
        def compute_activation_prob(x_i, customer):
            res = 1.0
            for channel in range(self.n_channels):
                res *= self.activation_prob[channel][customer] ** x_i[channel]
            return 1.0 - res
        # generate all permutation of self.n_agents
        agents_permutation = list(itertools.permutations(range(self.n_agents)))
        utilities = np.zeros(self.n_agents)
        for n in range(self.n_agents):
            res_n = 0.0
            for customer in range(self.n_customers):
                for permutation in agents_permutation:
                    agents_before_n = []
                    for index,p in enumerate(permutation):
                        if p < n:
                            agents_before_n.append(p)
                    res_n += compute_activation_prob(x[n], customer) * np.prod([1.0 - compute_activation_prob(x[i], customer) for i in agents_before_n])
            res_n /= len(agents_permutation)
            utilities[n] = res_n + 0.025*np.random.randn()
        return utilities
                    

    def regrets(self, x):
        return self.query_regrets(x)

    def query_regrets(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.reg_mem[idx]

    def query_utilities(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.util_mem[idx]

class RPS(problem):
    def __init__(self, dim=3, x_opt=0.33):
        self.xs = []
        self.fs = []
        self.x_ne = np.array([[x_opt]*dim, [x_opt]*dim])
        self.dim = dim
        # self.x_space = cartesian_product_transpose(*([np.linspace(0, 1, 21)]*dim))
        self.x_space = cartesian_product_transpose(*([np.linspace(0, 1, 11)]*dim))
        # revmove rows that do not sum to 1
        self.x_space = self.x_space[np.where(np.sum(self.x_space, axis=1) == 1)[0]]
        self.train_x = np.array(list(itertools.product(*[self.x_space]*2)))
        self.__name__ = f'RPS_{dim}_{x_opt}_{self.train_x.shape[0]}'
        problem.construct_general_mem(self)

    def compute_utilities(self, x):
        return self.utilities(x)

    def query_regrets(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.reg_mem[idx]

    def query_utilities(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.util_mem[idx]
    
    def utilities(self, x):
        if len(x) > 2:
            x = x.reshape([2, 3])
        val0 = (x[0][1] - x[0][2]) * x[1][0] + (x[0][2] - x[0][0]) * x[1][1] + (x[0][0] - x[0][1]) * x[1][2] + 0.025*np.random.randn()
        val1 = (x[1][1] - x[1][2]) * x[0][0] + (x[1][2] - x[1][0]) * x[0][1] + (x[1][0] - x[1][1]) * x[0][2] + 0.025*np.random.randn()
        # val0 = (x[1] - self.x_ne[1])**2 - (x[0] - self.x_ne[0])**2 + 0.025*np.random.randn()
        # val1 = - val0
        return val0, val1
    
    def regrets(self, x):
        return self.query_regrets(x)

    def evaluate_regret(self, x):
        if len(x) > 2:
            x = x.reshape([2, 3])
        if hasattr(self, 'util_mem'):
            val1, val2 = self.query_utilities(x)
        else:
            val1, val2 = self.utilities(x)
        res = 0.0
        rgt1, rgt2 = val1, val2
        for mis_x in self.x_space:
            rgt1 = max(rgt1, self.utilities(np.array([mis_x, x[1]]))[0])
            rgt2 = max(rgt2, self.utilities(np.array([x[0], mis_x]))[1])
        return rgt1 - val1 + rgt2 - val2

class Saddle(problem):
    def __init__(self, dim=2, x_opt=0.5):
        self.xs = []
        self.fs = []
        self.x_ne = np.array([x_opt]*dim)
        self.dim = dim
        self.x_space = np.linspace(0, 1, 101)
        self.train_x = np.array(list(itertools.product(*[self.x_space]*2)))
        self.__name__ = f'Saddle_{dim}_{x_opt}_{self.train_x.shape[0]}'
        problem.construct_general_mem(self)

    def compute_utilities(self, x):
        return self.utilities(x)

    def query_regrets(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.reg_mem[idx]

    def query_utilities(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.util_mem[idx]

    def utilities(self, x):
        val0 = (x[1] - self.x_ne[1])**2 - (x[0] - self.x_ne[0])**2 + 0.025*np.random.randn()
        val1 = - val0
        return val0, val1

    def regrets(self, x):
        return self.query_regrets(x)

    def evaluate_regret(self, x):
        if hasattr(self, 'util_mem'):
            val1, val2 = self.query_utilities(x)
        else:
            val1, val2 = self.utilities(x)
        res = 0.0
        rgt1, rgt2 = val1, val2
        for mis_x in self.x_space:
            rgt1 = max(rgt1, self.utilities(np.array([mis_x, x[1]]))[0])
            rgt2 = max(rgt2, self.utilities(np.array([x[0], mis_x]))[1])
        return rgt1 - val1 + rgt2 - val2

class Saddle_var(problem):
    '''
    Different version of saddle point game, with different grid distribution.
    '''
    def __init__(self, dim=2, x_opt=0.5):
        self.xs = []
        self.fs = []
        self.x_ne = np.array([x_opt]*dim)
        self.dim = dim
        self.x_space = np.linspace(0, 1, 21)
        self.train_x = np.array(list(itertools.product(*[self.x_space]*2)))
        self.__name__ = f'Saddle_{dim}_{x_opt}_{self.train_x.shape[0]}'
        problem.construct_general_mem(self)

    def compute_utilities(self, x):
        return self.utilities(x)

    def query_regrets(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.reg_mem[idx]

    def query_utilities(self, x):
        idx = find_nearest_points(x, self.train_x, dim=None)
        return self.util_mem[idx]

    def utilities(self, x):
        val0 = (x[1] - self.x_ne[1])**2 - (x[0] - self.x_ne[0])**2 + 0.025*np.random.randn()
        val1 = - val0
        return val0, val1

    def regrets(self, x):
        return self.query_regrets(x)

    def evaluate_regret(self, x):
        if hasattr(self, 'util_mem'):
            val1, val2 = self.query_utilities(x)
        else:
            val1, val2 = self.utilities(x)
        res = 0.0
        rgt1, rgt2 = val1, val2
        for mis_x in self.x_space:
            rgt1 = max(rgt1, self.utilities(np.array([mis_x, x[1]]))[0])
            rgt2 = max(rgt2, self.utilities(np.array([x[0], mis_x]))[1])
        return rgt1 - val1 + rgt2 - val2


ignore_list = ['random']