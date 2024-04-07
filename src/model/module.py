import torch

class LargeFeatureExtractor(torch.nn.Sequential):
    '''
    Map original input into the latent space
    '''
    def __init__(self,  data_dim:int=1, low_dim:bool=True, spectrum_norm:bool=False)->None:
        super(LargeFeatureExtractor, self).__init__()
        if spectrum_norm:
            add_spectrum_norm = lambda module: torch.nn.utils.parametrizations.spectral_norm(module)
        else:
            add_spectrum_norm = lambda module: module
        self.add_module('linear1', add_spectrum_norm(torch.nn.Linear(data_dim, 10000)))
        self.add_module('Sig0', torch.nn.Sigmoid())
        self.add_module('linear2',  add_spectrum_norm(torch.nn.Linear(10000, 100)))
        if low_dim:
            self.add_module('relu3', torch.nn.LeakyReLU(negative_slope=.1))
            self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(100, 1)))
        else:
            self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(100, 10)))


class ComplexFeatureExtractor(torch.nn.Sequential):
    '''
    Map original input into the latent space
    '''
    def __init__(self,  data_dim:int=1, low_dim:bool=True, spectrum_norm:bool=False)->None:
        super(ComplexFeatureExtractor, self).__init__()
        if spectrum_norm:
            add_spectrum_norm = lambda module: torch.nn.utils.parametrizations.spectral_norm(module)
        else:
            add_spectrum_norm = lambda module: module
        self.add_module('linear1', add_spectrum_norm(torch.nn.Linear(data_dim, 100)))
        self.add_module('Sig0', torch.nn.Sigmoid())
        self.add_module('linear2',  add_spectrum_norm(torch.nn.Linear(100, 100)))
        self.add_module('Sig1', torch.nn.Sigmoid())
        self.add_module('linear3',  add_spectrum_norm(torch.nn.Linear(100, 100)))
        self.add_module('Sig3', torch.nn.Sigmoid())
        self.add_module('linear_ad1',  add_spectrum_norm(torch.nn.Linear(100, 100)))
        self.add_module('Sig4', torch.nn.Sigmoid())
        self.add_module('Sig7', torch.nn.Sigmoid())
        self.add_module('linear_ad5',  add_spectrum_norm(torch.nn.Linear(100, 100)))
        self.add_module('Sig8', torch.nn.Sigmoid())
        self.add_module('linear_ad6',  add_spectrum_norm(torch.nn.Linear(100, 50)))
        # test if using higher dimensions could be better
        if low_dim:
            self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(50, 1)))
        else:
            self.add_module('relu3', torch.nn.LeakyReLU(negative_slope=.1))
            self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(50, 10)))

class SimFeatureExtractor(torch.nn.Sequential):
    '''
    Map original input into the latent space
    '''
    def __init__(self,  data_dim:int=1, low_dim:bool=True, spectrum_norm:bool=False)->None:
        super(SimFeatureExtractor, self).__init__()
        if spectrum_norm:
            add_spectrum_norm = lambda module: torch.nn.utils.parametrizations.spectral_norm(module)
        else:
            add_spectrum_norm = lambda module: module
        self.add_module('linear1', add_spectrum_norm(torch.nn.Linear(data_dim, 50)))
        self.add_module('drop1', torch.nn.Dropout(p=.1))
        self.add_module('Sig0', torch.nn.Sigmoid())
        self.add_module('linear2',  add_spectrum_norm(torch.nn.Linear(50, 20)))
        self.add_module('drop2', torch.nn.Dropout(p=.1))
        self.add_module('Sig1', torch.nn.Sigmoid())
        self.add_module('linear3',  add_spectrum_norm(torch.nn.Linear(20, 20)))
        self.add_module('drop3', torch.nn.Dropout(p=.1))
        self.add_module('Tanh1', torch.nn.Tanh())
        self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(20, 10)))
        self.add_module('drop4', torch.nn.Dropout(p=.1))
        if low_dim:
            self.add_module('linear5',  add_spectrum_norm(torch.nn.Linear(10, 1)))
        else:
            self.add_module('relu3', torch.nn.LeakyReLU(negative_slope=.1))
            self.add_module('linear5',  add_spectrum_norm(torch.nn.Linear(20, 10)))

class DeepSimFeatureExtractor(torch.nn.Sequential):
    '''
    Map original input into the latent space
    '''
    def __init__(self,  data_dim:int=1, low_dim:bool=True, spectrum_norm:bool=False)->None:
        super(DeepSimFeatureExtractor, self).__init__()
        if spectrum_norm:
            add_spectrum_norm = lambda module: torch.nn.utils.parametrizations.spectral_norm(module)
        else:
            add_spectrum_norm = lambda module: module
        self.add_module('linear1', add_spectrum_norm(torch.nn.Linear(data_dim, 50)))
        self.add_module('drop1', torch.nn.Dropout(p=.1))
        self.add_module('Sig1', torch.nn.Sigmoid())
        self.add_module('linear2',  add_spectrum_norm(torch.nn.Linear(50, 20)))
        self.add_module('drop2', torch.nn.Dropout(p=.1))
        self.add_module('Sig2', torch.nn.Sigmoid())
        self.add_module('linear3',  add_spectrum_norm(torch.nn.Linear(20, 20)))
        self.add_module('drop3', torch.nn.Dropout(p=.1))
        self.add_module('Tanh3', torch.nn.Tanh())
        self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(20, 20)))
        self.add_module('drop4', torch.nn.Dropout(p=.1))
        self.add_module('Tanh4', torch.nn.Tanh())
        self.add_module('linear5',  add_spectrum_norm(torch.nn.Linear(20, 20)))
        self.add_module('drop5', torch.nn.Dropout(p=.1))
        self.add_module('Sig5', torch.nn.Sigmoid())
        self.add_module('linear6',  add_spectrum_norm(torch.nn.Linear(20, 20)))
        self.add_module('drop6', torch.nn.Dropout(p=.1))
        self.add_module('Sig6', torch.nn.Sigmoid())
        self.add_module('linear7',  add_spectrum_norm(torch.nn.Linear(20, 20)))
        self.add_module('drop7', torch.nn.Dropout(p=.1))
        self.add_module('Sig7', torch.nn.Sigmoid())
        self.add_module('linear8',  add_spectrum_norm(torch.nn.Linear(20, 10)))
        self.add_module('drop8', torch.nn.Dropout(p=.1))

        if low_dim:
            self.add_module('linear_last',  add_spectrum_norm(torch.nn.Linear(10, 1)))
        else:
            self.add_module('relu3', torch.nn.LeakyReLU(negative_slope=.1))
            self.add_module('linear5',  add_spectrum_norm(torch.nn.Linear(20, 10)))

class AutoDeepSimFeatureExtractor(torch.nn.Sequential):
    '''
    Map original input into the latent space
    '''
    def __init__(self,  data_dim:int=1, low_dim:bool=True, spectrum_norm:bool=False, depth:int=50)->None:
        super(AutoDeepSimFeatureExtractor, self).__init__()
        if spectrum_norm:
            add_spectrum_norm = lambda module: torch.nn.utils.parametrizations.spectral_norm(module)
        else:
            add_spectrum_norm = lambda module: module
        self.add_module('linear_b', add_spectrum_norm(torch.nn.Linear(data_dim, 20)))
        self.add_module('drop_b', torch.nn.Dropout(p=.1))
        self.add_module('Sig_b', torch.nn.Sigmoid())
        for layer in range(depth):
            self.add_module(f'linear{layer}',  add_spectrum_norm(torch.nn.Linear(20, 20)))
            self.add_module(f"BN{layer}", torch.nn.BatchNorm1d(20))
            self.add_module(f'drop{layer}', torch.nn.Dropout(p=.1))
            self.add_module(f'Sig{layer}', torch.nn.Sigmoid())

        self.add_module('linear_last0',  add_spectrum_norm(torch.nn.Linear(20, 10)))
        self.add_module('drop_last0', torch.nn.Dropout(p=.1))
        if low_dim:
            self.add_module('linear_last',  add_spectrum_norm(torch.nn.Linear(10, 1)))
        else:
            self.add_module('relu3', torch.nn.LeakyReLU(negative_slope=.1))
            self.add_module('linear5',  add_spectrum_norm(torch.nn.Linear(20, 10)))