import glob
import numpy as np
from matplotlib import pyplot as plt
from src import ignore_list
import pickle

def load_data(file_path):
    # Load the data from the pickle file
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data

def save_data(file_path, data):
    # Save the data to the pickle file
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

ACQ_TRANSLATION = {'random': 'Random',
                    'ballet': 'ARISE',
                    'ballet-local': 'ARISE-global',
                    'ucb': 'UCB',
                    'ts': 'Thompson Sampling',
                    'pred': 'Prediction',
                    'sur': 'SUR',
                    'epsilon_greedy': 'Epsilon Greedy'}



regret_summary = dict()
fontsize=15
def load_and_plot(regret_summary, task, ax, file_pattern, scale='linear', xlim=50, title='debug', idx=0):

    regret_summary[f'{task}'] = {}
    files = glob.glob(file_pattern)

    for file in files:
        _tmp = load_data(file)
        if task not in _tmp.keys():
            continue
        for _method in _tmp[task].keys():
        # _method = list(_tmp[task].keys())[0]
            if _method in ignore_list: # skip random
                continue
            if _method == 'ballet' and 'roi' not in file: # identify local
                regret_summary[task]['ballet-local'] = _tmp[task][_method]
            else:
                regret_summary[task][_method] = _tmp[task][_method]
    
    _max_init = 0
    for _method in sorted(regret_summary[task].keys()):
        _regret = regret_summary[task][_method] 
        _mean = np.mean(_regret, axis=0)
        _max_init = max(_max_init, _mean[0])

    for _method in sorted(regret_summary[task].keys()):
        _regret = regret_summary[task][_method] 
        _mean = np.mean(_regret, axis=0)
        _std = np.std(_regret, axis=0)

        # shift to right by 1 step
        _mean = np.hstack([_max_init, _mean])
        _std = np.hstack([0, _std])
        _coefficients = 1/np.sqrt(_regret.shape[0])
        _iteration = _mean.shape[0]

        if _method in ['ballet']:
            ax.plot(_mean, label=ACQ_TRANSLATION.get(_method, f'unknown-{_method}'), linewidth=4)
        else:
            ax.plot(_mean, label=ACQ_TRANSLATION.get(_method, f'unknown-{_method}'))
        ax.fill_between(range(_iteration), _mean - _std * _coefficients , _mean + _std * _coefficients, alpha=0.3)

        # ax.set_ylabel('$f(x)$')
        ax.set_xlabel(f'Iteration\n({idx}) {title}', fontsize=fontsize)
        # ax.set_title(title, fontsize=18)
        ax.set_yscale(scale)
        ax.set_xlim(0, xlim)

fig = plt.figure(figsize=(10, 5))
ax_list = [plt.subplot(1, 2, idx) for idx in range(1, 3)]


load_and_plot(regret_summary=regret_summary, ax=ax_list[0], idx='c', task = 'hotelling', file_pattern= f"results/uai_rebuttal/*_3P_dk_SimReg_B2_R10_OS200_TI10_RI1_NI10_IP*.p", scale='linear', xlim=200, title='HOTEELLING (3P)')
load_and_plot(regret_summary=regret_summary, ax=ax_list[1], idx='d', task = 'BudgetAllocation', file_pattern= f"results/uai_rebuttal/*_3P_dk_SimReg_B2_R10_OS200_TI10_RI1_NI10_IP_*.p", scale='log', xlim=200, title='BUDGET ALLOCATION (3P)')

# set legend position in the upper middle among all the subplots
handles, labels = ax_list[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', shadow=True, ncol=2)
loc = (.4, .8)
fig.legend(handles, labels, loc='upper center',  bbox_to_anchor = (0, .1, 1, 1), shadow=True, ncol=5, fontsize=fontsize)
plt.tight_layout()
plt.savefig('summary_regrets_uai_rebuttal.png', bbox_inches='tight')