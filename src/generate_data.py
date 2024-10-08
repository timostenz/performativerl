from statistics import mean, stdev
import time
import itertools
import json
from joblib import Parallel, delayed
from tqdm import tqdm

from src.performative_prediction import Performative_Prediction
from src.envs.gridworld import Gridworld
from src.utils import *


def generate_data(params):
    """"
    """
    print('Begin generating performative prediction data\n')
    start = time.time()

    # Load Experiment Mode
    gradient = params['gradient']
    sampling = params['sampling']
    occupancy_iid = params['occupancy_iid']
    # Load Experiment Parameters
    n_jobs = params['n_jobs']
    # environment parameters
    eps = params['eps']
    fbeta = params['fbeta']
    betas = params['betas']
    fgamma = params['fgamma']
    gammas = params['gammas']
    num_followers = params['num_followers']
    # perormative prediction parameters
    max_iterations = params['max_iterations']
    flamda = params['flamda']
    lamdas = params['lamdas']
    freg = params['freg']
    regs = params['regs']
    # gradient parameters
    feta = params['feta']
    etas = params['etas']
    # sampling parameters
    seeds = params['seeds']
    fn_sample = params['fn_sample']
    n_samples = params['n_samples']
    # policy gradient
    policy_gradient = params['policy_gradient']
    nus = params['nus']
    # unregularized objective
    unregularized_obj = params['unregularized_obj']
    # lagrangian
    lagrangian = params['lagrangian']
    N = params['N']
    delta = params['delta']
    B = params['B']


    # Prepare Experiment Configurations
    configs = []
    if not gradient and not sampling and not policy_gradient and not unregularized_obj:
        # iterate lamdas
        for lamda in lamdas:
            configs.append({'beta': fbeta, 'lamda': lamda, 'gamma': fgamma, 'reg': freg})
        # iterate betas
        for beta in betas:
            configs.append({'beta': beta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg})
        # iterate regs
        for reg in regs:
            configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': reg})
        # iterate gammas
        for gamma in gammas:
            configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': gamma, 'reg': freg})
        # iterate gammas and lamdas
        for lamda, gamma in itertools.product(lamdas, gammas):
            configs.append({'beta': fbeta, 'lamda': lamda, 'gamma': gamma, 'reg': freg})
    if gradient:
        # iterate etas
        assert freg == 'L2'
        for eta in etas:
            if sampling:
                configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'eta': eta, 'n_sample': fn_sample})
            else:
                configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'eta': eta})
    if sampling:
        # iterate n_samples
        for n_sample in n_samples:
            if gradient:
                configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'eta': feta, 'n_sample': n_sample})
            else:
                configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'n_sample': n_sample})
    if policy_gradient:
        # iterate nus
        for nu in nus:
            configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'eta': feta, 'nu': nu})
    if unregularized_obj:
        # iterate lamdas
        for lamda in lamdas:
            configs.append({'beta': fbeta, 'lamda': lamda, 'gamma': fgamma, 'reg': freg})

    # remove duplicates
    configs = [dict(tup) for tup in set(tuple(d.items()) for d in configs)]

    # Generate Output
    if not sampling:
        # parallelize over configurations
        with tqdm_joblib(tqdm(desc="Executing Performative Prediction", total=len(configs))) as progress_bar:
            outputs = Parallel(n_jobs=min(n_jobs, len(configs)))(
                delayed(execute_performative_prediction)(config, eps, num_followers, max_iterations, gradient, sampling, occupancy_iid, policy_gradient, unregularized_obj, lagrangian, N, delta, B)
                for config in configs
            )
    else:
        outputs = []
        # parallelize over seeds
        configs = sorted(configs, key=lambda d: d['n_sample']) 
        for config in configs:
            output = {k: v for k, v in config.items()}
            with tqdm_joblib(tqdm(desc=f"Executing Performative Prediction for n_sample={config['n_sample']}", total=len(seeds))) as progress_bar:
                tmp_output = Parallel(n_jobs=min(n_jobs, len(seeds)))(
                    delayed(execute_performative_prediction)(config, eps, num_followers, max_iterations, gradient, sampling, occupancy_iid, policy_gradient, unregularized_obj, lagrangian, N, delta, B, seed)
                    for seed in seeds
                )
            d_diffs = [tmp_output[seed]['d_diff'] for seed in seeds]
            output['d_diff_mean'] = list(map(mean, zip(*d_diffs)))
            output['d_diff_std'] = list(map(stdev, zip(*d_diffs)))
            # get trajectory length - ONLY for the first seed
            output['trajectory_length'] = tmp_output[0]['trajectory_length']
            # get state space coverage grid - ONLY for the first seed
            output['state_space_coverage_iteration_grid'] = tmp_output[0]['state_space_coverage_iteration_grid']
            # get state space coverage counts - ONLY for the first seed
            output['state_visitation_counts_iteration'] = tmp_output[0]['state_visitation_counts_iteration']
            if gradient:
                sub_gaps = [tmp_output[seed]['sub_gap'] for seed in seeds]
                output['sub_gap_mean'] = list(map(mean, zip(*sub_gaps)))
                output['sub_gap_std'] = list(map(stdev, zip(*sub_gaps)))
            outputs.append(output)

    # Store Output
    with open(f'data/outputs.json', 'w') as f:
        json.dump(outputs, f, indent=4)

    end = time.time()
    print(f'Time elapsed: {end - start}')
    print('Finish generating data\n')

    return

def execute_performative_prediction(config, eps, num_followers, max_iterations, gradient, sampling, occupancy_iid, policy_gradient, unregularized_obj, lagrangian, N, delta, B, seed=1):
    """
    """
    beta = config['beta']
    lamda = config['lamda']
    gamma = config['gamma']
    reg = config['reg']
    if gradient or policy_gradient: eta = config['eta']
    else: eta = None
    if policy_gradient: nu = config['nu']
    else: nu = None
    if sampling and not lagrangian: n_sample = config['n_sample']
    elif sampling: n_sample = config['n_sample'] // 2
    else: n_sample = None

    env = Gridworld(beta, eps, gamma, num_followers, sampling, occupancy_iid, n_sample, seed)
    algorithm = Performative_Prediction(env, max_iterations, lamda, reg, gradient, eta, sampling, occupancy_iid, n_sample, policy_gradient, nu, unregularized_obj, lagrangian, N, delta, B)

    output = {k: v for k,v in config.items()}
    algorithm.execute()
    output['d_diff'] = algorithm.d_diff
    output['trajectory_length'] = env.trajectory_length
    output['state_space_coverage_iteration_grid'] = env.state_space_coverage_iteration_grid
    output['state_visitation_counts_iteration'] = env.state_visitation_counts_iteration
    if gradient or policy_gradient or unregularized_obj:
        output['sub_gap'] = algorithm.sub_gap

    # store initial visualization of env
    vis = env._get_env_vis()
    config_name = "limit_" + f"beta={beta}_lambda={lamda}_gamma={gamma}_reg={reg}"
    if gradient: config_name += f"eta={eta}"
    if sampling:
        config_name += f"n_sample={n_sample}_seed={seed}"
    with open(f'limiting_envs/{config_name}.json', 'w') as f:
        json.dump(vis, f, indent=4)

    return output