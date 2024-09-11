import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import time
import json
import pickle as pkl
import random

import sys
sys.path.insert(0, '..')
from ugi_rxn_mapper import ugi_rxn_mapper
from chemprop.args import PredictArgs
from chemprop.train import make_predictions


def get_ugi_prod(smi_list) -> str:  # generate 4c4c ugi prod from list of 4 bbs
    ugi_4c_4c_smarts = "[NH2,NH3:5].[CH1:2]=O.[C:1](=O)[OH,O-].[N+:3]#[C-:4]>>[C:1](=O)[N:5][C:2][C+0:4](=O)[N+0:3]"
    ugi_4c_4c_reaction = AllChem.ReactionFromSmarts(ugi_4c_4c_smarts)
    ugi_4c_4c_prod = ugi_4c_4c_reaction.RunReactants([Chem.MolFromSmiles(smi) for smi in smi_list])
    print(ugi_4c_4c_prod)
    return Chem.MolToSmiles(ugi_4c_4c_prod[0][0])


def set_model_args():
    """Load Chemprop model."""
    # define args for chemprop predictor
    args = PredictArgs()
    args.features_generator =  ["rdkit_2d","ifg_drugbank_2","ugi_qmdesc_atom"]
    args.number_of_molecules = 2
    args.gpu = 0
    args.checkpoint_paths = ['/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_2/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_1/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_8/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_6/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_0/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_4/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_9/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_7/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_3/model_0/model.pt', '/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_5/model_0/model.pt']
    args.no_features_scaling = False
    args.preds_path = "./preds.csv"

    # load model
    # model_objects = load_model(args)
    return args

# @silence_stdout()  
def get_expected_return(args, combination):

    smi_list = [all_bbs_list[i] for i in combination]
    name_list = [all_idx_list[i] for i in combination]
    prod_name = "_".join(name_list)
    if prod_name in prod_score_dict:
        return prod_score_dict[prod_name]

    prod_smi = get_ugi_prod(smi_list)
    mapped_rxn = ugi_rxn_mapper([prod_smi])[0]
    rxn_smi = [[mapped_rxn,"FC(F)(F)CO"]]
    score = make_predictions(args, rxn_smi)[0][0]
    prod_score_dict[prod_name] = score
    with open("prod_score_dict.json", 'w') as f:
        json.dump(prod_score_dict, f, indent=2)
    return score


if __name__ == "__main__":
    
    random.seed(42)
    amine_path = '../data/ugi/BBs_with_ace/ace_on_amine.smi'
    aldehyde_path = '../data/ugi/BBs/aldehyde.smi'
    acid_path = '../data/ugi/BBs/acid.smi'
    isocyanide_path = '../data/ugi/BBs/NC.smi'

    if os.path.exists("prod_score_dict.json"):
        with open("prod_score_dict.json", 'r') as f:
            prod_score_dict = json.load(f)
    else:
        prod_score_dict = {}
        
    args = set_model_args()



    with open(amine_path, 'r') as f:
        amine_content = f.readlines()
        amine_smi_list = [line.strip().split()[0] for line in amine_content] 
        amine_name_list = [line.strip().split()[1] for line in amine_content]

    with open(aldehyde_path, 'r') as f:
        aldehyde_content = f.readlines()
        aldehyde_smi_list = [line.strip().split()[0] for line in aldehyde_content]
        aldehyde_name_list = [line.strip().split()[1] for line in aldehyde_content]

    with open(acid_path, 'r') as f:
        acid_content = f.readlines()
        acid_smi_list = [line.strip().split()[0] for line in acid_content]
        acid_name_list = [line.strip().split()[1] for line in acid_content]

    with open(isocyanide_path, 'r') as f:
        isocyanide_content = f.readlines()
        isocyanide_smi_list = [line.strip().split()[0] for line in isocyanide_content]
        isocyanide_name_list = [line.strip().split()[1] for line in isocyanide_content]

    with open("ga_log.log", "w") as f:
        f.write(f"num of amine_ace: {len(amine_smi_list)} \n num of aldehyde: {len(aldehyde_smi_list)} \n num of acid: {len(acid_smi_list)} \n num of isocyanide: {len(isocyanide_smi_list)}")

    all_bbs_list = amine_smi_list + aldehyde_smi_list + acid_smi_list + isocyanide_smi_list
    all_idx_list = amine_name_list + aldehyde_name_list + acid_name_list + isocyanide_name_list

    # 定义参数
    POPULATION_SIZE = 200  # 种群大小
    GENERATIONS = 100  # 迭代次数
    MUTATION_RATE = 0.1  # 变异率

    # 假设我们有1000个商品，分成4个类别，每个类别的商品数量分别是
    category_sizes = [len(amine_smi_list), len(aldehyde_smi_list), len(acid_smi_list), len(isocyanide_smi_list)]

    # 初始化种群
    population = np.zeros((POPULATION_SIZE, len(category_sizes)), dtype=int)
    for i in range(len(category_sizes)):
        if i == 0:
            population[:, i] = np.random.choice(category_sizes[i], size=POPULATION_SIZE)
        else:
            population[:, i] = np.random.choice(range(np.sum(category_sizes[:i]), np.sum(category_sizes[:i+1])), size=POPULATION_SIZE)

    best_individuals, average_fittness = [], []

    for generation in tqdm(range(GENERATIONS), desc='Generation'):
        
        # 计算适应度
        fitness = np.array([get_expected_return(args, individual) for individual in population])

        # 保存当前代的最优解
        best_individuals.append(population[np.argmax(fitness)])

        # select according to fitness
        parents = population[np.random.choice(np.arange(POPULATION_SIZE), size=POPULATION_SIZE, p=fitness/fitness.sum())]

        # save average fitness of current generation
        average_fittness.append(np.mean(fitness))

        # 交叉
        np.random.shuffle(parents)
        children = np.zeros_like(parents)
        for i in range(0, POPULATION_SIZE, 2):
            crossover_point = np.random.randint(len(category_sizes))
            children[i, :crossover_point] = parents[i, :crossover_point]
            children[i, crossover_point:] = parents[i+1, crossover_point:]
            children[i+1, :crossover_point] = parents[i+1, :crossover_point]
            children[i+1, crossover_point:] = parents[i, crossover_point:]

        # 变异
        for i in range(POPULATION_SIZE):
            if np.random.random() < MUTATION_RATE:
                mutation_category = np.random.randint(len(category_sizes))
                if mutation_category == 0:
                    children[i, mutation_category] = np.random.choice(category_sizes[mutation_category])
                else:
                    children[i, mutation_category] = np.random.choice(range(np.sum(category_sizes[:mutation_category]), np.sum(category_sizes[:mutation_category+1])))

        # 更新种群
        population = children
        with open("ga_log.log", "a") as f:
            f.write(f"Generation {generation} finished at {time.ctime()}\n")
        with open("prod_score_dict.json", 'w') as f:
            json.dump(prod_score_dict, f, indent=2)
        # clear_output()
        
    # best_2000_individuals = np.array(best_individuals)[np.argsort(best_fitnesses)[-2000:]]
    with open("best_individuals.pkl", 'wb') as f:   # save best individuals of each generation
        pkl.dump(best_individuals, f)
    with open("average_fittness.pkl", 'wb') as f:
        pkl.dump(average_fittness, f)

    # for i, individual in enumerate(best_2000_individuals):
    #     with open("ga_log.log", "a") as f:
    #         f.write(f'Best combination {i+1}:', individual)
