import numpy as np
import torch
import random
import logging
import time
import pickle
import os
import math
from eg_mcts.arg.parse_args import args
from eg_mcts.utils.prepare_methods import prepare_starting_molecules, prepare_mlp, prepare_egmcts_planner
from eg_mcts.utils.smiles_process import  smiles_to_fp, reaction_smarts_to_fp
from eg_mcts.model.eg_network import EG_MLP
from eg_mcts.utils import setup_logger
from rdkit import Chem, DataStructs

def EG_MCTS_plan():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    starting_mols = prepare_starting_molecules(args.starting_molecules)

    routes = []
    route_file_name = args.test_mols
    for line in open(route_file_name, "r"):
        routes.append(line)
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         route_file_name))

    one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump)

    # create result folder
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    if args.use_value_fn:

        print('use_value_fn')
        model = EG_MLP(
            n_layers=args.n_layers,
            fp_dim=args.value_fp_dim,
            latent_dim=args.latent_dim,
            dropout_rate=0.1,
            device=device
        ).to(device)
        model_f = '%s/%s' % (args.save_folder, args.value_model)
        logging.info('Loading Experience Guidance Network from %s' % model_f)
        model.load_state_dict(torch.load(model_f,  map_location=device))

        model.eval()
        def value_fn(mol, template):
            mol_fp = smiles_to_fp(mol, fp_dim=args.mol_fp_dim).reshape(1,-1)
            template_fp = reaction_smarts_to_fp(template, fp_dim=args.template_fp_dim).reshape(1,-1)
            fp = np.hstack((mol_fp, template_fp))
            fp = torch.FloatTensor(fp).to(device)
            v = model(fp).item()
            return v
    else:
        logging.info('Not use Experience Guidance Network' )
        value_fn = lambda x,y: 0.5

    plan_handle = prepare_egmcts_planner(
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir
    )
    # all results
    result = {
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_lens': [],
        'reaction_nodes_lens': [],
        'mol_nodes_lens': [],
        'expand_model_call': [],
        'value_model_call': []
    }
    num_targets = len(routes)
    t0 = time.time()
    all_experience = {
        'mol':[],
        'template':[],
        'Q_value':[]
    }
    i = 0
    experience_filter = {}
    while 1:
        route = routes[i]
        target_mol = route
        print(i, target_mol)

        try:

            succ, route, msg, experience = plan_handle(target_mol, i)
        except:
            continue
        result['succ'].append(succ)
        result['cumulated_time'].append(time.time() - t0)
        result['iter'].append(msg[0])
        result['routes'].append(route)
        result['expand_model_call'].append(msg[1])
        if args.use_value_fn:
            result['value_model_call'].append(msg[2])
        else:
            result['value_model_call'].append(0)
        result['reaction_nodes_lens'].append(msg[3])
        result['mol_nodes_lens'].append(msg[4])
        if succ:
            result['route_lens'].append(route.length)
        else:
            result['route_lens'].append(None)

        tot_num = i + 1
        tot_succ = np.array(result['succ']).sum()
        avg_time = (time.time() - t0) * 1.0 / tot_num
        avg_expand_model_call = np.array(result['expand_model_call'], dtype=float).mean()
        avg_value_model_call = np.array(result['value_model_call'], dtype=float).mean()
        avg_iter = np.array(result['iter'], dtype=float).mean()
        avg_reaction_nodes_number = np.array(result['reaction_nodes_lens'], dtype=float).mean()
        avg_mol_nodes_number = np.array(result['mol_nodes_lens'], dtype=float).mean()
        logging.info(
            'Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f | avg expand_model_call: %.2f | avg value_model_call: %.2f | avg reaction_nodes_number: %.2f | avg mol_nodes_number: %.2f' %
            (tot_succ, tot_num, num_targets, avg_time, avg_iter, avg_expand_model_call, avg_value_model_call,
             avg_reaction_nodes_number, avg_mol_nodes_number))
        i += 1

        if i == len(routes):
            break
        if args.collect_expe:
            print('collect experience')
            for exp in experience:
                mol, template, Q_value = exp
                if (mol, template) in experience_filter.keys():
                    experience_filter[(mol, template)].append(Q_value)
                else:
                    experience_filter[(mol, template)] = []
                    experience_filter[(mol, template)].append(Q_value)
        break
    f = open(args.result_folder + '/plan.pkl', 'wb')
    pickle.dump(result, f)
    f.close()
    if args.collect_expe:
        for key in experience_filter.keys():
            mol, template = key
            Q_list = experience_filter[(mol, template)]
            avg_Q = sum(Q_list) / len(Q_list)
            all_experience['mol'].append(mol)
            all_experience['template'].append(template)
            if avg_Q < 0:
                avg_Q = 0.0
            elif avg_Q == 10.0:
                avg_Q = 1.0
            else:
                avg_Q = 1.0 / (1 + math.exp(-avg_Q + 6))
            all_experience['Q_value'].append(avg_Q)
        experience_file_name = '%s/%s.pkl'%(args.experience_root,args.experience_data)
        f = open(experience_file_name, 'wb')
        pickle.dump(all_experience, f)
        f.close()


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('plan.log')

    EG_MCTS_plan()