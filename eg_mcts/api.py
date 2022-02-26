import torch
import logging
import time

from eg_mcts.utils.prepare_methods import prepare_starting_molecules, prepare_mlp, \
    prepare_egmcts_planner
from eg_mcts.utils.smiles_process import smiles_to_fp, reaction_smarts_to_fp
from eg_mcts.model.eg_network import EG_MLP
from eg_mcts.utils.logger import setup_logger
import numpy as np
import os
dirpath = os.path.dirname(os.path.abspath(__file__))


class RSPlanner:
    def __init__(self,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=False,
                 starting_molecules=dirpath+'/dataset/origin_dict.csv',
                 mlp_templates=dirpath+'/one_step_model/template_rules_1.dat',
                 mlp_model_dump=dirpath+'/one_step_model/retro_star_value_ours.ckpt',
                 save_folder=dirpath+'/saved_EG_fn',
                 value_model='bets_EGN.pt',
                 viz=False,
                 viz_dir='viz'):


        device = torch.device('cuda:%d' % gpu if gpu >= 0 else 'cpu')
        starting_mols = prepare_starting_molecules(starting_molecules)
        self.use_value_fn = use_value_fn
        one_step = prepare_mlp(mlp_templates, mlp_model_dump)

        if use_value_fn:
            print('use_fn')
            model = EG_MLP(
                n_layers=1,
                fp_dim=4096,
                latent_dim=256,
                dropout_rate=0.1,
                device=device
            ).to(device)
            model_f = '%s/%s' % (save_folder, value_model)
            logging.info('Loading Experience Guidance Network from %s' % model_f)
            model.load_state_dict(torch.load(model_f, map_location=device))
            model.eval()

            def value_fn(mol, template):
                mol_fp = smiles_to_fp(mol, fp_dim=2048).reshape(1, -1)
                template_fp = reaction_smarts_to_fp(template, fp_dim=2048).reshape(1, -1)
                fp = np.hstack((mol_fp, template_fp))
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            value_fn = lambda x,y: 0.5

        self.plan_handle = prepare_egmcts_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )

    def plan(self, target_mol, target_molid = 0):
        t0 = time.time()
        succ, route, msg, expressions = self.plan_handle(target_mol,target_molid)

        if succ:
            result = {
                'succ': succ,
                'time': time.time() - t0,
                'iter': msg[0],
                'routes': route.serialize(),
                'route_len': route.length,
                'expand_model_call': msg[1],
                'value_model_call': 0,
                'reaction_nodes_lens': msg[3],
                'mol_nodes_lens': msg[4]
            }
            if self.use_value_fn:
                result['value_model_call'] = msg[2]
            return result

        else:
            logging.info('Synthesis path for %s not found. Please try increasing '
                         'the number of iterations.' % target_mol)
            return None


if __name__ == '__main__':

    planner = RSPlanner(
        gpu=-1,
        use_value_fn=True,
        iterations=500,
        expansion_topk=50,
        viz = False
    )

    result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
    print(result)

    # result = planner.plan('CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C')
    # print(result)
    #
    # result = planner.plan('CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1')
    # print(result)

