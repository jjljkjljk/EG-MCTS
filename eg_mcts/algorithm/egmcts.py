import os
import numpy as np
import logging
from eg_mcts.algorithm.search_tree import SearchTree


def egmcts(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,
            iterations, viz=False, viz_dir=None):
    search_tree = SearchTree(
        target_mol=target_mol,
        building_blocks=starting_mols,
        value_fn=value_fn
    )

    i = -1
    # MCTS parameter
    C = 0.5
    if not search_tree.succ:
        for i in range(iterations):
            # apply tree policy until reach the leaf node
            m_next = search_tree.root
            last_R_visit_time = i+1
            while m_next.children_list() != None:
                # at mol node select according to PUCT
                reaction_nodes = m_next.children_list()

                reaction_Qvalue = [reaction_node.v_self() for reaction_node in reaction_nodes]

                reaction_pro = [reaction_node.v_pro() for reaction_node in reaction_nodes]

                reaction_visit_time = [reaction_node.v_visit_time() for reaction_node in reaction_nodes]

                reaction_a = [reaction_Qvalue[i]/reaction_visit_time[i]+C*reaction_pro[i]*(last_R_visit_time)**0.5/(1+reaction_visit_time[i]) for i in range(len(reaction_pro))]

                m_next = reaction_nodes[np.argmax(reaction_a)]

                last_R_visit_time = m_next.v_visit_time()

                # at reaction node see reaction_node.select_child()
                m_next = m_next.select_child()

            if m_next == search_tree.root and m_next.open == False:
                break
            assert m_next.open
            # one step model
            result = expand_fn(m_next.mol)
            search_tree.expand_model_call += 1

            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                pros = result['scores']
                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    templates = result['template']
                reactant_lists = []

                for j in range(len(pros)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                # expansion phase
                search_tree.expand(m_next, reactant_lists, pros, templates)
                # update phase
                succ = search_tree.update(m_next)
                if succ:
                    break
            else:
                search_tree.expand(m_next, None, None, None)
                succ = search_tree.update(m_next)


        logging.info('information: reation_node number:  %d  |  mol_node number:  %d  | iter:  %d' % (
        len(search_tree.reaction_nodes), len(search_tree.mol_nodes), i+1))
        logging.info('Final search status | expand model call | value model call | iter: %s | %d | %d | %d'
                     % (search_tree.succ, search_tree.expand_model_call, search_tree.value_model_call, i+1))

    solution_route = None
    if search_tree.succ:
        solution_route = search_tree.get_solution_route(iterations+1)
        assert solution_route is not None

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if search_tree.succ:

            f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            solution_route.viz_route(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        search_tree.viz_search_tree(f)
    expressions = search_tree.get_tree_Qvalue()

    return search_tree.succ, solution_route, (i+1, search_tree.expand_model_call, search_tree.value_model_call, len(search_tree.reaction_nodes), len(search_tree.mol_nodes)), expressions
