import logging
import pickle
import pandas as pd
import networkx as nx
import json
from  tqdm import tqdm
from rdkit import Chem
from eg_mcts.utils.logger import setup_logger
from eg_mcts.utils.prepare_methods import prepare_starting_molecules, prepare_reaction_data
from eg_mcts.arg.parse_args import args
def clear_mapnum(mol):
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
    return mol

def deal_SMILES(mol_string):
    mol = Chem.MolFromSmiles(mol_string)
    clear_mapnum(mol)
    new_mol_string = Chem.MolToSmiles(mol)
    return new_mol_string


setup_logger('log.log')
ReGraph = nx.DiGraph()
staring_mols = prepare_starting_molecules('../'+args.starting_molecules)

reactions = prepare_reaction_data('../'+args.reaction_data)

def update_graph(inital = False):
    change = False
    for reaction in tqdm(reactions):
        reactants = reaction['reactants']

        reactants = reactants.split('.')
        products = reaction['products']
        products = products.split('.')
        if inital:
            in_staring_mol = True
            for i in range(len(reactants)):
                try:
                    reactants[i] = deal_SMILES(reactants[i])
                    if reactants[i] in staring_mols:
                        ReGraph.add_node(reactants[i], cost=0)
                except:
                    in_staring_mol = False
                    logging.info("deal reaction smiles failed: %s" % (reactants[i]))
                    break
            if in_staring_mol:
                in_staring_mol = set(reactants) < staring_mols
            if in_staring_mol:
                for product in products:
                    try:
                        product = deal_SMILES(product)
                        ReGraph.add_node(product, cost=1)
                        for reactant in reactants:
                            ReGraph.add_edge(reactant, product)
                        change = True
                    except:
                        logging.info("deal product smiles failed: %s" % (product))
        else:
            in_graph = True
            max_cost = 0
            for i in range(len(reactants)):
                try:
                    reactants[i] = deal_SMILES(reactants[i])
                    if reactants[i] not in ReGraph:
                        in_graph = False
                        break
                except:
                    logging.info("deal reactants smiles failed: %s" % (reactants[i]))
                    in_graph = False
                    break
                cost = ReGraph.nodes[reactants[i]]['cost']
                if cost > max_cost:
                    max_cost = cost
            if in_graph:
                for product in products:
                    try:
                        product = deal_SMILES(product)
                    except:
                        logging.info("deal product smiles failed: %s" % (product))
                        continue
                    if product in ReGraph:
                        current_cost = ReGraph.nodes[product]['cost']
                        if current_cost > max_cost+1:
                            ReGraph.nodes[product]['cost'] = max_cost+1
                            change = True
                    else:
                        change = True
                        ReGraph.add_node(product, cost=max_cost+1)
                    for reactant in reactants:
                        ReGraph.add_edge(reactant, product)
    return change

change = update_graph(inital=True)
while change:
    change = update_graph()
nx.write_gml(ReGraph, args.noc_data)

# ReGraph = nx.read_gml(args.noc_data)

