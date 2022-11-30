"""
Modified version of:
<https://github.com/binghong-ml/retro_star/alg.syn_route.py>
"""

from queue import Queue
from graphviz import Digraph

class SynRoute:
    def __init__(self, target_mol):
        self.target_mol = target_mol
        self.mols = [target_mol]
        self.values = [None]
        self.templates = [None]
        self.parents = [-1]
        self.children = [None]
        self.length = 0
        # template probality
        self.probs = [None]
        # template Q0 given by Experience Guidance Network
        self.Q0s = [None]

    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)
        self.values.append(None)
        self.templates.append(None)
        self.probs.append(None)
        self.Q0s.append(None)
        self.parents.append(parent_id)
        self.children.append(None)

        self.children[parent_id].append(len(self.mols)-1)

    def set_value(self, mol, value):
        assert mol in self.mols

        mol_id = self.mols.index(mol)
        self.values[mol_id] = value

    def add_reaction(self, mol, value, template, prob, Q0, reactants):
        assert mol in self.mols
        self.length += 1
        parent_id = self.mols.index(mol)
        self.values[parent_id] = value
        self.templates[parent_id] = template
        self.Q0s[parent_id] = Q0
        self.probs[parent_id] = prob
        self.children[parent_id] = []

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def viz_route(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr('node', shape='box')
        G.format = 'pdf'

        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            # if self.templates[i] is not None:
            #     name += ' | %s' % self.templates[i]
            names.append(name)

        node_queue = Queue()
        node_queue.put((0,-1))   # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            if parent_idx >= 0:
                # label = "prob:%.3f|Q0:%.3f" %(self.probs[parent_idx], self.Q0s[parent_idx])
                label = ''
                G.edge(names[parent_idx], names[idx], label=label)

            if self.children[idx] is not None:
                for c in self.children[idx]:
                    node_queue.put((c, idx))

        G.render()

    def serialize_reaction(self, idx):
        s = self.mols[idx]
        if self.children[idx] is None:
            return s

        # label = "prob:%.3f|Q0:%.3f" %(self.probs[idx], self.Q0s[idx])
        label = ''
        s += '>'+label+'>'
        s += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            s += '.'
            s += self.mols[self.children[idx][i]]

        return s

    def serialize(self):
        s = self.serialize_reaction(0)
        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                s += '|'
                s += self.serialize_reaction(i)

        return s 