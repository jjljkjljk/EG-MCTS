import numpy as np
import logging


class MolNode:
    def __init__(self, mol, init_value=np.inf, parent=None, is_known = False, zero_known_value = True):
        self.mol = mol
        # Vm = 1 if m is known
        # Vm = -inf if m can not be expanded
        # Vm = argmax Q
        self.value = init_value
        self.parent = parent

        self.id = -1
        if self.parent is None:
            self.depth = 0
        else: 
            self.depth = self.parent.depth

        self.is_known = is_known
        self.children = []
        self.succ = is_known
        # before expansion: open = True and VM = inf, after expansion: False and has Vm
        self.open = True
        # already in B
        if is_known:
            self.open = False
            if zero_known_value:
                self.value = 10.0

        if parent is not None:
            parent.children.append(self)

    def children_list(self):
        if len(self.children) != 0:
            # all_valid reactions
            valid_child = [child.valid for child in self.children]
            valid_index = np.where(np.array(valid_child)==1)
            if len(valid_index[0]) == 0:
                return None
            return [self.children[i] for i in valid_index[0]]
        else :
            return None

    def v_self(self):
        return self.value

    def serialize(self):
        text = '%d | %s' % (self.id, self.mol)
        return text


    def get_ancestors(self):
        if self.parent is None:
            return {self.mol}
        ancestors = self.parent.parent.get_ancestors()
        ancestors.add(self.mol)
        return ancestors