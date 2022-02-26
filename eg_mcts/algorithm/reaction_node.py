import numpy as np
import logging


class ReactionNode:
    def __init__(self, parent, pro, template, init_value):
        self.parent = parent

        self.depth = self.parent.depth + 1
        self.id = -1
        # update count
        self.count = 1
        # one_step_model pro
        self.pro = pro
        # one_step_model template
        self.template = template

        self.valid = True
        # child nodes: mol node
        self.children = []
        # given by experience guidance network
        self.value = init_value
        # Q_value :first time update = init_value
        self.Q_value = init_value
        # successfully found a valid synthesis route
        self.succ = False
        parent.children.append(self)

    def v_self(self):

        return self.Q_value

    def v_pro(self):
        return self.pro

    def v_visit_time(self):
        return self.count

    def set_invaild(self):
        self.valid = False
        newQ = -10.0
        self.Q_value = (self.Q_value * self.count + newQ) / (self.count + 1)
        self.count += 1

    # select a child mol node during selection phase
    def select_child(self):
        # check if all child nodes have been expanded
        # mol_node.open = true ( not expanded ) marked 1; expanded marked 0
        check_expansion = [1 if child.open else 0 for child in self.children]
        # not succeed marked 1
        check_success = [0 if child.succ else 1 for child in self.children]
        # not succeed and not expanded
        check_array = [check_expansion[i] & check_success[i] for i in range(len(check_expansion))]
        check = np.max(check_array)
        if check == 0:
            # no node which is not successful and not expanded
            # randomly select one not successful
            select_indexs = np.where(np.array(check_success) == 1)

            select_index = np.random.randint(0, len(select_indexs))

            return self.children[select_indexs[0][select_index]]
        else:
            # return the first one
            return self.children[np.argmax(check_array)]

    def set_success(self):
        self.succ = True
        self.Q_value = 10.0

    # update phase
    def update_Q(self):
        succe = True
        totoal_Vm = 0
        # record the child nodes information
        not_expand = 0
        for i in range(len(self.children)):
            node = self.children[i]
            succe = succe & node.succ
            if node.open:
                not_expand += 1
            else:
                totoal_Vm = totoal_Vm + node.value
        if succe:
            self.succ = True
            newQ = 10.0
            self.Q_value = (self.Q_value * self.count + newQ) / (self.count + 1)
            self.count += 1
            return
        else:
            # only use expanded child nodes
            newQ = totoal_Vm / (len(self.children) - not_expand)
            self.Q_value = (self.Q_value * self.count + newQ) / (self.count + 1)
            # self.Q_value = (self.Q_value * 0.5) + newQ * 0.5
            self.count += 1
            return

    def serialize(self):
        return '%d' % (self.id)