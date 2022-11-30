from eg_mcts.api import RSPlanner

planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=500,
    expansion_topk=500
)
result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
print(result)

