from rlutils.observers.pbrl import load
import hyperrectangles as hr
import matplotlib.pyplot as plt

from config.features import F


if False: # Tree size
    from joblib import load
    m = []
    for ep in range(1000, 50000, 1000):
        m.append(len(load(f"run_logs/wandering-lion-169/checkpoint_{ep}.joblib")["tree"]))
    m.append(len(load(f"run_logs/wandering-lion-169/50000.pbrl")["tree"]))
    print(m)

EP = 40
pbrl = load(f"models/2022-03-30_10-45-07/{EP}.pbrl", P={}, features=F)

print(pbrl.Pr)

# if True:
    # hr.diagram(pbrl.tree, pred_dims=["reward"], out_name=EP, out_as="svg")

if False:
    pbrl.plot_comparison_graph(figsize=(12,12))
    plt.savefig(f"{EP}.svg", bbox_inches="tight")

# pbrlA = load("run_logs/_tree_difference_test/50.pbrl")
# pbrlB = load("run_logs/_tree_difference_test/100.pbrl")
# print(hr.rules(pbrlA.tree - pbrlB.tree))



# data_grouped = hr.group_along_dim(pbrl.tree.space, "ep")
# print(sum([x.shape[0] for x in data_grouped]))
# print(pbrl.tree.space)

# pbrl.Pr[9,3] = pbrl.Pr[3,9] = 0.5

# print(hr.rules(pbrlA.tree, pred_dims=["reward"]))
# print("====")
# print(hr.rules(pbrlB.tree, pred_dims=["reward"]))



# print(pbrl.n(pbrl.episodes[2]))
# print(pbrl.phi(pbrl.episodes[2]))
# print(pbrl.n(pbrl.episodes[4]))

# print(hr.rule(pbrl.tree.leaves[1]))

# print(hr.rule(pbrl.tree.leaves[-2]))



"""

def transition_desc(self, i1, i2, sf=3, refactor=True):
    "Describe a transition via the source and destination bounding boxes."
    b2 = self.bb_desc(i2, sf)
    if b2 == "terminal": return f"({self.bb_desc(i1, sf, keep_all=True)}) to {b2}"
    if not refactor: return f"({self.bb_desc(i1, sf, keep_all=True)}) to ({b2})"
    diff_terms, same_terms = [], []
    for b1, b2 in zip(self.bb_desc(i1, sf, keep_all=True).split(" and "), self.bb_desc(i2, sf, keep_all=True).split(" and ")):
        if b1 == b2:
            if b1 != "any": same_terms.append(b1)
        else: diff_terms.append(f"({b1} to {b2})")
    return " and ".join(diff_terms) + (" while " + " and ".join(same_terms) if same_terms else "")

"""