from csrl.mdp import GridMDP
import numpy as np 
from csrl.oa import OmegaAutomaton
from csrl import ControlSynthesis
from multi import MultiControlSynthesis
import numpy as np 


shape = n_cols, n_rows = (4,4)

structure = np.array([
['E','B','E','E'],
['E','B','E','E'],
['E','B','E','E'],
['E','B','E','E'],
['E','E','E','E']
])

label = np.empty(shape,dtype=object)
label.fill(())
label[1,0] = ('a',)
label[1,2] = ('b',)
label[3,0] = ('c',)
label[3,2] = ('d',)
    
grid_mdp = GridMDP(shape=shape,structure=structure,label=label,figsize=5)
#grid_mdp.plot()

# LTL Specification
ltl1 = '(!a U b) & F d & FG c'
ltl2 = 'F(b U a) & F c & FG d' # LTL seems to have to contain all labels for MDP

# Translate the LTL formula to an LDBA
oa1 = OmegaAutomaton(ltl1)
oa2 = OmegaAutomaton(ltl2)

print('Number of Omega-automaton states (including the trap state):',oa1.shape[1])
print('Number of Omega-automaton states (including the trap state):',oa2.shape[1])

# Construct product MDPs
csrl1 = ControlSynthesis(grid_mdp,oa1)
csrl2 = ControlSynthesis(grid_mdp,oa2) 

mcsrl = MultiControlSynthesis([csrl1,csrl2], mdp=grid_mdp)

mcsrl.combined_qlearning(T=100, K=1000)