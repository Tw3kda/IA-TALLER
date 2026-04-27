import numpy as np

from mdp._trial_interface import TrialInterface
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator


class FirstVisitMonteCarloEvaluator(TrialBasedPolicyEvaluator):

    def __init__(self, trial_interface, gamma, exploring_starts, max_trial_length=np.inf, random_state=None):
        super().__init__(trial_interface, gamma, exploring_starts, max_trial_length, random_state)
        self.returns = {} 

    def process_trial_for_policy(self, df_trial, policy):
        g = 0
        # Access shared workspace data
        gamma = self.workspace.gamma
        current_v = self.workspace.v if self.workspace.v is not None else {}
        new_v = current_v.copy()
        
        # Calculate returns backwards
        for i in range(len(df_trial) - 1, -1, -1):
            row = df_trial.iloc[i]
            s = row['state']
            r = row['reward']
            g = gamma * g + r
            
            # First-visit logic
            if s not in df_trial.iloc[:i]['state'].values:
                if s not in self.returns:
                    self.returns[s] = []
                self.returns[s].append(g)
                new_v[s] = np.mean(self.returns[s])
        
        # Update the central workspace [cite: 10]
        self.workspace.replace_v(new_v)
        return {"states_updated": len(new_v)}