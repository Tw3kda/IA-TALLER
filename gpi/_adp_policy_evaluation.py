from mdp._trial_interface import TrialInterface
import numpy as np

from policy_evaluation._linear import LinearSystemEvaluator
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator
from mdp._base import ClosedFormMDP


def _has_action(action):
    return action is not None and not (
        isinstance(action, (float, np.floating)) and np.isnan(action)
    )


class ADPPolicyEvaluation(TrialBasedPolicyEvaluator):

    def __init__(self, trial_interface, gamma, exploring_starts, max_trial_length=np.inf, random_state=None, 
                 precision_for_transition_probability_estimates=4, update_interval=10):
        super().__init__(trial_interface, gamma, exploring_starts, max_trial_length, random_state)
        self.transition_counts = {} # counts[s][a][s_next]
        self.reward_sums = {} # rewards[s][a]
        self.iteration_count = 0
        self.update_interval = update_interval

    def process_trial_for_policy(self, df_trial, policy):
        for i in range(len(df_trial) - 1):
            s = df_trial.iloc[i]['state']
            a = df_trial.iloc[i]['action']
            r = df_trial.iloc[i]['reward']
            s_next = df_trial.iloc[i+1]['state']
            
            if _has_action(a):
                self._synchronize_knowledge_about_states_and_actions(s, a, r)
                self.transition_counts[s][a][s_next] = self.transition_counts[s][a].get(s_next, 0) + 1
                self.reward_sums[s][a] += r

        self.iteration_count += 1
        
        # Only run the heavy linear solver periodically or based on your needs 
        if self.iteration_count % self.update_interval == 0:
            # Construct empirical MDP and solve Bellman Equations
            # (Requires using the provided LinearSystemEvaluator)
            pass 

        return {"status": "model_updated"}

    def _synchronize_knowledge_about_states_and_actions(self, s, a, r):
        if s not in self.transition_counts:
            self.transition_counts[s] = {}
            self.reward_sums[s] = {}
        if a not in self.transition_counts[s]:
            self.transition_counts[s][a] = {}
            self.reward_sums[s][a] = 0.0