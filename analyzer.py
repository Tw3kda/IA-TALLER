from mdp import get_random_policy, ClosedFormMDP
from mdp._mdp_utils import get_policy_from_dict
from policy_iteration._standard import StandardPolicyIteration
from policy_evaluation._linear import LinearSystemEvaluator
# Ensure this matches your actual import path for the improver
from policy_improvement._standard import StandardPolicyImprover 
import numpy as np
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, mdp, gamma):
        self.mdp = mdp
        self.mdp_closed = ClosedFormMDP.from_mdp(mdp)
        self.gamma = gamma
        self.pe = LinearSystemEvaluator(mdp=self.mdp, gamma=self.gamma)
        self._v = None
        self._q = None
        self._opt_policy = None
        self._approaches = None
        
    def prepare(self):
        pi = StandardPolicyIteration(
            init_policy=get_random_policy(mdp=self.mdp),
            policy_evaluator=self.pe,
            policy_improver=StandardPolicyImprover()
        )
        pi.run()
        self._v = self.pe.v.copy()
        # Ensure ClosedFormMDP has this method from our previous step
        self._q = self.mdp_closed.get_q_values_from_v_values(v=self._v, gamma=self.gamma)
        self._opt_policy = pi.policy_improver.policy

    def reset(self, approaches: dict):
        self._approaches = approaches
        self._histories = {a: [] for a in approaches}
        self._policy_histories = {a: [] for a in approaches}
        self._q_values_histories = {a: [] for a in approaches}
        if self._v is None:
            self.prepare()

    def step(self):
        for a_name, a_obj in self._approaches.items():
            report = a_obj.step()
            v_est = a_obj.v
            
            # Get true values for the CURRENT policy being evaluated
            self.pe.reset(a_obj.policy)
            v_true_curr = self.pe.v
            
            # Store histories
            self._policy_histories[a_name].append({s: a_obj.policy(s) for s in self.mdp_closed.states if not self.mdp_closed.is_terminal_state(s)})
            self._q_values_histories[a_name].append(a_obj.q.copy() if a_obj.q else {})
            
            # Calculate errors
            # diff_actual: error relative to current policy's true value
            # diff_optimal: error relative to the MDP's optimal value
            diff_actual = np.array([v_est.get(s, 0) - v_true_curr[s] for s in self.mdp_closed.states])
            diff_optimal = np.array([v_est.get(s, 0) - self._v[s] for s in self.mdp_closed.states])
            self._histories[a_name].append([diff_actual, diff_optimal])

    def plot_rmse_over_time(self, ax=None):
        """Implementation for Exercise 1.2"""
        if ax is None:
            fig, ax = plt.subplots()
        
        for a_name in self._approaches:
            # history[t][0] is the diff_actual_policy array
            rmse = [np.sqrt(np.mean(h[0]**2)) for h in self._histories[a_name]]
            ax.plot(rmse, label=a_name)
            
        ax.set_xlabel("Iterations")
        ax.set_ylabel("RMSE (State Values)")
        ax.legend()
        ax.set_title("Value Convergence Over Time")