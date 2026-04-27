import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mdp import get_random_policy, ClosedFormMDP
from mdp._mdp_utils import get_policy_from_dict
from policy_iteration._standard import StandardPolicyIteration
from policy_evaluation._linear import LinearSystemEvaluator
from policy_improvement._standard import StandardPolicyImprover

class Analyzer:
    def __init__(self, mdp, gamma):
        """
        :param mdp: The MDP environment (e.g., LakeMDP)
        :param gamma: Discount factor
        """
        self.mdp = mdp
        self.mdp_closed = ClosedFormMDP.from_mdp(mdp)
        self.gamma = gamma
        
        # Policy evaluator for ground truth calculations
        self.pe = LinearSystemEvaluator(mdp=self.mdp, gamma=self.gamma)

        # True values and optimal policy placeholders
        self._v = None
        self._q = None
        self._opt_policy = None

        # Tracking histories
        self._approaches = None
        self._report_histories = {}
        self._histories = {}
        self._policy_histories = {}
        self._q_values_histories = {}

    def prepare(self):
        """
        Computes optimal policy and true state values using Exact Linear Equation solving.
        Used as the 'Ground Truth' for error calculations.
        """
        pi = StandardPolicyIteration(
            init_policy=get_random_policy(mdp=self.mdp),
            policy_evaluator=self.pe,
            policy_improver=StandardPolicyImprover()
        )
        pi.run()
        self._v = self.pe.v.copy()
        self._q = self.mdp_closed.get_q_values_from_v_values(v=self._v, gamma=self.gamma)
        self._opt_policy = pi.policy_improver.policy

    def reset(self, approaches: dict):
        """
        :param approaches: Dictionary of algorithm objects (e.g., {'ADP': gpi_obj})
        """
        self._approaches = approaches
        self._report_histories = {a: [] for a in approaches}
        self._histories = {a: [] for a in approaches}
        self._policy_histories = {a: [] for a in approaches}
        self._q_values_histories = {a: [] for a in approaches}

        if self._v is None:
            self.prepare()

    def step(self):
        """
        Executes one iteration for all algorithms and records performance[cite: 13].
        """
        for a_name, a_obj in self._approaches.items():
            report = a_obj.step()
            self._report_histories[a_name].append(report)

            # Estimated values from the learning algorithm
            v_est = a_obj.v
            
            # Ground truth for the CURRENT policy being followed [cite: 15]
            self.pe.reset(a_obj.policy)
            v_true_current_pi = self.pe.v
            
            # Record current policy (non-terminal states only)
            states = [s for s in self.mdp_closed.states if not self.mdp_closed.is_terminal_state(s)]
            self._policy_histories[a_name].append({s: a_obj.policy(s) for s in states})
            self._q_values_histories[a_name].append(a_obj.q.copy())

            # Errors: [0] relative to current policy, [1] relative to optimal V* [cite: 18]
            diff_actual_policy = np.array([v_est.get(s, 0) - v_true_current_pi[s] for s in self.mdp_closed.states])
            diff_optimal_policy = np.array([v_est.get(s, 0) - self._v[s] for s in self.mdp_closed.states])
            
            self._histories[a_name].append([diff_actual_policy, diff_optimal_policy])

    def create_error_dataframe(self, approach_name):
        """
        Creates the CSV-ready DataFrame requested in Exercise 1.2.
        """
        states = [s for s in self.mdp_closed.states if not self.mdp_closed.is_terminal_state(s)]
        data = []
        for h in self._histories[approach_name]:
            # Absolute error for each state in that iteration [cite: 18]
            errors = [np.abs(h[1][self.mdp_closed.states.index(s)]) for s in states]
            data.append(errors)
        
        return pd.DataFrame(data, columns=[str(s) for s in states])

    def get_series_of_suboptimal_actions(self, approach_name):
        """
        Tracks how many states have a sub-optimal action assigned over time.
        """
        series = []
        states = [s for s in self.mdp_closed.states if not self.mdp_closed.is_terminal_state(s)]
        opt_actions = {s: self._opt_policy(s) for s in states}
        
        for policy_map in self._policy_histories[approach_name]:
            mistakes = 0
            for s, opt_a in opt_actions.items():
                if s not in policy_map or policy_map[s] != opt_a:
                    mistakes += 1
            series.append(mistakes)
        return series

    def plot_rmse_over_time(self, ax=None):
        """
        Exercise 1.3: Shows RMSE convergence relative to the optimal policy[cite: 21].
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        for a_name in self._approaches:
            # Using absolute error vs optimal V* [cite: 18]
            rmse = [np.sqrt(np.mean(h[1]**2)) for h in self._histories[a_name]]
            ax.plot(rmse, label=a_name)
            
        ax.set_xlabel("Iterations (Trials)")
        ax.set_ylabel("RMSE vs Optimal V*")
        ax.set_title("Value Function Convergence")
        ax.legend()
        ax.grid(True)

    def get_trial_lengths(self, approach_name):
        """
        Extracts trial lengths from the step reports.
        """
        return [r.get('steps', 0) for r in self._report_histories[approach_name]]