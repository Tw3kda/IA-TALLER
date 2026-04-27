from ._base import PolicyEvaluator
import numpy as np

class LinearSystemEvaluator(PolicyEvaluator):

    def __init__(self, mdp, gamma):
        super().__init__(gamma)
        self.mdp = mdp
        self.states = list(mdp.states)
        self.size = len(self.states)
        self.probs = {}
        
        # Pre-cache transitions for speed
        for s in self.states:
            if not mdp.is_terminal_state(s):
                p_s = {
                    a: mdp.get_transition_distribution(s, a) 
                    for a in mdp.get_actions_in_state(s)
                }
                if p_s:
                    self.probs[s] = p_s
        self.rewards = np.array([mdp.get_reward(s) for s in self.states])

    def _after_reset(self):
        """Computes exact V and Q values using matrix algebra."""
        P_policy = np.zeros((self.size, self.size))
        
        for i, s in enumerate(self.states):
            if s in self.probs:
                # Handle both function policies and objects with .act()
                a = self.policy(s) if callable(self.policy) else self.policy.act(s)
                
                if a in self.probs[s]:
                    for s_prime, prob in self.probs[s][a].items():
                        j = self.states.index(s_prime)
                        P_policy[i, j] = prob
                else:
                    raise ValueError(f"Action {a} not defined for state {s}")

        # Solve (I - gamma * P) * V = R
        try:
            A = np.eye(self.size) - self.gamma * P_policy
            v_sol = np.linalg.solve(A, self.rewards)
        except np.linalg.LinAlgError:
            # Fallback for gamma=1.0 or singular matrices
            gamma_safe = min(0.99999999, self.gamma)
            A = np.eye(self.size) - gamma_safe * P_policy
            v_sol = np.linalg.solve(A, self.rewards)

        self._v_values = {s: val for s, val in zip(self.states, v_sol)}
        
        # Derive Q-values from V-values: Q(s,a) = R(s) + gamma * sum(P(s'|s,a)V(s'))
        self._q_values = {}
        for i, s in enumerate(self.states):
            if s in self.probs:
                self._q_values[s] = {}
                r = self.rewards[i]
                for a, successors in self.probs[s].items():
                    expected_v = sum(p * self._v_values[ss] for ss, p in successors.items())
                    self._q_values[s][a] = r + self.gamma * expected_v

    @property
    def provides_state_values(self):
        return True

    @property
    def v(self):
        return self._v_values
    
    @property
    def q(self):
        return self._q_values