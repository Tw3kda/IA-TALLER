from abc import ABC, abstractmethod

class PolicyEvaluator(ABC):

    def __init__(self, gamma):
        self.gamma = gamma
        self.policy = None
        self._v_values = None
        self._q_values = None

    def reset(self, policy):
        """
        :param policy: the policy (callable or object with act()) subject to evaluation
        """
        self.policy = policy
        self._after_reset()
    
    def _after_reset(self):
        """
        Hook function called after the policy has been reset. 
        Typical usage is to trigger the computation of v and q values.
        """
        pass

    @property
    @abstractmethod
    def provides_state_values(self):
        """True if this evaluator gives access to state values through the property `v`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def v(self):
        """Returns a dictionary v where v[s] estimates V^pi(s)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def q(self):
        """Returns a 2-depth dictionary q where q[s][a] estimates Q^pi(s,a)."""
        raise NotImplementedError