from abc import ABC, abstractmethod

class PolicyIteration(ABC):

    def __init__(self, policy_evaluator, policy_improver):
        self.policy_evaluator = policy_evaluator
        self.policy_improver = policy_improver

    @abstractmethod
    def step(self):
        """
        Executes one full iteration: Evaluation followed by Improvement.
        :return: True if the policy changed (improved), False if it converged.
        """
        raise NotImplementedError
    
    def run(self, max_iter=10**6):
        """
        Runs the algorithm until convergence or max_iter is reached.
        """
        for _ in range(max_iter):
            improved = self.step()
            if not improved:
                break
        return self.policy_improver.policy