from ._base import PolicyIteration

class StandardPolicyIteration(PolicyIteration):

    def __init__(self, init_policy, policy_evaluator, policy_improver):
        """
        :param init_policy: The initial policy (usually random)
        :param policy_evaluator: Evaluator (e.g., LinearSystemEvaluator)
        :param policy_improver: Improver (e.g., StandardTrialInterfaceBasedPolicyImprover)
        """
        super().__init__(policy_evaluator, policy_improver)
        self.policy = init_policy
    
    def step(self):
        # 1. Policy Evaluation: Compute V and Q for the current policy
        self.policy_evaluator.reset(self.policy)
        
        # 2. Policy Improvement: Update policy to be greedy w.r.t. the new Q-values
        # The improver returns True if the policy actually changed
        has_improved = self.policy_improver.improve(self.policy_evaluator.q)
        
        # 3. Update the local tracking of the policy
        self.policy = self.policy_improver.policy
        
        return has_improved