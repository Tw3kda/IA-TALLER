try:
    from ._base import GeneralPolicyIterationComponent
except ImportError:
    from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd
from abc import abstractmethod


class TrialBasedPolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float, # Note: gamma is usually passed to Workspace, but kept for signature
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
    ):
        super().__init__()
        self.trial_interface = trial_interface
        self.exploring_starts = exploring_starts
        self.max_trial_length = max_trial_length
        self.rng = random_state if random_state is not None else np.random.RandomState()

    def step(self):
        # Use the policy currently stored in the shared workspace
        current_policy = self.workspace.policy
        
        # Generate trial using the interface
        df_trial = self.trial_interface.generate_trial(
            policy=current_policy, 
            exploring_starts=self.exploring_starts,
            max_steps=self.max_trial_length,
            rng=self.rng
        )
        
        # Process and return statistics [cite: 13]
        stats = self.process_trial_for_policy(df_trial, current_policy)
        if not isinstance(stats, dict):
            stats = {}
        stats["trial_length"] = len(df_trial)
        return stats

    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError