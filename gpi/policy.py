from __future__ import annotations
import time
import numpy as np

from connect4.policy import Policy
from connect4.trial_interface import Connect4TrialInterface

# Import your GPI components
# Ensure these paths match your folder structure
from gpi._adp_policy_evaluation import ADPPolicyEvaluation
from gpi._standard_trial_interface_based_policy_improver import StandardTrialInterfaceBasedPolicyImprover

class Connect4LearnedPolicy(Policy):
    """
    Concrete implementation of the Policy object required by Exercise 3.
    """
    def __init__(self, improver: StandardTrialInterfaceBasedPolicyImprover):
        self.improver = improver

    def act(self, s):
        # The improver already knows how to pick the best action based on Q-values
        return self.improver.get_action(s)

def learn_policy(trial_interface: Connect4TrialInterface, timeout: int | None = None) -> Policy:
    # 1. Instantiate components
    improver = StandardTrialInterfaceBasedPolicyImprover(trial_interface)
    evaluator = ADPEvaluator(trial_interface, gamma=0.9, exploring_starts=True)
    
    # 2. Initialize the GPI orchestrator provided in your _base.py
    gpi_engine = GeneralPolicyIteration(gamma=0.9, components=[evaluator, improver])
    
    start_time = time.time()
    while True:
        if timeout and (time.time() - start_time) > timeout:
            break
        
        # This calls step() on both evaluator and improver [cite: 10]
        gpi_engine.step()
        
        # Break condition for non-timeout runs
        if not timeout: break 

    return gpi_engine.policy # Returns the improver/policy from workspace [cite: 10]