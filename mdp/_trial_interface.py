import numpy as np
import pandas as pd

class TrialInterface:
    def __init__(self, mdp, seed=None):
        self.mdp = mdp
        self.rs = np.random.RandomState(seed)

    def generate_trial(self, policy, exploring_starts=False, max_steps=np.inf, rng=None):
        """
        Generates a trial for Policy Evaluation.
        :param exploring_starts: If True, pick the first state and action randomly.
        """
        local_rs = rng if rng is not None else self.rs
        
        # Determine starting state
        if exploring_starts:
            # Pick any non-terminal state randomly
            non_terminals = [s for s in self.mdp.states if not self.mdp.is_terminal_state(s)]
            s = non_terminals[local_rs.choice(len(non_terminals))]
        else:
            # Pick from MDP initial states
            s = self.mdp.init_states[local_rs.choice(len(self.mdp.init_states))]

        rows = []
        steps = 0
        while not self.mdp.is_terminal_state(s) and steps < max_steps:
            # Determine action
            if exploring_starts and steps == 0:
                # First action is random if exploring starts is enabled
                available_actions = self.mdp.get_actions_in_state(s)
                a = available_actions[local_rs.choice(len(available_actions))]
            else:
                # Use the policy (the improver's act/get_action method)
                a = policy.act(s) if hasattr(policy, 'act') else policy(s)
            
            r = self.mdp.get_reward(s)
            rows.append({"state": s, "action": a, "reward": r})
            
            # Transition
            dist = self.mdp.get_transition_distribution(s, a)
            succs = list(dist.keys())
            probs = [dist[k] for k in succs]
            s = succs[local_rs.choice(len(succs), p=probs)]
            steps += 1
            
        # Add final terminal state
        rows.append({"state": s, "action": None, "reward": self.mdp.get_reward(s)})
        return pd.DataFrame(rows)
        
    def get_actions(self, state):
        """Used by StandardTrialInterfaceBasedPolicyImprover[cite: 24]."""
        return self.mdp.get_actions_in_state(state)