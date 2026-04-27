import numpy as np

def get_random_policy(mdp, seed=None, deterministic=True):
    """
    Creates a random policy for the MDP[cite: 29].
    :param mdp: The MDP object to derive actions from.
    :param seed: Seed for the RandomState[cite: 7].
    :param deterministic: If True, the policy always returns the same action for a state once picked.
    :return: A function/callable that accepts a state 's' and returns an action.
    """
    rs = np.random.RandomState(seed)
    choices = {}

    def policy(s):
        # Check if we already made a deterministic choice for this state
        if deterministic and s in choices:
            return choices[s]
            
        actions = mdp.get_actions_in_state(s)
        if not actions:
            # According to common MDP logic, terminal states have no actions.
            return None 
            
        # Sample an action using the provided RandomState 
        action = actions[rs.choice(len(actions))]
        
        if deterministic:
            choices[s] = action
            
        return action

    return policy

def get_policy_from_dict(action_map):
    """
    Converts a dictionary mapping (e.g., a learned Q-table or optimal policy) into a callable.
    :param action_map: Keys are states, values are actions.
    :return: A lambda function f(s) that returns action_map[s].
    """
    return lambda s: action_map.get(s)