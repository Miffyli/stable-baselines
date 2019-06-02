import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO
from stable_baselines.common.identity_env import IdentityEnv
from stable_baselines.common.vec_env import DummyVecEnv

MODEL_LIST = [
    A2C,
    ACER,
    ACKTR,
    DQN,
    PPO1,
    PPO2,
    TRPO,
]

@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_load_parameters(model_class):
    """
    Test if ``load_parameters`` loads given parameters correctly (the model actually changes)
    and that the backwards compatability with a list of params works

    :param model_class: (BaseRLModel) A RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    # create model
    model = model_class(policy="MlpPolicy", env=env)

    # test action probability for given (obs, action) pair
    env = model.get_env()
    obs = env.reset()
    observations = np.array([obs for _ in range(10)])
    observations = np.squeeze(observations)

    actions = np.array([env.action_space.sample() for _ in range(10)])
    original_actions_probas = model.action_probability(observations, actions=actions)

    # Get dictionary of current parameters
    params = model.get_parameters()
    # Modify all parameters to be random values
    random_params = dict((param_name, np.random.random(size=param.shape)) for param_name, param in params.items())
    # Update model parameters with the new zeroed values
    model.load_parameters(random_params)
    # Get new action probas
    new_actions_probas = model.action_probability(observations, actions=actions)

    # Check that at least some action probabilities are different now
    assert not np.any(np.isclose(original_actions_probas, new_actions_probas)), "Action probabilities did not change " \
                                                                                "after changing model parameters."
    # Also check that new parameters are there (they should be random_params)
    new_params = model.get_parameters()
    comparisons = [np.all(np.isclose(new_params[key], random_params[key])) for key in random_params.keys()]
    assert all(comparisons), "Parameters of model are not the same as provided ones."


    # Now test the backwards compatibility with params being a list instead of a dict.
    # Since `get_parameters` returns a dictionary, we can not trust the ordering (prior Python 3.7),
    # we get the exact ordering from private method `_get_parameter_list()`.
    # Same function is used in case .pkl files store a list, so this test will also cover that
    # scenario.
    tf_param_list = model._get_parameter_list()
    # Make random parameters negative to make sure the results should be different from
    # previous random values
    random_param_list = [-np.random.random(size=tf_param.shape) for tf_param in tf_param_list]
    model.load_parameters(random_param_list)

    # Compare results against the previous load
    new_actions_probas_list = model.action_probability(observations, actions=actions)
    assert not np.any(np.isclose(new_actions_probas, new_actions_probas_list)), "Action probabilities did not " \
                                                                                "change after changing model " \
                                                                                "parameters (list)."


    # Test `exact_match` functionality of load_parameters
    # Create dictionary with one variable name missing
    truncated_random_params = dict((param_name, np.random.random(size=param.shape)) 
                                   for param_name, param in params.items())
    # Remove some element
    _ = truncated_random_params.pop(list(truncated_random_params.keys())[0])
    # With exact_match=True, this should be an expection
    with pytest.raises(RuntimeError):
        model.load_parameters(truncated_random_params, exact_match=True)
    # Make sure we did not update model regardless
    new_actions_probas = model.action_probability(observations, actions=actions)
    assert np.all(np.isclose(new_actions_probas_list, new_actions_probas)), "Action probabilities changed " \
                                                                            "after load_parameters raised " \
                                                                            "RunTimeError (exact_match=True)."

    # With False, this should be fine
    model.load_parameters(truncated_random_params, exact_match=False)
    # Also check that results changed, again
    new_actions_probas = model.action_probability(observations, actions=actions)
    assert not np.any(np.isclose(new_actions_probas_list, new_actions_probas)), "Action probabilities did not " \
                                                                                "change after changing model " \
                                                                                "parameters (exact_match=False)."

    del model, env
