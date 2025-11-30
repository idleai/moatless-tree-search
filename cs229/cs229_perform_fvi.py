from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from copy import deepcopy
import torch
from torch import nn

# from cs229_predict_state_for_fvi import MultiLayerPerceptron

"""
Parts of code drawn from CS229 PS4 inverted pendulum problem
"""

STATE_DIM = 774

ALL_POSSIBLE_ACTIONS = [i for i in range(10)]


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(775, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 774),
        )

    def forward(self, x):
        return self.layers(x)


def initialize_mdp_data(
    state_dim,
) -> (
    dict
):  # mdp_params = {"transition_counts":__, "transition_probs": __, "reward_counts": __, "reward": __, "value": __}
    """
    Returns a variable containing all the parameters/state needed for an MDP.

    Assume that no transitions or rewards have been observed.
    Initialize the value function array to small random values (0 to 0.10, say).
    Initialize the transition probabilities uniformly (ie, probability of
        transitioning for state x to state y using action a is exactly
        1/num_states).
    Initialize all state rewards to zero.

    Args:
        num_states: The number of states

    Returns: The initial MDP parameters
    """

    theta = np.random.rand(state_dim) * 0.1

    return theta


# {
# "theta": theta
# "transition_counts": transition_counts,
# "transition_probs": transition_probs,
# "reward_counts": reward_counts,
# "reward": reward,
# "value": value,
# "num_states": num_states,
# }


def calculate_state_prime(model, initial_state, action):

    state_initial_w_action = np.concatenate((initial_state, [action]))
    x = torch.tensor(state_initial_w_action, dtype=torch.float32)
    with torch.no_grad():
        state_final = model(x)

    return state_final


def get_reward(state):
    """
    See create_example() in cs229_create_fittedvalueiteration_dataset.py for
    how the state vector is constructed from individual features
    """
    return state[4]


def get_value(state, theta):
    """
    Here we assume a linear function (i.e., the value function equals the
    dot product between the state and theta vectors)
    """
    return np.dot(state, theta)


def get_q_value(state_initial, gamma, state_final, theta):
    return get_reward(state_initial) + gamma * get_value(state_final, theta)


def update_value_function(theta, x, y, eps=1e-5, step_size=1e-5):
    """
    This update rule currently assumes we are solving a linear regression problem.
    """

    n_examples, dim = x.shape

    Y = np.broadcast_to(y, (dim, n_examples))

    stopping_criterion_not_reached = True
    while stopping_criterion_not_reached:

        theta_old = deepcopy(theta)

        # mat: (dim, n_examples)
        mat = Y - np.broadcast_to(x @ theta, [dim, n_examples])

        theta = theta + step_size * np.sum(np.multiply(mat, x.T), axis=1)

        if np.linalg.norm(theta_old - theta) < eps:
            stopping_criterion_not_reached = False

    return theta


def main(plot=True):
    # Seed the randomness of the simulation so this outputs the same thing each time
    # np.random.seed(0)

    GAMMA = 0.995
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20

    # # You should reach convergence well before this
    # max_failures = 500

    # # Load model
    MODEL_PATH = "./fvi/state_predictor.pt"
    model = MultiLayerPerceptron()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # # Load buffer of example states
    # NOTE: We only need states, not transitions, because we will simulate transitions
    # using the model loaded above (a MLP) for various actions
    # The initial states (in trajectories from search trees) are only in the x dataset,
    # and the final states are only in the y dataset.
    # Hence, we will load both datasets.
    # This may lead to some redundant samples, but with randomness and a large buffer,
    # we will consider this to be OK.
    data_x = np.load("./datasets/fvi_nn_x.npy")
    data_y = np.load("./datasets/fvi_nn_y.npy")
    data_x = data_x[:, :-1]
    data = np.vstack((data_x, data_y))
    num_samples = data.shape[0]

    # Initialize parameters (of value function that we are trying to learn)
    theta = initialize_mdp_data(STATE_DIM)

    # This is the criterion to end the simulation.
    # You should change it to terminate when the previous
    # 'NO_LEARNING_THRESHOLD' consecutive value function computations all
    # converged within one value function iteration. Intuitively, it seems
    # like there will be little learning after this, so end the simulation
    # here, and say the overall algorithm has converged.

    consecutive_no_learning_trials = 0
    batch_size = 100
    error = []
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:

        # choose a random sample of states from the buffer
        idx = np.random.randint(num_samples, size=batch_size)
        data_batch = data[idx, :]
        y_batch = np.zeros((batch_size,))

        # for each sample in the batch
        for i in range(batch_size):

            state_initial = data_batch[i, :]
            q_value_per_action = np.zeros((len(ALL_POSSIBLE_ACTIONS),))

            for j, action in enumerate(ALL_POSSIBLE_ACTIONS):

                state_final = calculate_state_prime(model, state_initial, action)
                q_value_per_action[j] = get_q_value(
                    state_initial, GAMMA, state_final, theta
                )

            y_batch[i] = max(q_value_per_action)

        # update theta for this batch
        theta_old = deepcopy(theta)
        theta = update_value_function(theta, data_batch, y_batch)

        # check for convergence
        max_abs_change = np.max(np.abs(theta - theta_old))
        error.append(max_abs_change)
        if max_abs_change <= tolerance:
            converged_in_one_iteration = True
        else:
            converged_in_one_iteration = False

        print(f"Concluded iteration {len(error)}. Error: {error}.")
        # update progress towards training completion if convergence occurs
        if converged_in_one_iteration:
            consecutive_no_learning_trials = consecutive_no_learning_trials + 1
        else:
            consecutive_no_learning_trials = 0

    if plot:
        # plot the learning curve (time balanced vs. trial)
        plt.plot([i for i in range(len(error))], error, "k-")
        plt.xlabel("Num epochs")
        plt.ylabel("Error (max absolute change in theta)")
        plt.savefig("./fvi_error.pdf")

    return theta


if __name__ == "__main__":
    theta = main()
    output_dir = "./fvi"

    with open(os.path.join(output_dir, "fvi_theta.npy"), "wb") as f:
        np.save(f, theta)
