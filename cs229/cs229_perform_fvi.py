from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os

# from cs229_predict_state_for_fvi import MultiLayerPerceptron

"""
Parts of code drawn from CS229 PS4 inverted pendulum problem
"""

STATE_DIM = 772

ALL_POSSIBLE_ACTIONS = [i for i in range(10)]


class MultiLayerPerceptron(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(773, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 772),
        )

    def forward(self, x):
        return self.layers(x)


class MultiLayerPerceptronScalarOutput(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(772, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


##### Define the dataset


class FVI_VFDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx, :]
        x = torch.tensor(entry[:-1], dtype=torch.float32)
        y = torch.tensor(entry[-1], dtype=torch.float32)
        return x, y


# def initialize_mdp_data(
#     state_dim,
# ) -> dict:

#     theta = np.random.rand(state_dim) * 0.1
#     return theta


def normalize(x, eps=1e-12):
    if len(x.size()) == 1:
        return (x - torch.mean(x)) / torch.std(x)
    else:
        return (x - torch.mean(x, axis=1, keepdims=True)) / torch.std(
            x, axis=1, keepdims=True
        )


def calculate_state_prime(model, initial_state, action):

    state_initial_w_action = np.concatenate((initial_state, [action]))
    x = normalize(torch.tensor(state_initial_w_action, dtype=torch.float32))
    with torch.no_grad():
        state_final = model(x).numpy()

    return state_final


def get_reward(initial_state, final_state):
    """
    See create_example() in cs229_create_fittedvalueiteration_dataset.py for
    how the state vector is constructed from individual features
    """
    return final_state[2] - initial_state[2]


def get_value(state, model, device):
    """
    Here we assume a nonlinear function
    """
    with torch.no_grad():
        x = normalize(torch.tensor(state, dtype=torch.float32).to(device))
        y = model(x).cpu().numpy()

    return y


def update_value_function(
    model, device, train_x, train_y, batch_size=5, max_num_epochs=1000, eps=1e-5
):
    train_data = np.concatenate([train_x, np.expand_dims(train_y, axis=1)], axis=1)
    train_dataset = FVI_VFDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    prev_train_loss = 1e6
    converged = False
    while not converged:
        for epoch in range(max_num_epochs):
            total_loss = 0
            for x, y in train_loader:
                xx = normalize(x).to(device)
                yy = y.to(device)
                yy_pred = model(xx)
                loss = loss_fn(torch.unsqueeze(yy, 1), yy_pred)

                optimizer.zero_grad()  # reset the computed gradients to 0
                loss.backward()  # compute the gradients
                optimizer.step()  # take one step using the computed gradients and optimizer
                total_loss += loss.item()  # track your loss
            train_loss = total_loss / len(train_loader)
            if abs(train_loss - prev_train_loss) <= eps:
                converged = True
                break
            elif epoch == max_num_epochs - 1:
                print(
                    "Warning: value function training did not converge before max epochs hit"
                )
                converged = True
                break
            else:
                # if epoch % 50 == 0:
                #     print(f"    Training loss on epoch {epoch}: {train_loss}")
                prev_train_loss = train_loss
    return


def get_q_value(state_initial, gamma, state_final, model, device):
    return get_reward(state_initial, state_final) + gamma * get_value(
        state_final, model, device
    )


# def update_value_function(theta, x, y, eps=1e-5, step_size=1e-10):
#     """
#     This update rule currently assumes we are solving a linear regression problem.
#     """

#     n_examples, dim = x.shape

#     Y = np.broadcast_to(y, (dim, n_examples))

#     stopping_criterion_not_reached = True
#     while stopping_criterion_not_reached:

#         theta_old = deepcopy(theta)

#         # mat: (dim, n_examples)
#         mat = Y - np.broadcast_to(x @ theta, [dim, n_examples])

#         theta = theta + step_size * np.sum(np.multiply(mat, x.T), axis=1)

#         if np.linalg.norm(theta_old - theta) < eps:
#             stopping_criterion_not_reached = False

#     return theta


def main(
    hidden_dim,
    nn_batch_size,
    nn_num_epochs,
    batch_size,
    batch_size_vf,
    max_iter,
    plot=True,
):
    # Seed the randomness of the simulation so this outputs the same thing each time
    # np.random.seed(0)

    GAMMA = 0.99
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20

    # # Load model for state prediction from (initial_state,action)
    MODEL_PATH = "./fvi/state_predictor_128_3000_128_sgdoptim_normalized.pt"
    # MODEL_PATH = "./fvi/state_predictor.pt"
    model = MultiLayerPerceptron(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # Initialize value function model
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    value_function_model = MultiLayerPerceptronScalarOutput(hidden_dim=hidden_dim).to(
        device
    )

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
    # theta = initialize_mdp_data(STATE_DIM)

    # This is the criterion to end the simulation.
    consecutive_no_learning_trials = 0
    error = []
    first_iteration = True
    num_iter = 0
    while (
        consecutive_no_learning_trials < NO_LEARNING_THRESHOLD and num_iter < max_iter
    ):

        # choose a random sample of states from the buffer
        idx = np.random.randint(num_samples, size=batch_size)
        data_batch = data[idx, :]
        y_batch = np.zeros((batch_size,))

        # data_batch = data
        # batch_size = data_batch.shape[0]
        # y_batch = np.zeros((batch_size,))

        if first_iteration:
            test_convergence_batch = deepcopy(data_batch)
            prior_test_convergence_values = get_value(
                test_convergence_batch, value_function_model, device
            )
            first_iteration = False

        # for each sample in the batch
        for i in range(batch_size):

            state_initial = data_batch[i, :]
            q_value_per_action = np.zeros((len(ALL_POSSIBLE_ACTIONS),))

            for j, action in enumerate(ALL_POSSIBLE_ACTIONS):

                state_final = calculate_state_prime(model, state_initial, action)
                q_value_per_action[j] = get_q_value(
                    state_initial, GAMMA, state_final, value_function_model, device
                )

            y_batch[i] = max(q_value_per_action)

        # update theta for this batch
        update_value_function(
            value_function_model, device, data_batch, y_batch, batch_size=batch_size_vf
        )

        # check for convergence
        test_convergence_values = get_value(
            test_convergence_batch, value_function_model, device
        )
        max_abs_change = np.max(
            np.abs(test_convergence_values - prior_test_convergence_values)
        )
        error.append(max_abs_change)
        if max_abs_change <= TOLERANCE:
            converged_in_one_iteration = True
        else:
            converged_in_one_iteration = False
            prior_test_convergence_values = deepcopy(test_convergence_values)

        print(f"Concluded iteration {num_iter}. Error: {max_abs_change}.")
        # update progress towards training completion if convergence occurs
        if converged_in_one_iteration:
            consecutive_no_learning_trials = consecutive_no_learning_trials + 1
        else:
            consecutive_no_learning_trials = 0

        num_iter += 1

    if plot:
        # plot the learning curve (time balanced vs. trial)
        plt.plot([i for i in range(len(error))], error, "k-")
        plt.xlabel("Num epochs")
        plt.ylabel("Error (max absolute change in theta)")
        plt.savefig(
            f"./fvi_error_{nn_batch_size}_{nn_num_epochs}_{hidden_dim}_sgdoptim_{batch_size}.png"
        )

    return value_function_model


if __name__ == "__main__":
    hidden_dim = 128
    nn_batch_size = 128
    nn_num_epochs = 3000
    batch_size = 750
    batch_size_vf = 5
    max_iter = 20
    model = main(
        hidden_dim=hidden_dim,
        nn_batch_size=nn_batch_size,
        nn_num_epochs=nn_num_epochs,
        batch_size=batch_size,
        batch_size_vf=batch_size_vf,
        max_iter=max_iter,
    )
    output_dir = "./fvi"
    torch.save(
        model.state_dict(),
        f"./fvi/value_function_{batch_size}_{nn_num_epochs}_{hidden_dim}_sgdoptim_normalized_{batch_size}_{batch_size_vf}_{max_iter}.pt",
    )
