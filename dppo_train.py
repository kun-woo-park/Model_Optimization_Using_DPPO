import numpy as np
import torch
import torch.nn as nn
import gym
import gym_Aircraft
from dppo import DPPO, Memory, train_agent
from custom_model import FClayer, WaveNET, init_weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    ############## Hyperparameters ##############
    succeed_coef = 8000         # maximum reward when agent avoids collision
    collide_coef = -4000        # reward when agent doesn't avoid collision
    change_cmd_penalty = -100   # reward when agent changes command values
    cmd_penalty = -0.15         # coefficient of penaly on using command
    cmd_suit_coef = -100        # coefficient of suitable command
    start_cond_coef = 100       # coefficient of condition on begining

    value_tune_epi = 5000       # number of episodes for value net fine tuning

    solved_reward = 7000        # stop training if avg_reward > solved_reward
    log_interval = 50           # print avg reward in the interval
    max_episodes = 50000 + value_tune_epi     # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.0001                 # learning rate
    betas = (0.9, 0.999)        # betas for adam optimizer
    gamma = 0.999               # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    step_size = 10000           # lr scheduling step size
    random_seed = 0             # random seed
    experiment_version = "val_net"  # experiment version
    weights_initialize = False  # choose weight initialize or not
    ##############################################

    # load learned model
    actor_model = torch.load("./Custom_model_fin")
    critic_model = torch.load("./Custom_model_fin")
    test_model = torch.load("./Custom_model_fin")
    # initialize weights if it should
    if weights_initialize:
        actor_model.apply(init_weights)
        critic_model.apply(init_weights)
    # load mean and std of trained data
    mean = np.load('mean_test.npy')
    std = np.load('std_test.npy')
    # set final nodes for each model(final node of critic to one and add softmax to actor)
    num_final_nodes = critic_model.fin_fc.in_features
    critic_model.fin_fc = nn.Linear(num_final_nodes, 1)
    actor_model.fin_fc = nn.Sequential(actor_model.fin_fc, nn.Softmax(dim=-1))
    test_model.fin_fc = nn.Sequential(test_model.fin_fc, nn.Softmax(dim=-1))
    test_model = test_model.to(device)

    # set requires grad to false for fixing until several episodes
    for param in actor_model.parameters():
        param.requires_grad = False

    for param in test_model.parameters():
        param.requires_grad = False

    # creating environment
    env_name = "acav-v0"
    env = gym.make(env_name)
    env.env.__init__(succeed_coef, collide_coef, change_cmd_penalty,
                     cmd_penalty, start_cond_coef, cmd_suit_coef)
    # set random seed of environment
    torch.manual_seed(random_seed)
    env.seed(random_seed)

    # set replay buffer and dppo model
    memory = Memory()
    dppo = DPPO(actor_model, critic_model, lr, betas,
              gamma, K_epochs, eps_clip, step_size)
    # train agent
    rewards, total_res, test_rewards, test_total_res = train_agent(env, dppo, test_model, mean, std, memory, max_episodes, max_timesteps, value_tune_epi,
                update_timestep, log_interval, solved_reward, experiment_version)
    # save reward
    np.savetxt("{}.csv".format(experiment_version),
               np.array(rewards), delimiter=",")





