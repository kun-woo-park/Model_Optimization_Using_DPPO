import numpy as np
import copy
import itertools
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# plot function
def plot(epi_idx, rewards, total_res, test_rewards, test_total_res):
    plt_res = total_res[-1]
    test_plt_res = test_total_res[-1]

    plt.figure(num=0, figsize=(8, 12))
    plt.clf()
    plt.subplot(411)
    plt.title('Episodes %s. reward: %s' % (epi_idx, rewards[-1]))
    plt.plot(rewards, label="rewards")
    plt.plot(test_rewards, label="rewards-test")
    plt.grid(), plt.legend()

    plt.subplot(412)
    plt.plot(plt_res[:, 0], label=r'$\dot{h}_{cmd}$')
    plt.plot(test_plt_res[:, 0], label=r'$\dot{h}_{cmd-test}$')
    plt.ylabel(r'$\dot{h}_{cmd}$ ($m/s$)'), plt.grid(), plt.legend()

    plt.subplot(413)
    plt.plot(plt_res[:, 10], label=r'${h}$')
    plt.plot(test_plt_res[:, 10], label=r'${h-test}$')
    plt.ylabel(r'$h$ (m)'), plt.grid(), plt.legend()

    plt.subplot(414)
    plt.plot(plt_res[:, 1], label=r'${r}$')
    plt.plot(test_plt_res[:, 1], label=r'${r-test}$')
    plt.ylabel(r'$r$ (m)'), plt.grid(), plt.legend()

    plt.legend()
    plt.pause(0.001)


# Replay buffer
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, actor_model, critic_model):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = actor_model

        # critic
        self.value_layer = critic_model

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action = action_probs.max(0)[1]

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


# Deterministic PPO algorithm
class DPPO:
    def __init__(self, actor_model, critic_model, lr, betas, gamma, K_epochs, eps_clip, step_size):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(actor_model, critic_model).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size)
        self.policy_old = ActorCritic(actor_model, critic_model).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


# training loop
def train_agent(env, dppo, test_model, mean, std, memory, max_episodes, max_timesteps, value_tune_epi,
                update_timestep, log_interval, solved_reward, experiment_version):
    # logging variables
    running_reward = 0
    test_running_reward = 0
    avg_length = 0
    time_step = 0

    # initialize lists for print
    rewards = []
    total_res = []
    test_rewards = []
    test_total_res = []

    # train loop
    for i_episode in range(1, max_episodes+1):
        epi_reward = 0
        test_epi_reward = 0

        state = env.reset()
        test_env = copy.deepcopy(env)

        res_list = np.zeros(12)
        test_res_list = np.zeros(12)
        test_done = False

        for t in range(max_timesteps):
            time_step += 1
            state = (state-mean)/std

            # Test with the initial model
            with torch.no_grad():
                if not test_done:
                    if t == 0:
                        test_state = state
                    test_action = test_model(torch.from_numpy(
                        test_state).float().to(device)).max(0)[1].item()
                    test_state, test_reward, test_done, test_info = test_env.step(
                        test_action)
                    test_state = (test_state - mean) / std
                    cmd_list, r_list, elev_list, azim_list, Pm_list, Pt_list, h_list, height_diff_list = test_info[
                        "info"]
                    Pm_list = Pm_list.tolist()
                    Pt_list = Pt_list.tolist()
                    merged_data = itertools.chain([cmd_list], [r_list], [elev_list], [
                                                  azim_list], Pm_list, Pt_list, [h_list], [height_diff_list])
                    merged_data = np.array(list(merged_data))
                    test_res_list = np.vstack([test_res_list, merged_data])
                    test_epi_reward += test_reward
                    if test_done:
                        test_running_reward += test_epi_reward
                        test_res_list = np.delete(test_res_list, 0, 0)
                        test_total_res.append(test_res_list)

            # select action with model to tune the value network
            if i_episode < value_tune_epi:
                action = dppo.policy_old.act(state, memory)
            else:
                if i_episode == value_tune_epi:
                    for param in dppo.policy_old.action_layer.parameters():
                        param.requires_grad = True
                    for param in dppo.policy.action_layer.parameters():
                        param.requires_grad = True
                action = dppo.policy_old.act(state, memory)

            # Running policy_old
            state, reward, done, info = env.step(action)

            # save data to print
            cmd_list, r_list, elev_list, azim_list, Pm_list, Pt_list, h_list, height_diff_list = info[
                "info"]
            Pm_list = Pm_list.tolist()
            Pt_list = Pt_list.tolist()
            merged_data = itertools.chain([cmd_list], [r_list], [elev_list], [
                                          azim_list], Pm_list, Pt_list, [h_list], [height_diff_list])
            merged_data = np.array(list(merged_data))
            res_list = np.vstack([res_list, merged_data])

            # Saving reward and is_terminal
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                dppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += reward
            epi_reward += reward
            if done:
                res_list = np.delete(res_list, 0, 0)
                total_res.append(res_list)
                break

        avg_length += t

        if i_episode >= value_tune_epi:
            dppo.scheduler.step()

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(dppo.policy.state_dict(),
                       './DPPO_{}.pth'.format(experiment_version))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = avg_length / log_interval
            running_reward = running_reward / log_interval
            test_running_reward = test_running_reward / log_interval
            rewards.append(running_reward)
            test_rewards.append(test_running_reward)
            plot(i_episode, rewards, total_res, test_rewards, test_total_res)

            print('Episode {} | avg length: {} | run_reward: {} | min_r: {:.2f} | reward: {} | init_height_diff: {:.2f} \n \t               test_run_reward: {} | test_min_r: {:.2f} | test_reward: {}'              .format(
                i_episode, avg_length, running_reward, min(total_res[-1][:, 1]),                      epi_reward, total_res[-1][0, -1], test_running_reward, min(test_total_res[-1][:, 1]), test_epi_reward), end="\r")
            running_reward = 0
            test_running_reward = 0
            avg_length = 0

    torch.save(dppo.policy.state_dict(), './DPPO_{}.pth'.format(experiment_version))

    return rewards, total_res, test_rewards, test_total_res
