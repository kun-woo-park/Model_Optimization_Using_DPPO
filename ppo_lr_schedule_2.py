import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from IPython.display import clear_output

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# plot function for jupyter notebook
def plot(epi_idx, rewards, total_res, test_rewards, test_total_res):
    plt_res = total_res[-1]
    test_plt_res = test_total_res[-1]

    clear_output(True)
    plt.figure(figsize=(20, 15))
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
    plt.show()


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


class FClayer(nn.Module):  # define fully connected layer with Leaky ReLU activation function
    def __init__(self, innodes, nodes):
        super(FClayer, self).__init__()
        self.fc = nn.Linear(innodes, nodes)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        out = self.act(out)
        return out


# define custom model named wave net, which was coined after seeing the nodes sway
class WaveNET(nn.Module):
    def __init__(self, block, planes, nodes, num_classes=3):
        super(WaveNET, self).__init__()
        self.innodes = 5

        self.layer1 = self._make_layer(block, planes[0], nodes[0])
        self.layer2 = self._make_layer(block, planes[1], nodes[1])
        self.layer3 = self._make_layer(block, planes[2], nodes[2])

        self.fin_fc = nn.Linear(self.innodes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, nodes):

        layers = []
        layers.append(block(self.innodes, nodes))
        self.innodes = nodes
        for _ in range(1, planes):
            layers.append(block(self.innodes, nodes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fin_fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


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

    def act(self, state, memory, e_greedy):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        if np.random.rand() >= e_greedy:
            action = action_probs.max(0)[1]
        else:
            action = dist.sample()

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
class PPO:
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
