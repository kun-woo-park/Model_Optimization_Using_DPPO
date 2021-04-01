import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

# creating environment
succeed_coef = 8000         # maximum reward when agent avoids collision
collide_coef = -4000        # reward when agent doesn't avoid collision
change_cmd_penalty = -100   # reward when agent changes command values
cmd_penalty = -0.15          # coefficient of penaly on using command
cmd_suit_coef = -100         # coefficient of suitable command
start_cond_coef = 100       # coefficient of condition on begining
step_size = 10000          # lr scheduling step size

lr = 0.0001
betas = (0.9, 0.999)

env_name = "acav-v0"
env = gym.make(env_name)
env.env.__init__(succeed_coef, collide_coef, change_cmd_penalty,
                 cmd_penalty, start_cond_coef, cmd_suit_coef)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
random_seed = 0


policy_net = torch.load("./Custom_model_fin").to(device)
target_net = torch.load("./Custom_model_fin").to(device)
target_net.eval()
mean = np.load('mean_test.npy')
std = np.load('std_test.npy')


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


policy_net.apply(init_weights)
target_net.apply(init_weights)


n_actions = env.action_space.n


optimizer = optim.Adam(policy_net.parameters(), lr=lr, betas=betas)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
memory = ReplayMemory(20000)


steps_done = 0



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        #         return policy_net(state).max(1)[1].view(1, 1)
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


torch.manual_seed(random_seed)
env.seed(random_seed)


num_episodes = 50000
total_res = []
reward_list = []
for i_episode in range(num_episodes):
    total_reward = 0

    # 환경과 상태 초기화
    res_list = np.zeros(11)
    state = env.reset()
    state = (state-mean)/std
    state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    for t in count():
        # 행동 선택과 수행

        action = select_action(state)
        next_state, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], dtype=torch.float32).to(device)

        next_state = (next_state-mean)/std
        next_state = torch.from_numpy(
            next_state.astype(np.float32)).unsqueeze(0).to(device)

        # 새로운 상태 관찰
        if not done:
            next_state = next_state
        else:
            next_state = None

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # 최적화 한단계 수행(목표 네트워크에서)
        optimize_model()

        # Data save

        cmd_list, r_list, elev_list, azim_list, Pm_list, Pt_list, h_list, height_diff_list = info[
            "info"]
        Pm_list = Pm_list.tolist()
        Pt_list = Pt_list.tolist()
        merged_data = itertools.chain([cmd_list], [r_list], [elev_list], [
                                      azim_list], Pm_list, Pt_list, [h_list])
        merged_data = np.array(list(merged_data))
        res_list = np.vstack([res_list, merged_data])

        total_reward += reward

        if done:
            res_list = np.delete(res_list, 0, 0)

            total_res.append(res_list)
            reward_list.append(total_reward)

            now = time.localtime()
            print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year,
                  now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
            print("episode : {} | final step : {} | total reward : {}".format(
                i_episode, t, total_reward.item()))
            break

    # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    scheduler.step()

print('Complete')
env.close()


def step_average(data, n):
    res = [np.mean(data[i*n:(i+1)*n]) for i in range(int((len(data)/n)))]
    return np.array(res)


for i in range(len(reward_list)):
    reward_list[i] = reward_list[i].cpu().detach().numpy()


average_number = 50
filtered_data = step_average(reward_list, average_number)
plt.figure(figsize=(15, 10))
plt.xlabel("Episode")
plt.ylabel("Total rewards")
plt.plot(filtered_data)


np.savetxt("DQN_non_model_reward.csv", np.array(filtered_data), delimiter=",")
