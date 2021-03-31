import gym
import numpy as np
import random
import warnings
from gym import error, spaces, utils
from gym.utils import seeding


class AircraftEnv(gym.Env):

    def __init__(self, succeed_coef = 8000, collide_coef = -2000, change_cmd_penalty = -200, cmd_penalty = -1, start_cond_coef = 90, cmd_suit_coef = 0.01, reset = False):
        super(AircraftEnv).__init__()
        # setting seed
        if not reset:
            self.seed()
        
        # initialize reward coefficients
        
        self.succeed_coef = succeed_coef
        self.collide_coef = collide_coef
        self.change_cmd_penalty= change_cmd_penalty
        self.cmd_penalty = cmd_penalty
        self.start_cond_coef = start_cond_coef
        self.cmd_suit_coef = cmd_suit_coef
        
        # initialize constants and variables
        
        self.Deg2Rad = np.pi/180                # Deg to Rad
        self.g = 9.8                            # Gravity acceleration
        self.K_alt = .8*2                       # hdot loop gain
        self.AoA0 = -1.71*self.Deg2Rad          # zero lift angle of attac
        # 1m/s^2 ACC corresponds to 0.308333deg AOA
        self.Acc2AoA = 0.308333*self.Deg2Rad
        self.zeta_ap = 0.7                      # pitch acceleration loop damping
        self.omega_ap = 4                       # pitch acceleration loop bandwidth
        self.dist_sep = 101                     # near mid-air collision range
        self.dt = 0.1                           # control frequency
        self.tf = 30                            # final time
        self.t = np.arange(0, self.tf, self.dt) # time array
        self.N = len(self.t)                    # length of maximum scenario
        self._state = np.zeros(5)               # initialize states
        self.h_cmd_count = 0
        self.h_cmd_suit=0
        self.t_step = 0

        # mother ship initial conditions

        self.hm0 = 1000
        self.Vm = 200
        self.gamma0 = 0*self.Deg2Rad
        self.Pm_NED = np.array([0, 0, -self.hm0])
        self.Vm_NED = np.array(
            [self.Vm * np.cos(self.gamma0), 0, -self.Vm * np.sin(self.gamma0)])
        self.X0 = np.array([self.g / np.cos(self.gamma0),
                            0, self.hm0, -self.Vm_NED[2], 0])

        # target initial conditions
        self.ht0 = 1000 + 50*(2*(self.np_random.rand())-1)
        self.Vt = 200
        self.approach_angle = 50 * self.Deg2Rad * (2 * self.np_random.rand() - 1)
        self.psi0 = np.pi + self.approach_angle + 2 * self.np_random.randn() * self.Deg2Rad
        self.psi0 = np.arctan2(np.sin(self.psi0), np.cos(self.psi0))
        self.Pt_N = 2000 * (1 + np.cos(self.approach_angle))
        self.Pt_E = 2000 * np.sin(self.approach_angle)
        self.Pt_D = -self.ht0
        # initial NED position
        self.Pt_NED = np.array([self.Pt_N, self.Pt_E, self.Pt_D])
        # initial NED velocity
        self.Vt_NED = np.array(
            [self.Vt * np.cos(self.psi0), self.Vt * np.sin(self.psi0), 0])

        # initialize variables
        self.X = np.zeros((self.N, len(self.X0)))
        self.X[0, :] = self.X0
        self.dotX_p = 0
        self.theta0 = self.gamma0 + \
            self.X0[0] * self.Acc2AoA + self.AoA0        # initial pitch angle
        # initial DCM NED-to-Body
        self.DCM = np.zeros((3, 3))
        self.DCM[0, 0] = np.cos(self.theta0)
        self.DCM[0, 2] = -np.sin(self.theta0)
        self.DCM[1, 1] = 1
        self.DCM[2, 0] = np.sin(self.theta0)
        self.DCM[2, 2] = np.cos(self.theta0)
        self.Pr_NED = self.Pt_NED - self.Pm_NED       # relative NED position
        self.Vr_NED = self.Vt_NED - self.Vm_NED       # relative NED velosity
        # relative position (Body frame)
        self.Pr_Body = np.dot(self.DCM, self.Pr_NED)

        # radar outputs
        self.r = np.linalg.norm(self.Pr_Body)  # range
        self.vc = -np.dot(self.Pr_NED, self.Vr_NED) / \
            self.r  # closing velocity
        # target vertival look angle (down +)
        self.elev = np.arctan2(self.Pr_Body[2], self.Pr_Body[0])
        # target horizontal look angle (right +)
        self.azim = np.arctan2(
            self.Pr_Body[1], self.Pr_Body[0] / np.cos(self.theta0))
        self.los = self.theta0 - self.elev  # line of sight angle
        self.dlos = 0
        self.daz = 0

        # initialize variables
        self.los_p = self.los
        self.dlos_p = self.dlos
        self.azim_p = self.azim
        self.daz_p = self.daz
        self.gamma = self.gamma0
        self.hdot_cmd = 0
        self.hdot = 0
        self.height_diff = self.r * self.los

        # set action space and obs space
        self.high = np.array(
            [5000, 400, np.pi, 2*np.pi, 2*np.pi], dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-self.high, high=self.high, dtype=np.float32)  # r, vc, los, daz, dlos`
        

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def model(self, z, t, hdot_cmd):                # computes state derivatives
        Vm = 200
        # state vector: a (pitch acc), adot, h (alt), hdot, R (ground-track range)
        a, adot, h, hdot, R = z
        gamma = np.arcsin(hdot / Vm)                # fight path angle
        ac = self.K_alt * (hdot_cmd - hdot) + self.g / \
            np.cos(gamma)    # pitch acceleration command
        # maneuver limit
        ac = np.clip(ac, -30, 30)

        addot = self.omega_ap * self.omega_ap * \
            (ac - a) - 2 * self.zeta_ap * self.omega_ap * adot
        hddot = a * np.cos(gamma) - self.g
        Rdot = Vm * np.cos(gamma)
        # returns state derivatives
        return np.array([adot, addot, hdot, hddot, Rdot])

    def step(self, action):
        done = False
        reward = 0

        # set end condition      
        if self.r <= self.dist_sep:
            reward = self.collide_coef
            done = True
        
        elif self.t_step > len(self.t) - 1:
            reward = self.succeed_coef
            done = True
            
        elif self.r >= 5000:
            reward = self.succeed_coef
            done = True
        
        elif self.t_step > 3 and self.r > self.dist_sep and abs(self.elev) > 40*self.Deg2Rad and abs(self.azim) > 40*self.Deg2Rad:
            reward = self.succeed_coef
            done = True

        # make a step and observe next state
        if not done:
            # set an action
            if action == 0:
                if self.hdot_cmd != 0:
                    reward += self.change_cmd_penalty
                    self.h_cmd_count += 1
                self.hdot_cmd = 0
            elif action == 1:
                if self.hdot_cmd != -20:
                    reward += self.change_cmd_penalty
                    self.h_cmd_count += 1
                self.hdot_cmd = -20
            elif action == 2:
                if self.hdot_cmd != 20:
                    reward += self.change_cmd_penalty
                    self.h_cmd_count += 1
                self.hdot_cmd = 20
            else:
                warnings.warn(
                    "The action should be 0 or 1 or 2 but other was detected.")

            self.dotX = self.model(
                self.X[self.t_step, :], self.t[self.t_step], self.hdot_cmd)
            self.X[self.t_step + 1, :] = self.X[self.t_step, :] + \
                0.5 * (3 * self.dotX - self.dotX_p) * self.dt
            self.dotX_p = self.dotX
            self.Pt_NED = self.Pt_NED + self.Vt_NED * self.dt

            self.a, self.adot, self.h, self.hdot, self.R = self.X[self.t_step+1, :]

            self.gamma = np.arcsin(self.hdot/self.Vm)
            self.theta = self.gamma + self.a*self.Acc2AoA + self.AoA0

            self.DCM = np.zeros((3, 3))
            self.DCM[0, 0] = np.cos(self.theta)
            self.DCM[0, 2] = -np.sin(self.theta)
            self.DCM[1, 1] = 1
            self.DCM[2, 0] = np.sin(self.theta)
            self.DCM[2, 2] = np.cos(self.theta)

            self.Pm_NED = np.array([self.R, 0, -self.h])
            self.Vm_NED = np.array(
                [self.Vm*np.cos(self.gamma), 0, -self.Vm*np.sin(self.gamma)])

            self.Pr_NED = self.Pt_NED - self.Pm_NED
            self.Vr_NED = self.Vt_NED - self.Vm_NED
            self.Pr_Body = np.dot(self.DCM, self.Pr_NED)

            self.r = np.linalg.norm(self.Pr_Body)
            self.vc = -np.dot(self.Pr_NED, self.Vr_NED)/self.r
            self.elev = np.arctan2(self.Pr_Body[2], self.Pr_Body[0])
            self.azim = np.arctan2(
                self.Pr_Body[1], self.Pr_Body[0]/np.cos(self.theta))
            psi = np.arctan2(self.Vt_NED[1], self.Vt_NED[0])

            # los rate and azim rate estimation
            self.los = self.theta - self.elev
            # filtered LOS rate, F(s)=20s/(s+20)
            self.dlos = (30*(self.los-self.los_p) + 0*self.dlos_p) / 3
            self.daz = (30*(self.azim-self.azim_p) + 0*self.daz_p) / \
                3  # filtered azim rate, F(s)=20s/(s+20)

            self.los_p = self.los
            self.dlos_p = self.dlos
            self.azim_p = self.azim
            self.daz_p = self.daz
            self._state = np.array(
                [self.r, self.vc, self.los, self.daz, self.dlos])

            self.t_step += 1
            self.h_cmd_suit = self.hdot_cmd * self.height_diff
            
            if self.h_cmd_suit > 0:
                reward += np.abs(self.hdot_cmd) * self.t_step * self.cmd_penalty + self.cmd_suit_coef
            else:
                reward += np.abs(self.hdot_cmd) * self.t_step * self.cmd_penalty

        return self._state.flatten(), reward, done, {"info": [self.hdot_cmd, self.r, self.elev, self.azim, self.Pm_NED, self.Pt_NED, self.h, self.height_diff]}

    def reset(self):
        self.__init__(self.succeed_coef, self.collide_coef, self.change_cmd_penalty, self.cmd_penalty, self.start_cond_coef, self.cmd_suit_coef, True)
        return self._state

    def render(self):
        pass