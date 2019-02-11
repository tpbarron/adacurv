import os
import time

import gym
from gym import error, spaces

import numpy as np
import pybullet as pb
import pybullet_data
from bbbot import basketball_robot



class BasketballEnv(gym.Env):

    def __init__(self, render=False, delay=False, horizon=500):
        if render:
            mode = pb.GUI
        else:
            mode = pb.DIRECT
        self.delay = delay
        self.horizon = horizon

        self.client = pb.connect(mode)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        pb.setGravity(0,0,-10, physicsClientId=self.client)
        self.plane = pb.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot = basketball_robot.BasketballRobot(self.client, delay=delay)

        self.n_step = 0
        self.hoop = None
        self.cyl = None
        self.ball = None

        self.hoopStartPos = [1.5, 0.0, 1.0]
        self.hoopStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        self.cylStartPos = [0.2, 0.0, 0.4+0.2]
        self.cylStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        self.ballStartPos = [0.2, 0.0, 0.4+2*0.2+0.13]
        # self.ballStartPos = [2.0, 0.0, 10.0]
        self.ballStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        self.initial_ball_z = None

        # concat one arm pos, and vel
        # 12 + 12 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,))
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,))

    def get_state(self):
        robot_state = self.robot.state()
        ball_state = self.get_ball_state()
        state = np.concatenate([robot_state, ball_state])
        return state

    def close(self):
        pb.disconnect(physicsClientId=self.client)

    def reset(self):
        self.n_step = 0
        self.robot.move_to_initial_position()
        self._load_scene()
        self.robot.move_to_pickup_position()

        ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        self.initial_ball_z = ball_pos[2]

        return self.get_state()

    def _load_scene(self):
        if self.hoop is None:
            self.hoop = pb.loadSDF(os.path.join(os.path.dirname(__file__), 'assets/bbbot_gazebo/models/hoop/model.sdf'), physicsClientId=self.client)[0]
        pb.resetBasePositionAndOrientation(self.hoop, self.hoopStartPos, self.hoopStartOrientation, physicsClientId=self.client)
        if self.cyl is None:
            self.cyl = pb.loadSDF(os.path.join(os.path.dirname(__file__), 'assets/bbbot_gazebo/models/cylinder/model.sdf'), physicsClientId=self.client)[0]
        pb.resetBasePositionAndOrientation(self.cyl, self.cylStartPos, self.cylStartOrientation, physicsClientId=self.client)
        if self.ball is None:
            self.ball = pb.loadSDF(os.path.join(os.path.dirname(__file__), 'assets/bbbot_gazebo/models/ball/model.sdf'), physicsClientId=self.client)[0]
        pb.resetBasePositionAndOrientation(self.ball, self.ballStartPos, self.ballStartOrientation, physicsClientId=self.client)

    # def reward_predicted_ball_trajectory(self):
    #     ball_lin_vel, ball_ang_vel = pb.getBaseVelocity(self.ball)
    #     ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball)
    #     hoop_pos, hoop_orient = pb.getBasePositionAndOrientation(self.hoop, physicsClientId=self.client)
    #
    #     ball_lin_vel = np.array(ball_lin_vel)
    #     v0 = np.linalg.norm(ball_lin_vel)
    #
    #     print ("v0:", v0)
    #
    #     ball_vec = ball_lin_vel.copy()
    #     ball_vec_proj = ball_vec.copy()
    #     ball_vec_proj[2] = 0.0
    #
    #     print ("Vecs: ", ball_vec, ball_vec_proj)
    #     theta = np.arccos(np.dot(ball_vec, ball_vec_proj) / (np.linalg.norm(ball_vec) * np.linalg.norm(ball_vec_proj)))
    #     print ("Theta: ", theta)
    #
    #     z_h = hoop_pos[2] + 0.1 - ball_pos[2]
    #     print ("Zh: ", z_h)
    #
    #     a = np.tan(theta)
    #     b = 10.0 / (v0**2.0 * np.cos(theta)**2.0)
    #
    #     print ("a,b: ", a, b)
    #     c1 = 0.5 * b ** 2.0
    #     c2 = 1.5 * a * b
    #     c3 = 1. + a**2.0 - b * z_h
    #
    #     print ("c1, c2, c3: ", c1, c2, c3)
    #     sol1 = (-c2 + np.sqrt(c2**2.0 - 4.0 * c1 * c3)) / (2.0 * c1)
    #     print ("Sol: ", sol1)
    #     sol2 = (-c2 - np.sqrt(c2**2.0 - 4.0 * c1 * c3)) / (2.0 * c1)
    #     print ("Sol: ", sol2)
    #     input("")
    #     return sol1

    def reward_idealized_ball_velocity(self):
        idealized_lin_vel = self.get_idealized_ball_velocity()
        if idealized_lin_vel is None:
            print ("------------------------------------")
            print ("WARNING: no feasible ball trajectory")
            print ("------------------------------------")
            return -1000.0

        ball_lin_vel, ball_ang_vel = pb.getBaseVelocity(self.ball, physicsClientId=self.client)
        actual_lin_vel = np.array(ball_lin_vel)
        cost = np.linalg.norm(idealized_lin_vel - actual_lin_vel)
        # print ("Ball velocity cost: ", cost)
        return -cost
        # return np.exp(-0.01*cost)

    def get_idealized_ball_velocity(self):
        theta = 55.0 * np.pi / 180.0 #np.pi / 4.0
        ball_lin_vel, ball_ang_vel = pb.getBaseVelocity(self.ball, physicsClientId=self.client)
        ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        hoop_pos, hoop_orient = pb.getBasePositionAndOrientation(self.hoop, physicsClientId=self.client)

        x = hoop_pos[0] - ball_pos[0]
        z = (hoop_pos[2] + 0.1) - ball_pos[2]

        g = 10.0
        term = (x**2.0 * g) / (x * np.sin(2.0 * theta) - 2.0 * z * np.cos(theta)**2.0)
        if term <= 0.0:
            return None
        v0 = np.sqrt( term )
        v0_x = np.cos(theta) * v0
        v0_z = np.sin(theta) * v0

        idealized_lin_vel = np.array([v0_x, 0.0, v0_z])
        return idealized_lin_vel

    def reward_distance_to_hoop(self):
        ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        hoop_pos, hoop_orient = pb.getBasePositionAndOrientation(self.hoop, physicsClientId=self.client)

        # print (ball_pos, hoop_pos)
        ball_pos = np.array(ball_pos)
        hoop_pos = np.array(hoop_pos)
        dist = np.linalg.norm(hoop_pos - ball_pos)
        return -dist

    def reward_ball_height(self):
        ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        rew = ball_pos[2] - self.initial_ball_z
        return rew

    def ball_has_left_gripper(self):
        left_closest = pb.getClosestPoints(self.robot.robot, self.ball, 0.05, 8, physicsClientId=self.client)
        # print ("LClosest: ", len(left_closest))
        right_closest = pb.getClosestPoints(self.robot.robot, self.ball, 0.05, 17, physicsClientId=self.client)
        # print ("RClosest: ", len(right_closest))
        min_dist = np.inf
        for lc in left_closest:
            if lc[8] < min_dist:
                min_dist = lc[8]
        for lc in right_closest:
            if lc[8] < min_dist:
                min_dist = lc[8]
        return min_dist > 0.05

    def ball_caught(self):
        hoop_aabb = pb.getAABB(self.hoop, physicsClientId=self.client)
        hoop_pos, hoop_orient = pb.getBasePositionAndOrientation(self.hoop, physicsClientId=self.client)
        ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        ball_radius = 0.13/2.0
        hoop_radius = 0.5
        hoop_height = 0.36

        ball_pos = np.array(ball_pos)
        hoop_pos = np.array(hoop_pos)

        # print ("Ball pos: ", ball_pos)
        # print ("Hoop pos: ", hoop_pos)
        dist_to_hoop = np.linalg.norm(ball_pos[0:2] - hoop_pos[0:2])
        # print ("Dist to hoop in x-y plane: ", dist_to_hoop, ball_radius, hoop_radius)
        if dist_to_hoop + ball_radius > hoop_radius:
            return False

        if ball_pos[2] < hoop_aabb[1][2]:
            return False

        return True
        # Hoop aabb ((1.7469999999999999, -0.253, 0.7270352281135979), (2.253, 0.253, 1.0929647647338447)) (2.0, 0.0, 1.0)
        # print ("Hoop aabb", hoop_aabb, hoop_pos)

    def ball_out_of_play(self):
        # if ball z vel is negative and ball below hoop, then consider to be finished
        ball_lin_vel, ball_ang_vel = pb.getBaseVelocity(self.ball, physicsClientId=self.client)
        # print ("ball_lin_vel: ", ball_lin_vel)
        if ball_lin_vel[2] < 0:
            ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
            hoop_pos, hoop_orient = pb.getBasePositionAndOrientation(self.hoop, physicsClientId=self.client)
            # print ("ball and hoop:", ball_pos[2], hoop_pos[2])
            if ball_pos[2] < hoop_pos[2]-0.1:
                return True
        return False

    def ball_on_cyl(self):
        ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        ball_lin_vel, ball_ang_vel = pb.getBaseVelocity(self.ball, physicsClientId=self.client)
        if np.abs(ball_pos[2] - self.initial_ball_z) < 0.01:
            return True
        # if ball_lin_vel[2] < 1e-5:
        #     # probably sitting on cyl
        #     return True
        return False

    def complete_trajectory(self):
        for i in range(250):
            pb.stepSimulation(physicsClientId=self.client)
            if self.delay:
                time.sleep(1./240.)
        if self.ball_caught():
            print ("-----------")
            print ("Ball caught")
            print ("-----------")
            return 1000.0
        elif self.ball_on_cyl():
            # print ("-----------")
            print ("Ball on cyl")
            # print ("-----------")
            return -1000.0
        return -1.0 #self.reward_idealized_ball_velocity()

        # while True:
        #     # print ("complete_trajectory")
        #     if self.ball_caught():
        #         return 1000.0
        #     elif self.ball_out_of_play():
        #         return -1000.0
        #         # return self.reward_distance_to_hoop()
        #     elif self.ball_on_cyl():
        #         return -1000.0
        #     else:
        #         for i in range(10):
        #             pb.stepSimulation(physicsClientId=self.client)
        #             if self.delay:
        #                 time.sleep(1./240.)

    def get_ball_state(self):
        ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        ball_lin_vel, ball_ang_vel = pb.getBaseVelocity(self.ball, physicsClientId=self.client)
        return np.array(ball_pos+ball_lin_vel)

    def step(self, action):
        self.n_step += 1
        self.robot.act(action)

        done = False
        state = self.get_state()
        # print (state.shape)
        # input("")
        thrown = self.ball_has_left_gripper()
        if thrown:
            # print ("Ball left gripper" )
            self.robot.zero_control()
            rew = self.complete_trajectory()
        else:
            # rew = self.reward_predicted_ball_trajectory()
            # rew = self.reward_distance_to_hoop()
            # rew = self.reward_predicted_ball_trajectory()
            rew = self.reward_idealized_ball_velocity()
            # if rew == -1000:
            #     done = True

        done = done or thrown or self.n_step >= self.horizon
        return state, rew, done, {}

    # def step(self, action):
    #     self.n_step += 1
    #     self.robot.act(action)
    #
    #     state = self.robot.state()
    #     rew = self.reward_ball_height()
    #     done = self.ball_has_left_gripper() or self.n_step >= self.horizon
    #     return state, rew, done, {}


def make_basketball_env_rendered():
    env = BasketballEnv(render=True, delay=True)
    return env

BasketballEnvRendered = make_basketball_env_rendered


import numpy as np
if __name__ == "__main__":
    env = BasketballEnv(render=True)
    obs = env.reset()
    print (obs, env)

    # env.robot.move_to_initial_position()
    # for i in range(100):
    #     pb.stepSimulation()
    # ideal_vel = env.get_idealized_ball_velocity()
    # pb.resetBaseVelocity(env.ball, linearVelocity=ideal_vel)
    # i = 0
    # while True:
    #     import time
    #     time.sleep(0.1) #1.0/240.0)
    #     pb.stepSimulation()
    #     i += 1
    #     print (i)

    while True:
        done = False
        while not done:
            import time
            time.sleep(1.0/240.0)
            action = np.random.randn(6) * 10.0
            obs, rew, done, info = env.step(action)
            # print ("Caught: ", env.ball_caught())
            # print ("Out of play: ", env.ball_out_of_play())
        obs = env.reset()
        print ("REsetting")
