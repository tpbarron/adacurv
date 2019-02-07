import os

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

        self.hoopStartPos = [2.0, 0.0, 1.0]
        self.hoopStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        self.cylStartPos = [0.2, 0.0, 0.4+0.2]
        self.cylStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        self.ballStartPos = [0.2, 0.0, 0.4+2*0.2+0.13]
        # self.ballStartPos = [2.0, 0.0, 10.0]
        self.ballStartOrientation = pb.getQuaternionFromEuler([0,0,0])
        self.initial_ball_z = None

        # concat one arm pos, and vel
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,))
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

    def get_state(self):
        robot_state = self.robot.state()
        ball_state = self.get_ball_state()
        state = np.concatenate([robot_state, ball_state])
        return state

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
    #
    #     v0 = np.linalg.norm(ball_lin_vel)
    #     theta = pass
    #
    #     return 1.0

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

        dist_to_hoop = np.linalg.norm(ball_pos[0:2] - hoop_pos[0:2])
        if dist_to_hoop + ball_radius > hoop_radius:
            return False

        if ball_pos[2] > hoop_aabb[1][2]:
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
        ball_lin_vel, ball_ang_vel = pb.getBaseVelocity(self.ball, physicsClientId=self.client)
        if ball_lin_vel[2] < 1e-5:
            # probably sitting on cyl
            return True
        return False

    def complete_trajectory(self):
        while True:
            # print ("complete_trajectory")
            if self.ball_caught():
                return 1.0
            elif self.ball_out_of_play():
                return self.reward_distance_to_hoop()
            elif self.ball_on_cyl():
                return -1000.0
            else:
                for i in range(10):
                    pb.stepSimulation(physicsClientId=self.client)
                    if self.delay:
                        time.sleep(1./240.)

    def get_ball_state(self):
        ball_pos, ball_orient = pb.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        ball_lin_vel, ball_ang_vel = pb.getBaseVelocity(self.ball, physicsClientId=self.client)
        return np.array(ball_pos+ball_lin_vel)

    def step(self, action):
        self.n_step += 1
        self.robot.act(action)

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
            rew = self.reward_distance_to_hoop()

        done = thrown or self.n_step >= self.horizon
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

    while True:
        done = False
        while not done:
            # import time
            # time.sleep(1.0/240.0)
            action = np.random.randn(6)
            obs, rew, done, info = env.step(action)
            # print ("Caught: ", env.ball_caught())
            # print ("Out of play: ", env.ball_out_of_play())
        obs = env.reset()