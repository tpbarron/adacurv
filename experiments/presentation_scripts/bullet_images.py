import pybullet as pb

from pybullet_envs.gym_locomotion_envs import HopperBulletEnv, HalfCheetahBulletEnv, Walker2DBulletEnv
from PIL import Image

import numpy as np


# env = HopperBulletEnv(render=True)
# env = HalfCheetahBulletEnv(render=True)
env = Walker2DBulletEnv(render=True)

while True:
    obs = env.reset()
    done = False
    # pb.setGravity(0,0,0)

    i = 0
    while not done:


        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)

        import time
        time.sleep(0.1)



        # viewMat = pb.computeViewMatrix([0, -10, 10], [0, 1, 0], [0, 0, 1])
        # width, height, rgb_px, depth_px, segmask = pb.getCameraImage(1024, 1024, viewMatrix=viewMat)
        #
        # result = Image.fromarray(rgb_px)#.astype(np.uint8))
        # result.save('out.png')
        # input("")
