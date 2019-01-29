
import pickle
import sys
import time
import zmq

import numpy as np
import gym

class SocketEnv(gym.Env):
    def __init__(self, env_name, port=33333):
        super(SocketEnv, self).__init__()

        self.env_name = env_name
        self.port = port
        self.env_uuid = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        #self.socket.connect("tcp://localhost:%s" % port)
        self.socket.connect("tcp://c101943:%s" % port)
        print ("Connected!")
        self._send_init()

    def _send_init(self):
        md = dict(type='init', env_name=self.env_name)
        self.socket.send_json(md)
        init_resp_serialized = self.socket.recv()
        init_resp = pickle.loads(init_resp_serialized)
        uid, env_id, horizon, action_dim, observation_dim = init_resp

        print ("Recv'd init resp: ", init_resp)
        self.env_uuid = uid
        self._env_id = env_id
        self._horizon = horizon
        self._action_dim = action_dim
        self._observation_dim = observation_dim

    def _send_action(self, action, flags=0, copy=True, track=False):
        md = dict(type='action', uuid=self.env_uuid)
        self.socket.send_json(md, flags|zmq.SNDMORE)
        action_serialized = pickle.dumps(action)
        return self.socket.send(action_serialized, flags, copy=copy, track=track)

    def _send_reset(self, flags=0, copy=True, track=False):
        md = dict(type='reset', uuid=self.env_uuid)
        return self.socket.send_json(md, flags)

    def _send_close(self, flags=0, copy=True, track=False):
        md = dict(type='close', uuid=self.env_uuid)
        return self.socket.send_json(md, flags)
    

    def step(self, action):
        assert self.env_uuid is not None, "Must initialize connection first"
        self._send_action(action)
        action_resp_serialized = self.socket.recv()
        action_resp = pickle.loads(action_resp_serialized)
        observation, reward, done, info = action_resp
        return observation, reward, done, info

    def reset(self):
        self._send_reset()
        obs_resp_serialized = self.socket.recv()
        obs = pickle.loads(obs_resp_serialized)
        return obs

    def close(self):
        self._send_close()
        self.socket.close()
        
    def seed(self, u):
        pass

if __name__ == "__main__":
    env_name = sys.argv[1]
    env = SocketEnv(env_name)
    obs = env.reset()
    done = False
    env_tmp = gym.make(env_name)

    for i in range(10):
        while not done:
            # import time
            # time.sleep(0.5)
            act = env_tmp.action_space.sample()
            obs, rew, done, info = env.step(act)
            print (rew)
        env.reset()
        done = False
        print ("resetting env ", env_name)

    env.close()
