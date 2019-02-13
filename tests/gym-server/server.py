import pickle
import time
import uuid
import zmq

import numpy as np
import gym

import signal

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    def exit_gracefully(self,signum, frame):
        self.kill_now = True

class EnvServer:

    #def __init__(self, port=33333):
    def __init__(self, port=6001):
        self.port = port
        self.context = zmq.Context()
        self.socket = None
        self._setup()
        self.client_env_map = {}

    def _setup(self):
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % self.port)

    def listen(self):
        print ("Listening")
        #killer = GracefulKiller()
        while True:
            #if killer.kill_now:
            #    self.socket.close()
            #    break
            message_data = self.socket.recv_json()
            message_type = message_data['type']
            print ("Recvd message of type ", message_type)
            if message_type == 'init':
                uid = str(uuid.uuid4())
                env = gym.make(message_data['env_name'])
                self.client_env_map[uid] = env

                env_id = env.spec.id
                horizon = env.spec.timestep_limit
                try:
                    action_dim = env.env.action_dim
                except AttributeError:
                    action_dim = env.env.action_space.shape[0]
                observation_dim = env.env.observation_space.shape[0]
                serialized = pickle.dumps([uid, env_id, horizon, action_dim, observation_dim])
                print ("Initialized")
                self.socket.send(serialized, 0, copy=True, track=False)
                # self.socket.send_string(uid)
            elif message_type == 'action':
                msg = self.socket.recv(flags=0, copy=True, track=False)
                action = pickle.loads(msg)
                obs, rew, done, info = self.client_env_map[message_data['uuid']].step(action)
                serialized = pickle.dumps([obs, rew, done, info], protocol=0)
                self.socket.send(serialized, 0, copy=True, track=False)
            elif message_type == 'reset':
                obs = self.client_env_map[message_data['uuid']].reset()
                obs_serialized = pickle.dumps(obs)
                self.socket.send(obs_serialized, 0, copy=True, track=False)
            elif message_type == 'close':
                val = self.client_env_map.pop(message_data['uuid'], None)
                if val is not None:
                    self.socket.send_string("True")
                else:
                    self.socket.send_string("False")
            else:
                print ("message type unknown")


if __name__ == "__main__":
    e = EnvServer()
    e.listen()
