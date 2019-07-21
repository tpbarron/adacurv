import numpy as np
import zmq
import pickle

def send_init(socket, env_name):
    md = dict(
        type='init',
        env_name=env_name
    )
    return socket.send_json(md)

def send_action(socket, uuid, action, flags=0, copy=True, track=False):
    md = dict(
        type='action',
        uuid=uuid
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    action_serialized = pickle.dumps(action)
    return socket.send(action_serialized, flags, copy=copy, track=track)

def send_reset(socket, uuid, flags=0, copy=True, track=False):
    md = dict(
        type='reset',
        uuid=uuid
    )
    return socket.send_json(md, flags)

def send_close(socket, uuid, flags=0, copy=True, track=False):
    md = dict(
        type='close',
        uuid=uuid
    )
    return socket.send_json(md, flags)
