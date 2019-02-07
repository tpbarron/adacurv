import os
import time
import numpy as np
import pybullet as pb

class BasketballRobot:

    def __init__(self, pb_client_id, delay=True):
        self.pb_client_id = pb_client_id
        self.delay = delay

        self.robot_urdf = os.path.join(os.path.dirname(__file__), 'assets/bbbot_description/urdf/bbbot.urdf')
        self.robot = pb.loadURDF(self.robot_urdf, [0.0, 0.0, 0.0], physicsClientId=self.pb_client_id)

        self.joint_indices_left = [2, 3, 4, 5, 6, 7]
        self.joint_indices_right = [11, 12, 13, 14, 15, 16]


    def state(self):
        joint_states = pb.getJointStates(self.robot, self.joint_indices_left+self.joint_indices_right, physicsClientId=self.pb_client_id)
        # positions
        state = [j[0] for j in joint_states]
        # velocities
        state += [j[1] for j in joint_states]
        state = np.array(state)
        return state

    def move_to_initial_position(self):
        pos_left = [2.5, 0.75, -0.4, 0, 0, 3.14]
        pos_right = [-2.5, 0.75, -0.4, 0, 0, 3.14]
        pb.setJointMotorControlArray(self.robot, self.joint_indices_left, pb.POSITION_CONTROL, targetPositions=pos_left, physicsClientId=self.pb_client_id)
        pb.setJointMotorControlArray(self.robot, self.joint_indices_right, pb.POSITION_CONTROL, targetPositions=pos_right, physicsClientId=self.pb_client_id)

        for s in range(100):
            pb.stepSimulation(physicsClientId=self.pb_client_id)
            if self.delay:
                time.sleep(1./240.)

    def move_to_pickup_position(self):
        pickup_pos_left = [2.9390760359372385, 1.217528331414226, -0.17592668464016736, 0, 1.2410842115944702, 3.14]
        pickup_pos_right = [-2.9390760359372385, 1.217528331414226, -0.17592668464016736, 0, 1.2410842115944702, 3.14]
        pb.setJointMotorControlArray(self.robot, self.joint_indices_left, pb.POSITION_CONTROL, targetPositions=pickup_pos_left, physicsClientId=self.pb_client_id)
        pb.setJointMotorControlArray(self.robot, self.joint_indices_right, pb.POSITION_CONTROL, targetPositions=pickup_pos_right, physicsClientId=self.pb_client_id)

        for s in range(100):
            pb.stepSimulation(physicsClientId=self.pb_client_id)
            if self.delay:
                time.sleep(1./240.)

    def zero_control(self):
        z = np.zeros((6,))
        pb.setJointMotorControlArray(self.robot, self.joint_indices_left, pb.VELOCITY_CONTROL, targetVelocities=z, physicsClientId=self.pb_client_id)
        pb.setJointMotorControlArray(self.robot, self.joint_indices_right, pb.VELOCITY_CONTROL, targetVelocities=z, physicsClientId=self.pb_client_id)

    def act(self, u, mode=pb.VELOCITY_CONTROL, action_repeat=10):
        """
        u should be a vector of inputs for the left arm, will be mirrors on the right
        """

        if mode == pb.POSITION_CONTROL:
            u *= 0.05
            u_left = u.copy()
            u_right = u.copy()
            u_right[0] *= -1

            # When using POSITION_CONTROL u is the change in joints
            joint_states_left = pb.getJointStates(self.robot, self.joint_indices_left, physicsClientId=self.pb_client_id)
            joint_pos_left = np.array([j[0] for j in joint_states_left])
            target_left = joint_pos_left + u_left

            joint_pos_right = joint_pos_left.copy()
            joint_pos_right[0] *= -1
            target_right = joint_pos_right + u_right

            pb.setJointMotorControlArray(self.robot, self.joint_indices_left, mode, targetPositions=target_left, physicsClientId=self.pb_client_id)
            pb.setJointMotorControlArray(self.robot, self.joint_indices_right, mode, targetPositions=target_right, physicsClientId=self.pb_client_id)

        elif mode == pb.VELOCITY_CONTROL:
            u_left = u[0:6] #.copy()
            u_right = u[6:12]
            # u_right[0] *= -1

            # u_left = u.copy()
            # u_right = u.copy()
            # u_right[0] *= -1
            forces = [1000.0] * 6

            pb.setJointMotorControlArray(self.robot, self.joint_indices_left, mode, targetVelocities=u_left, forces=forces, physicsClientId=self.pb_client_id)
            pb.setJointMotorControlArray(self.robot, self.joint_indices_right, mode, targetVelocities=u_right, forces=forces, physicsClientId=self.pb_client_id)
        elif mode == pb.TORQUE_CONTROL:
            u *= 100.0
            u_left = u.copy()
            u_right = u.copy()
            u_right[0] *= -1

            pb.setJointMotorControlArray(self.robot, self.joint_indices_left, mode, forces=u_left, physicsClientId=self.pb_client_id)
            pb.setJointMotorControlArray(self.robot, self.joint_indices_right, mode, forces=u_right, physicsClientId=self.pb_client_id)


        for s in range(action_repeat):
            pb.stepSimulation(physicsClientId=self.pb_client_id)
            if self.delay:
                time.sleep(1./240.)
