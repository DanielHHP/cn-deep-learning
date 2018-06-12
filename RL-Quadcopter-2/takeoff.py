import numpy as np
from physics_sim import PhysicsSim

class Takeoff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 1000
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
    
    def reward_calc_v1(self):
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        return reward

    def reward_calc_v2(self):
        reward = 1.
        # position penalty
        reward -= 0.3 * abs(self.sim.pose[:3] - self.target_pos).sum()
        # height penalty
        reward -= 0.6 * abs(self.sim.pose[2] - self.target_pos[2])
        # eular penalty
        reward -= 3 * abs(self.sim.pose[3:6]).sum()
        # eular speed penalty
        reward -= 3 * abs(self.sim.angular_v).sum()
        # z speed reward
        reward += 3 * (self.sim.v[2])
        # speed penalty
        reward -= 3 * abs(self.sim.v[:2]).sum()
        # crash penalty
        if (self.sim.pose[2] <= 0.):
            reward -= 50
        # bonus reward
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 100
        
        return reward
    
    def reward_calc_v3(self):
        reward = 1.
        # position penalty
        reward -= 0.3 * abs(self.sim.pose[:3] - self.target_pos).sum()
        # height penalty
        reward -= 0.6 * abs(self.sim.pose[2] - self.target_pos[2])
        # crash penalty
        if (self.sim.pose[2] <= 0.):
            reward -= 50
        # bonus reward
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 200
        
        return reward

    def reward_calc_v4(self):
        reward = 0.
        # height
        reward += self.sim.pose[2]

        # pos
        reward -= .5 * abs(self.sim.pose[:1]).sum()

        # v
        reward += .5 * self.sim.v[2]

        # angular v
        reward -= .25 * (abs(self.sim.angular_v[:3])).sum()

        return reward

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = self.reward_calc_v4()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        if self.sim.pose[2] >= self.target_pos[2]:
            done = True
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state