import cv2
import rclpy
import time
import gym_gazebo
import gymnasium as gym
import numpy as np
from cv_bridge import CvBridge
from gym_gazebo.core.gazebo_env import GazeboEnv
from rclpy.node import Node as RosNode
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist as RosTwist

NUM_OBS_STATES = 3
NUM_ACTIONS = 3

class LineFollowerEnv(GazeboEnv):
    def __init__(self):
        # Intializing ROS2
        rclpy.init(args=None)
        
        super(LineFollowerEnv, self).__init__(launch_pkg='linefollow_ros',
                                    launch_file='my_agent.launch.xml',
                                    world_name='monza_gym')

        # Setting up ROS2 node
        self.ros_node = RosNode('line_follower_RL')

        # Initializing variables
        self.is_resetting = False
        self.latest_obs = None
        self.new_obs_event = False

        # Setting the observation and action spaces
        self.observation_space = gym.spaces.Discrete(NUM_OBS_STATES)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # ROS2 topics
        observation_topic = "/camera/image_raw"
        velocity_topic = "/cmd_vel"

        # Publishing and subscribing to ROS2 topics
        self.pub_cmd_vel_msg = self.ros_node.create_publisher(
            RosTwist, velocity_topic, 10)
            
        self.sub_obs = self.ros_node.create_subscription(
            RosImage, observation_topic, self.callback_obs_feed_, 10)


    def callback_obs_feed_(self, msg: RosImage):
        '''
        @brief observation callback resets the new_obs_event flag
        @param msg: Image from Gazebo sim
        '''
        # Takes in observation, returns a flag an observation is recieved
        if not self.is_resetting:
            self.latest_obs = msg
            self.new_obs_event = True


    ## MAIN FUNCTIONS:
    def reset(self, seed=None, options=None):
        '''
        @brief resets the simulation
        '''
        self.is_resetting = True
        print("Resetting")

        # Zero out velocities
        vel_cmd = RosTwist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Reset position and joints
        self._reset_agents()

        # Unpause the sim
        self._pause_sim(False)

        # Allow some time to settle
        time.sleep(0.1) # Wait 100 ms
        
        self.is_resetting = False

        # Wait for next observation to come in before kickstarting the data loop
        self.new_obs_event = False
        while not self.new_obs_event:
            # Telling executor (node) to spin
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # # Pause the sim after the first observation comes in
        self._pause_sim(True)
        
        state, _ = self.process_obs(self.latest_obs)

        print("Done resetting")
        return state, {}
    
    
    def step(self, action: int):
        '''
        @brief takes a step in the simulation
        @param action
        @retval state
        @retval reward
        @retval terminated
        @retval truncated
        '''
        # Process the action
        vel_cmd = self.process_action(action)

        # Unpause the sim
        self._pause_sim(False)

        # Moving the agent
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Leave running until next observation comes in
        self.new_obs_event = False
        while not self.new_obs_event:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # Pause the simulation after the next observation comes in
        self._pause_sim(True)

        state, terminated = self.process_obs(self.latest_obs)
        
        reward = self.process_reward(action, state, terminated)

        truncated = False # Used if we ran maximum number of steps.
        return state, reward, terminated, truncated, {}

    
    ## HELPER FUNCTIONS
    def process_action(self, action: int):
        '''
        @brief Translate the action the robot is taking into a Twist message
        @param action: an integer where 0 is left, 1 is forward and 2 is right 
        @retval vel_cmd: Twist message equivalent of the action
        '''
        vel_cmd = RosTwist()

        # Define base speeds (adjust these if the robot moves too fast or slow)
        linear_speed = 0.3
        angular_speed = 0.7

        if action == 0:
            # 0: Drive Left
            vel_cmd.linear.x = linear_speed
            vel_cmd.angular.z = angular_speed
            
        elif action == 1:
            # 1: Drive Forward
            vel_cmd.linear.x = linear_speed
            vel_cmd.angular.z = 0.0
            
        elif action == 2:
            # 2: Drive Right
            vel_cmd.linear.x = linear_speed
            vel_cmd.angular.z = -angular_speed
            
        else:
            # Fallback: Stop
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0
        
        return vel_cmd


    def process_reward(self, action, state, terminated):
        '''
        @brief depending on the action, state and terminated values assign
               a reward
        @retval reward: the reward for the current step
        '''
        
        # ==========================================
        # CONFIGURABLE REWARD PARAMETERS
        # ==========================================
        REWARD_CENTER = 5.0       # Reward for keeping the line in the center
        REWARD_EDGE = 0.0         # Small reward for surviving on the edges
        PENALTY_LOST_LINE = -10.0 # Heavy penalty for losing the line
        # ==========================================

        reward = 0.0

        if terminated:
            # The robot lost the line completely
            reward = PENALTY_LOST_LINE
        else:
            # Reward based on the STATE the robot is currently in
            if state == 1:
                reward = REWARD_CENTER
            elif state == 0 or state == 2:
                reward = REWARD_EDGE

        return reward


    def process_obs(self, obs: RosImage):
        '''
        @brief Identifies the line in the image and populates the state array
        '''
        cv_image = CvBridge().imgmsg_to_cv2(obs, desired_encoding='bgr8')
        height, width = cv_image.shape[:2]

        PIXELS_FROM_BOTTOM = 50   
        THRESHOLD_VALUE = 100      

        scan_y = height - PIXELS_FROM_BOTTOM
        if scan_y < 0: 
            scan_y = 0
            
        row_bgr = cv_image[scan_y:scan_y+1, 0:width]
        row_gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)[0]
        row_is_line = row_gray < THRESHOLD_VALUE
        
        line_indices = np.where(row_is_line)[0]

        # --- VISUALIZATION: Draw State Boundaries ---
        # Draws blue lines showing the 3 zones (Left, Center, Right)
        cv2.line(cv_image, (int(width/3), 0), (int(width/3), height), (255, 0, 0), 2)
        cv2.line(cv_image, (int(2*width/3), 0), (int(2*width/3), height), (255, 0, 0), 2)

        if len(line_indices) > 0:
            left_edge = line_indices[0]
            right_edge = line_indices[-1]
            cx = (left_edge + right_edge) // 2

            # --- VISUALIZATION: Draw Tracking Square ---
            # Draws a red square exactly where the logic detects the line
            square_size = 25
            half_size = square_size // 2
            top_left = (cx - half_size, scan_y - half_size)
            bottom_right = (cx + half_size, scan_y + half_size)
            cv2.rectangle(cv_image, top_left, bottom_right, (0, 0, 255), -1)

            left_boundary = width / 3
            right_boundary = 2 * (width / 3)

            if cx < left_boundary:
                state = 0  
            elif cx > right_boundary:
                state = 2  
            else:
                state = 1  
                
            terminated = False

        else:
            state = -1
            terminated = True

        # --- VISUALIZATION: Show the Live Feed ---
        cv2.imshow("Robot View", cv_image)
        cv2.waitKey(1)  # 1ms delay is required for OpenCV to render the window

        return state, terminated

    # Shutting down ROS2
    def user_close(self):
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()