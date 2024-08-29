import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random


class env(gym.Env):

    #initiatlisation process for the 3-d environment
    def __init__(self):
        super(env, self).__init__()
        #using the constructor of the gym.Env constructor

        #environment attributes
        self.environment_array = np.full(shape = (20,20,20), fill_value = -1, dtype = np.float32)
        self.action_space = spaces.Discrete(6)
        self.adjusted_low = np.array([0.1, 0.1, 0.1])
        self.adjusted_high = np.array([20 - 0.1, 20 - 0.1, 20 - 0.1])
        self.observation_space = spaces.Box(low=self.adjusted_low, high=self.adjusted_high, dtype=np.float32)

        # self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([20, 20, 20]), dtype=np.float32)

        #rendering is done with pybullet, create physics client 
        self.physicsClient = p.connect(p.GUI) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define the agent
        self.agent = None
        self.agent_start_pos = None
        self.agent_start_orientation = None

        self.destination = None
        
        self.walls =  []
        self.obstacles = []
        self.objects = []

        self.destination_hit = 0

        self.step_size = 1.0

        self.movements = {
            0: [self.step_size, 0, 0], 
            1: [-self.step_size, 0, 0], # right and left, movement is on x axis
            2: [0, self.step_size, 0],  
            3: [0, -self.step_size, 0], #forward and backwards, movement on y axis
            4: [0, 0, self.step_size],
            5: [0, 0, -self.step_size] #up and down, movement on z axis
        }


        #calling functions to initialise the environment, agent, destination, etc.

        self.create_agent(10 ,10 ,15)
        self.create_destination(13,13,17)
        self.create_cuboid_room(20,20,20)
        self.create_obstacles()

        self.objects = self.walls + self.obstacles 
        self.objects.append(self.destination)

        #configuring more pybullet settings
        self.ground = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, 0)
        p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[10, 10, 10])
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)


        
    
        print("Agent ID: ", self.agent)
        print("Destination ID: ", self.destination)
        print("Walls Id: ", self.walls)
        print("All objects list: ", self.objects)

        
       
    # a function to step the agent. this function gets the current location of the agent, then performs 'action' action on it 
    # action is an integer value from 0 to 5, representing each direction
    def step(self, action):
        
        
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent)
        direction = self.movements[action]


        #to make the agent turn towards the direction it is to move to , find yaw value and then convert to quarternion
        yaw = np.arctan2(direction[1], direction[0])
        new_ori = p.getQuaternionFromEuler([0, 0, yaw])

        # new position of agent 
        new_pos = [agent_pos[0] + direction[0],
                agent_pos[1] + direction[1],
                agent_pos[2] + direction[2]]
        
        #set new pos
        p.resetBasePositionAndOrientation(self.agent, new_pos, new_ori)
        
        #check if new position has a collision
        collision_bool, collision_info = self.check_collision()

        # if there was a collision, check for collision type -
        # with obstacles & walls return -100 reward and end episode
        # with destination, give reward
        # normal steps have small punihsment to dissuade more steps
        done = False
        reward = 0

        if collision_bool:
            if self.destination in collision_info:
                reward = 10000
                self.destination_hit += 1
                print("Agent Reached Destination, col_info:", collision_info )
                print("Times destination hit: ", self.destination_hit)

            else:
                reward = -1000
                print("Agent reached obstacles, col_info:", collision_info)
                print("Times destination hit: ", self.destination_hit)

            done = True
        else:
            reward = -1
            done = False
        dis_pos = tuple(map(lambda x: int(round(x)), new_pos))


        return np.array(dis_pos), reward, done, collision_info
    

    #create agenbt with x,y,z coords
    def create_agent(self,x ,y ,z):
        self.agent_start_pos = [x, y, z]
        self.agent_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.agent = p.createMultiBody(baseMass = 0, 
                                    baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[0.5, 0.5,0.5]),
                                    basePosition=self.agent_start_pos,
                                    baseOrientation = self.agent_start_orientation)


    #create environment cuboid with x,y,z coordinates
    def create_cuboid_room(self, x, y, z):
        wall_thickness = 0.1

        self.walls.append( p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[x / 2, y / 2, wall_thickness / 2]),
            basePosition=[x / 2, y / 2, wall_thickness / 2],
        ) )
        self.walls.append( p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[x / 2, y / 2, wall_thickness / 2]),
            basePosition=[x / 2, y / 2, z - wall_thickness / 2],
        ) )
        self.walls.append( p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, y / 2, z / 2]),
            basePosition=[wall_thickness / 2, y / 2, z / 2], 
        ) )
        self.walls.append( p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, y / 2, z / 2]),
            basePosition=[x - wall_thickness / 2, y / 2, z / 2],
        ) )
        self.walls.append( p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[x / 2, wall_thickness / 2, z / 2]),
            basePosition=[x / 2, wall_thickness / 2, z / 2],
        ) )
        self.walls.append( p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[x / 2, wall_thickness / 2, z / 2]),
            basePosition=[x / 2, y - wall_thickness / 2, z / 2],
        ) )
        
    #create objscts in the room
    def create_obstacles(self):

        self.obstacles.append( p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 3, 15]),
                basePosition=[18.5,18.5 , 0],  
                ) )
        
        self.obstacles.append( p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 4, 5]),
                basePosition=[13, 13, 0],
                ) )
            
        self.obstacles.append( p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 5, 8]),
                basePosition=[7, 7, 0], 
                ) )
        
    #create destination location
    def create_destination(self,x ,y ,z):

        self.destination =  p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius = 0.5),
                basePosition=[x, y, z], 
                )



    #check for collisions
    def check_collision(self):
        #step the simulation, then check
        p.stepSimulation()
        agent_id = self.agent

        agent_pos, agent_orientation = p.getBasePositionAndOrientation(agent_id)


        # checks for 0.4 around the object
        extent = 0.4 

        # find the corners of min and max
        aabbMin = [agent_pos[0] - extent, agent_pos[1] - extent, agent_pos[2] - extent]
        aabbMax = [agent_pos[0] + extent, agent_pos[1] + extent, agent_pos[2] + extent]

        overlapping_objects = p.getOverlappingObjects(aabbMin, aabbMax)
        buf_obj = []

        #if overlapping objects, then return the collision 
        if len(overlapping_objects) > 1:
            print("Collision List: ", overlapping_objects)
            for i in range (len(overlapping_objects)):
                if overlapping_objects[i][0] in self.objects:
                    print("Collision detected between the agent and object", overlapping_objects[i][0])
                    buf_obj.append(overlapping_objects[i][0])


            return True, buf_obj 

        return False, None
    
    #this function checks if there is an object currently in a random location by using rays
    #repeatedly generate random values until a free space is found
    def select_random_start(self):

        while True:
            x = random.randint(1,19)
            y = random.randint(1,19)
            z = random.randint(1,19)

            start_point = (x + 0.01, y + 0.01, z + 0.01) 
            end_point = (x -0.01, y + 0.01, z - 0.01) 

            hitInfo = p.rayTest(start_point, end_point)

            if hitInfo[0][0] != -1:
                continue
            else:
                return x, y, z

    #return fuvnctions
    def return_action_space(self):
        return self.action_space
    
    def return_obs_space(self):
        return self.observation_space
    
    def return_reward(self):
        return self.reward
    
    def return_destination_hit(self):
        return self.destination_hit
                

    #reset method, called at start of each episode
    def reset(self):
        # agent position reset to a random empty position, reset reward, and return agent's state (position)

        # x,y,z = 10,10,15

        x,y,z = self.select_random_start()
        p.resetBasePositionAndOrientation(self.agent, (x,y,z) , self.agent_start_orientation)
        print("Agent reset to: ", x, y, z , "to orientation: ", self.agent_start_orientation)

        self.reward = 0
        pos, a = p.getBasePositionAndOrientation(self.agent)
        dis_pos = tuple(map(lambda x: int(round(x)), pos))

        return np.array(dis_pos)


    def close(self):
        pass
