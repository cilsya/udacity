from math import *
import random
#import numpy as np

### ------------------------------------- ###
# Below, is the robot class
#
# This robot lives in 2D, x-y space, and its motion is
# pointed in a random direction, initially.
# It moves in a straight line until it comes close to a wall 
# at which point it stops.
#
# For measurements, it  senses the x- and y-distance
# to landmarks. This is different from range and bearing as
# commonly studied in the literature, but this makes it much
# easier to implement the essentials of SLAM without
# cluttered math.
#
class robot:
    
    # --------
    # init:
    #   creates a robot with the specified parameters and initializes
    #   the location (self.x, self.y) to the center of the world
    #
    def __init__(self, world_size = 100.0, measurement_range = 30.0,
                 motion_noise = 1.0, measurement_noise = 1.0):
        self.measurement_noise = 0.0
        self.world_size = world_size
        self.measurement_range = measurement_range
        self.x = world_size / 2.0
        self.y = world_size / 2.0
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.landmarks = []
        self.num_landmarks = 0
    
    
    # returns a positive, random float
    def rand(self):
        return random.random() * 2.0 - 1.0
    
    
    # --------
    # move: attempts to move robot by dx, dy. If outside world
    #       boundary, then the move does nothing and instead returns failure
    #
    def move(self, dx, dy):
        
        x = self.x + dx + self.rand() * self.motion_noise
        y = self.y + dy + self.rand() * self.motion_noise
        
        if x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size:
            return False
        else:
            self.x = x
            self.y = y
            return True


    # --------
    # sense: returns x- and y- distances to landmarks within visibility range
    #        because not all landmarks may be in this range, the list of measurements
    #        is of variable length. Set measurement_range to -1 if you want all
    #        landmarks to be visible at all times
    #
    
    ## TODO: paste your complete the sense function, here
    ## make sure the indentation of the code is correct
    def sense(self):
        ''' This function does not take in any parameters, instead it references internal variables
            (such as self.landamrks) to measure the distance between the robot and any landmarks
            that the robot can see (that are within its measurement range).
            This function returns a list of landmark indices, and the measured distances (dx, dy)
            between the robot's position and said landmarks.
            This function should account for measurement_noise and measurement_range.
            One item in the returned list should be in the form: [landmark_index, dx, dy].
            '''
           
        measurements = []
        
        ## TODO: iterate through all of the landmarks in a world
        for i in range(len(self.landmarks)):
            
            
            ## TODO: For each landmark
            current_landmark = self.landmarks[i]
            current_landmark_id = i
            
            ## 1. compute dx and dy, the distances between the robot and the landmark
            dx = current_landmark[0] - self.x
            dy = current_landmark[1] - self.y
                        
            ## 2. account for measurement noise by *adding* a noise component to dx and dy
            ##    - The noise component should be a random value between [-1.0, 1.0)*measurement_noise
            ##    - Feel free to use the function self.rand() to help calculate this noise component
            ##    - It may help to reference the `move` function for noise calculation
            
            # Add random deviation of the measurement noise to the landmark distance components.
            # Using the self.rand() method provided.
            dx = self.measurement_noise * self.rand() + dx
            dy = self.measurement_noise * self.rand() + dy
            
            ## 3. If either of the distances, dx or dy, fall outside of the internal var, measurement_range
            ##    then we cannot record them; if they do fall in the range, then add them to the measurements list
            ##    as list.append([index, dx, dy]), this format is important for data creation done later
        
            # Measure distance. We cannot sense outside the range that is visible to the robot,
            # unless self.measurement_range was set to -1
            if self.measurement_range != -1:
                
                # Review Note for change:
                # You are logically correct here. But the robot is little dumb and doesn't 
                # make moves along shortest path as you would have seen in the move function. 
                # The robot only moves in x and y direction. So, when comparing the measurement 
                # range, you should compare absolute values of dx and dy individually with the 
                # measurement range.
                #
                # How I interpret this note is that the distance is a Manhanttan distance, meaning
                # there are no diagonals. Get the distance from the x component only and add it
                # to the distance of the y component only
                
                #landmark_distance = np.sqrt(dx**2 + dy**2)
                #landmark_distance = sqrt(dx**2 + dy**2)
                landmark_distance = dx + dy
                
                # Skip iteration if outside visibility range.
                if landmark_distance > self.measurement_range:
                    continue
                    
            # If we got this far, we can add the measurement to the list
            # in the form [landmark_index, dx, dy]
            measurements.append( [ current_landmark_id,
                                   dx,
                                   dy ] )
        
        ## TODO: return the final, complete list of measurements
        return measurements


    # --------
    # make_landmarks:
    # make random landmarks located in the world
    #
    def make_landmarks(self, num_landmarks):
        self.landmarks = []
        for i in range(num_landmarks):
            self.landmarks.append([round(random.random() * self.world_size),
                                   round(random.random() * self.world_size)])
        self.num_landmarks = num_landmarks


    # called when print(robot) is called; prints the robot's location
    def __repr__(self):
        return 'Robot: [x=%.5f y=%.5f]'  % (self.x, self.y)



####### END robot class #######