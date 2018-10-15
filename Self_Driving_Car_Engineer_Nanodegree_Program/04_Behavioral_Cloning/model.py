#-------------------------- Imports (Start) ------------------------------------

# Native modules
import cv2
import csv
import os
import numpy as np
from copy import deepcopy as deepcopy
from six import string_types

# Third party modules
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Custom modules

#-------------------------- Imports (End) --------------------------------------

#-------------------------- Classes (Start) ------------------------------------

class BehavioralCloningModel(object):
    """
    Description: Class to manage model architecture creation.
    """
    
    def __init__( self ):
        """
        Description: Constructor-like method.
        Number of arguments: 0
        Arguments: N/A
        Return: N/A
        """
        
        pass
    
    def get_rows_from_logs( self,
                            in_data_path = None,
                            in_data_filename = 'driving_log.csv',
                            in_skip_header = False ):
        """
        Description: Extract data from each row of the .csv file.
        Number of arguments: 3
        Arguments:
           in_data_path    Type: string
                           Description: Path to .csv file.
                           Default: None
           in_data_filename    Type: string
                               Description: .csv file name.
                               Default: '/driving_log.csv'
           in_skip_header  Type: boolean
                           Description: True - Skip the first row because
                                               it is a header
                                        False - Do not skip the first row.
                           Default: False
        Return: -1 - Bad argument input
                0 - Failed
                
                else:
                
                list of list of tokens of each line in .csv file.
        """
        
        #-------------------
        # Input Validation
        # (Start)
        #-------------------
        
        if isinstance( in_data_path, string_types) == False:
            return -1
           
        if isinstance( in_data_filename, string_types) == False:
            return -1
        
        if isinstance( in_skip_header, bool) == False:
            return -1
        
        #-------------------
        # Input Validation
        # (End)
        #-------------------
        
        #-------------------
        # Variables
        # (Start)
        #-------------------
        
        # Pass argument variables to local variables.
        data_path = deepcopy( in_data_path )
        data_filename = deepcopy( in_data_filename )
        skip_header = deepcopy( in_skip_header )
        
        # List of tokens per row to return.
        rows = []
        
        #-------------------
        # Variables
        # (End)
        #-------------------
        
        # Construct the path
        path_and_file = os.path.join( data_path,
                                     data_filename )
        #
        # Normalized the path based on operating system.
        # (i.e. windows may use \\ where as linux may use /
        #  do not want something like c:/temp\\data.csv but either
        #  c:/temp/data.csv or c:\\temp\\data.csv
        path_and_file = os.path.abspath(path_and_file)
        
        with open( path_and_file ) as csv_file:
           reader = csv.reader(csv_file)
           if skip_header:
               next(reader, None)
           for row in reader:
               rows.append(row)
        
        return rows

    def get_images( self,
                    in_data_path = None,
                    in_data_filename = 'driving_log.csv' ):
        """
        Description: Given a root folder, find all nested folders recursively
                     and look for the .csv file that list where to find
                     all the images. Extract from the .csv file all the 
                     image paths for the center, left, right and the 
                     steering measurements.
        Number of arguments: 2
        Arguments:
            in_data_path    Type: string
                            Description: Path to .csv file.
                            Default: None
            in_data_filename    Type: string
                                Description: .csv file name.
                                Default: '/driving_log.csv'
        Return: -1 - Bad argument input
                 0 - Failed
                 
                 else:
                 
                 Tuple of
                 ( list_images_center,
                   list_images_left,
                   list_images_right,
                   list_measurement_steering )
        """
        
        #-------------------
        # Input Validation
        # (Start)
        #-------------------
        
        if isinstance( in_data_path, string_types) == False:
            return -1
           
        if isinstance( in_data_filename, string_types) == False:
            return -1
        
        #-------------------
        # Input Validation
        # (End)
        #-------------------
        
        #-------------------
        # Variables
        # (Start)
        #-------------------
        
        # Pass argument variables to local variables.
        data_path = deepcopy( in_data_path )
        data_filename = deepcopy( in_data_filename )
        
        # To be returned in tuple
        list_images_center = []
        list_images_left = []
        list_images_right = []
        list_measurement_steering = []
        
        #-------------------
        # Variables
        # (End)
        #-------------------
        
        # What this is doing is looking for every folder in the root folder
        # recursively. It then checks if that folder has the .csv file in it.
        # A list of full path folders that have the .csv files is the result.
        # Using list comprehension and functional approach to programming.
        # NOTE: filter()  creates a list of elements for which a 
        #       function returns true.
        #       lambda() used to make small one-off functions.
        #                lambda arguments : expression
        list_folders = [x[0] for x in os.walk(data_path)]
        list_folders_filtered = \
            list( filter( 
                lambda current_folder: os.path.isfile(
                  os.path.abspath(os.path.join(current_folder,data_filename))),
                  list_folders))
        
        # Cycle through the folders that have the .csv file and extract
        # the data.
        for current_folder in list_folders_filtered:
            rows = self.get_rows_from_logs( in_data_path = current_folder,
                                            in_skip_header = True )
            center = []
            left = []
            right = []
            measurements = []
            for row in rows:
                measurements.append(float(row[3]))
                center.append( os.path.abspath(
                    os.path.join(current_folder, row[0].strip())))
                left.append(os.path.abspath(
                    os.path.join(current_folder, row[1].strip())))
                right.append(os.path.abspath(
                    os.path.join(current_folder, row[2].strip())))
            list_images_center.extend(center)
            list_images_left.extend(left)
            list_images_right.extend(right)
            list_measurement_steering.extend(measurements)
        
        return ( list_images_center,
                 list_images_left,
                 list_images_right,
                 list_measurement_steering )
                 
    def consolidate_data( self,
                          in_list_image_paths_center = None,
                          in_list_image_paths_left = None,
                          in_list_image_paths_right = None,
                          in_list_measurement_steering = None,
                          in_correction_value = 0.2 ):
        """
        Description: The data extracted using the .get_images() method
                     returns list of center, left, and right images as well
                     as a list of steering measurement for the center image.
                     This method consolidates all the images into 1 list and
                     also pre-processes a corresponding list of the same
                     number of elements for the steering measurement. The 
                     steering measurement for the center will be the same
                     number, but the left will be adding a correction value 
                     and the right will be subtracing a correctoin value.
        Number of Arguments: 5
        Arguments: 
            in_list_image_paths_center  Type: List of strings
                                        Description: List of full path to 
                                                     center images.
                                        Default: None
            in_list_image_paths_left    Type: List of strings
                                        Description: List of full path to 
                                                     left images.
                                        Default: None
            in_list_image_paths_right   Type: List of strings
                                        Description: List of full path to 
                                                     right images.
                                        Default: None
            in_list_measurement_steering   Type: List of strings
                                           Description: List of full path to 
                                                     right images.
                                           Default: None
            in_correction_value        Type: float
                                           Description: Offset correction value
                                                   for left and right image.
                                           Default: 0.2
        Return: -1 - Bad input argument
                 0 - Failed
                 
                 else:
                 Tuple of list
                     ( list_image_paths,
                       list_measurement_steerings )
        """
        
        #-------------------
        # Input Validation
        # (Start)
        #-------------------
        
        # if isinstance( in_list_image_paths_center, list) == False:
        #     in_list_image_paths_center = [in_list_image_paths_center]
            
        #     if all([isinstance(x, string_types) \
        #         for x in in_list_image_paths_center]) == False:
        #        return -1
           
        # if isinstance( in_list_image_paths_left, list) == False:
        #     in_list_image_paths_left = [in_list_image_paths_left]
            
        #     if all([isinstance(x, string_types) \
        #         for x in in_list_image_paths_left]) == False:
        #        return -1
        
        # if isinstance( in_list_image_paths_right, list) == False:
        #     in_list_image_paths_right = [in_list_image_paths_right]
            
        #     if all([isinstance(x, string_types) \
        #         for x in in_list_image_paths_right]) == False:
        #        return -1
        
        # if isinstance( in_list_measurement_steering, list) == False:
        #     in_list_measurement_steering = [in_list_measurement_steering]
            
        #     if all([isinstance(x, float) \
        #         for x in in_list_measurement_steering]) == False:
        #        return -1
               
        # if isinstance( in_correction_value, float) \
        #    and \
        #    isinstance( in_correction_value, int) == False:
        #     return -1
            
        #-------------------
        # Input Validation
        # (End)
        #-------------------
        
        #-------------------
        # Variables
        # (Start)
        #-------------------
        
        # Pass argument variables to local variables.
        list_image_paths_center = deepcopy( in_list_image_paths_center )
        list_image_paths_left = deepcopy( in_list_image_paths_left )
        list_image_paths_right = deepcopy( in_list_image_paths_right )
        list_measurement_steering = deepcopy( in_list_measurement_steering )
        correction_value = deepcopy( in_correction_value )
        
        # List of tokens per row to return.
        list_image_paths = []
        list_measurement_steerings = []
        
        #-------------------
        # Variables
        # (End)
        #-------------------
        
        # Populate the images list
        list_image_paths.extend( list_image_paths_center )
        list_image_paths.extend( list_image_paths_left )
        list_image_paths.extend( list_image_paths_right )
        
        # Poplate and pre-process the steering measurement list.
        list_measurement_steerings.extend(list_measurement_steering)
        list_measurement_steerings.extend(
            [float(x) + float(correction_value) 
                for x in list_measurement_steering])
        list_measurement_steerings.extend(
            [float(x) - float(correction_value) 
                for x in list_measurement_steering])
        
        return ( list_image_paths,
                 list_measurement_steerings )
                 
    def generator( self,
                   in_samples = None,
                   in_batch_size=32):
        """
        Description: Based on code from 
                     Project Behavioral Cloning - Lesson 18. Generators
        Number of arguments: 2
        Arguments:
            in_samples  Type: list
                        Description: List of items, does not matter what
                                     the type is in the list.
                        Default: None
            in_batch_size Type: int
                          Description: batch size portions to feed into 
                                       Behavioral Cloning model
        Return: Instead of using return, the generator uses yield, which 
                still returns the desired output values but saves the current 
                values of all the generator's variables. When the generator is 
                called a second time it re-starts right after the yield 
                statement, with all its variables set to the same values 
                as before.
        """
        
        #-------------------
        # Input Validation
        # (Start)
        #-------------------
        
        if isinstance( in_samples, list) == False:
            return -1
           
        if isinstance( in_batch_size, int) == False:
            return -1
        
        #-------------------
        # Input Validation
        # (End)
        #-------------------
        
        #-------------------
        # Variables
        # (Start)
        #-------------------
        
        # Pass argument variables to local variables.
        samples = deepcopy(in_samples)
        batch_size = deepcopy(in_batch_size)
        
        #-------------------
        # Variables
        # (End)
        #-------------------
        
        num_samples = len(samples)
        
        # Loop forever so the generator never terminates
        while 1: 
            samples = sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                
                images = []
                angles = []
                for image_path, measurement in batch_samples:
                    original_image = cv2.imread(image_path)
                    
                    # NOTE: Make sure to convert to the right color space.
                    image = cv2.cvtColor( original_image,
                                          cv2.COLOR_BGR2RGB )
                    images.append(image)
                    angles.append(measurement)
                    
                    # Data augmentation
                    # Idea from 
                    # Project Behavioral Cloning - Lesson 12. Data Augmentation
                    #
                    # Flipping using openCV
                    # https://docs.opencv.org/2.4/modules/core/doc/
                    # operations_on_arrays.html#flip
                    images.append(cv2.flip(image,1))
                    angles.append(measurement*-1.0)
                    
                # trim image to only see section with road
                inputs = np.array(images)
                outputs = np.array(angles)
                
                yield sklearn.utils.shuffle(inputs, outputs)
                
    def create_model_and_preprocess_layers(self):
        """
        Description: Creates a Keras sequential model and preprocessing layers
                     of normalization, mean centering and cropping.
                     
                     Based on code from 
                     Project Behavioral Cloning - Lesson 10. Data Preprocessing
                     Project Behavioral Cloning - Lesson 14. Cropping Images 
                                                                       in Keras
        Number of arguments: 0
        Arguments: N/A
        Return: Keras sequential model
        """
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
        
        # 50 rows pixels from the top of the image
        # 20 rows pixels from the bottom of the image
        # 0 columns of pixels from the left of the image
        # 0 columns of pixels from the right of the image
        model.add(Cropping2D(cropping=((50,20), (0,0))))
        return model
        
    def model_architecture_placeholder(self):
        """
        Description: Placeholder basic neural network to verify the pipeline
                     is working. It's predicted output is expected to be 
                     terrible.
                     
                    Based on code from
                    Project Behavioral Cloning - Lesson 8. Training Your 
                                                                        Network
                     
        Number of arguments: 0
        Arguments: N/A
        Return: N/A
        """
        
        # Create and pre-process model architecture and layers
        model = self.create_model_and_preprocess_layers()
        
        # basic neural network.
        model.add(Flatten(input_shape = (160, 320, 3)))
        model.add(Dense(1))

        return model
        
    def model_architecture_lenet(self):
        """
        Description: Based on LeNet model architecture.
                     Project Behavioral Cloning - Lesson 11. More Networks
                     
        Number of arguments: 0
        Arguments: N/A
        Return: N/A
        """
        
        # Create and pre-process model architecture and layers
        model = self.create_model_and_preprocess_layers()
        
        # Based on LeNet5 architecture
        model.add(Convolution2D(6,5,5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Convolution2D(6,5,5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))
        
        return model
        
    def model_architecture_nvidia(self):
        """
        Description: NVIDIA's End to End Learning for Self-Driving Cars
                     Based on code from 
                     Project Behavioral Cloning - Lesson 15. Even more powerful
                             network
                     
                     Based on paper: 
                     http://images.nvidia.com/content/tegra/automotive/images/
                     2016/solutions/pdf/end-to-end-dl-using-px.pdf
        Number of arguments: 0
        Arguments: N/A
        Return: N/A
        """
        
        model = self.create_model_and_preprocess_layers()
        model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        
        return model


#-------------------------- Classes (End) --------------------------------------

#-------------------------- Auto-Execute (Start) -------------------------------

if __name__ == "__main__":
    
    # Root folder holding data.
    root_folder = '/opt/carnd_p3/data'
    file_name_model = 'model.h5'
    
    # Instantiate object from class
    o_bcm = BehavioralCloningModel()
    
    # Get data from root folder
    a, b, c, d = o_bcm.get_images( in_data_path = root_folder,
                                   in_data_filename = 'driving_log.csv' )
    list_images_center = a
    list_images_left = b
    list_images_right = c
    list_measurement_steering = d
    
    a, b = o_bcm.consolidate_data( 
        in_list_image_paths_center = list_images_center,
        in_list_image_paths_left = list_images_left,
        in_list_image_paths_right = list_images_right,
        in_list_measurement_steering = list_measurement_steering,
        in_correction_value = 0.2 )
    #
    list_image_paths = a
    list_measurement_steerings = b
    
    
    #----------------------------
    
    
    # Using scikit learn to make train and test split.
    from sklearn.model_selection import train_test_split
    samples = list(zip(list_image_paths, list_measurement_steerings))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    # Creating generators
    train_generator = o_bcm.generator(
                               in_samples = train_samples,
                               in_batch_size=32 )
    validation_generator = o_bcm.generator(
                               in_samples = validation_samples,
                               in_batch_size=32 )
    
    
    #----------------------------
    
    # Create a model architecture
    #model = o_bcm.model_architecture_placeholder()
    #model = o_bcm.model_architecture_lenet()
    model = o_bcm.model_architecture_nvidia()
    
    # Compile the model
    
    model.compile( loss='mse',
                   optimizer='adam')
                   
    # Fit (i.e. train) the model but use generators
    o_history = model.fit_generator(
                        train_generator,
                        samples_per_epoch= len(train_samples),
                        validation_data = validation_generator,
                        nb_val_samples = len(validation_samples),
                        nb_epoch = 3,
                        verbose = 1 )

    # Save the model
    model.save(file_name_model)
    
    #----------------------------
    
    # Print final information
    print( o_history.history.keys() )
    print( 'final loss' )
    print( o_history.history['loss'] )
    print( 'final validation loss' )
    print( o_history.history['val_loss'])
    
#-------------------------- Auto-Execute (End) ---------------------------------
