import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

READ_PATH = os.path.join(os.getcwd(), 'data/genimage/original/imagenet_ai_0419_biggan')
WRITE_PATH = os.path.join(os.getcwd(), 'data', 'biggan', 'RGB')

def split_rgb_channels(read_path, write_path, **kwargs):
    '''
    Splits images into R, G, B channels, assumes that images are stored in the following structure:
    ├── read_path
    |    ├── dataset_directory eg. train, test etc.
    |        ├── category_directory eg. cat, dog, real, fake etc.
    '''
    RED_CHANNEL_PREFIX = kwargs.get('RED_CHANNEL_PREFIX', 'R_')
    GREEN_CHANNEL_PREFIX = kwargs.get('GREEN_CHANNEL_PREFIX', 'G_')
    BLUE_CHANNEL_PREFIX = kwargs.get('BLUE_CHANNEL_PREFIX', 'B_')
    
    RED_CHANNEL_DIR = kwargs.get('RED_CHANNEL_DIR', 'red')
    GREEN_CHANNEL_DIR = kwargs.get('GREEN_CHANNEL_DIR', 'green')
    BLUE_CHANNEL_DIR = kwargs.get('BLUE_CHANNEL_DIR', 'blue')

    for dataset_directory in os.listdir(read_path):
        for category_directory in os.listdir(os.path.join(read_path, dataset_directory)):
            # set paths and create directories for R, G, B channels
            red_write_path = os.path.join(write_path, RED_CHANNEL_DIR, dataset_directory, category_directory)
            green_write_path = os.path.join(write_path, GREEN_CHANNEL_DIR, dataset_directory, category_directory)
            blue_write_path = os.path.join(write_path, BLUE_CHANNEL_DIR, dataset_directory, category_directory)
            
            os.makedirs(red_write_path, exist_ok=True)
            os.makedirs(green_write_path, exist_ok=True)
            os.makedirs(blue_write_path, exist_ok=True)
                      
            for file in os.listdir(os.path.join(read_path, dataset_directory, category_directory)):
                # read image in rgb
                img = cv2.imread(os.path.join(read_path, dataset_directory, category_directory, file), cv2.IMREAD_COLOR)
                
                # Convert image from BGR to RGB (matplotlib uses RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # get R G B channels
                r, g, b = cv2.split(img_rgb)
                print(os.path.join(read_path, dataset_directory, category_directory, file))
                
                # save R G B channels
                cv2.imwrite(os.path.join(red_write_path, RED_CHANNEL_PREFIX + file), r)
                cv2.imwrite(os.path.join(green_write_path, GREEN_CHANNEL_PREFIX + file), g)
                cv2.imwrite(os.path.join(blue_write_path, BLUE_CHANNEL_PREFIX + file), b)
                
                silent = kwargs.get('silent', False)
                if not silent:
                    print("Done: ", os.path.join(dataset_directory, category_directory, file))
                
if __name__ == "__main__":
    split_rgb_channels(READ_PATH, WRITE_PATH)