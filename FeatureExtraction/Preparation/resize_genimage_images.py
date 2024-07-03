import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

READ_PATH = "D:\\studia_zadania\\AIContentDetection\\data\\genimage\\original"
WRITE_PATH = "D:\\studia_zadania\\AIContentDetection\\data\\genimage\\resized512x512"
NEW_SIZE = (512, 512)

# READ_PATH = os.path.join(os.getcwd(), 'data', 'genimage', 'original')
# WRITE_PATH = os.path.join(os.getcwd(), 'data', 'genimage', 'resized512x512')

def resize_images(read_path, write_path, new_size, **kwargs):
    '''
    Resize images to new_size, assumes that images are stored in the following structure:
    ├── read_path
    |    ├── dataset_directory eg. train, test etc.
    |        ├── category_directory eg. cat, dog, real, fake etc.
    '''
    for generator_directory in os.listdir(read_path):
        for dataset_directory in os.listdir(os.path.join(read_path, generator_directory)):
            for category_directory in os.listdir(os.path.join(read_path, generator_directory, dataset_directory)):
                # set paths and create directories for R, G, B channels
                write_dir_path = os.path.join(write_path, generator_directory, dataset_directory, category_directory)     
                os.makedirs(write_dir_path, exist_ok=True)
                        
                for file in os.listdir(os.path.join(read_path, generator_directory, dataset_directory, category_directory)):
                    # set write path for resized images
                    write_file_path = os.path.join(write_dir_path, file)
                    
                    # if write file already exists, skip
                    if os.path.exists(write_file_path):
                        continue
                    
                    # read image in rgb
                    img = cv2.imread(os.path.join(read_path, generator_directory, dataset_directory, category_directory, file), cv2.IMREAD_COLOR)
                    
                    # resize image to new_size
                    resized_img = cv2.resize(img, new_size)
                    
                    # save image
                    cv2.imwrite(write_file_path, resized_img)
                    
                    silent = kwargs.get('silent', False)
                    if not silent:
                        print("Done: ", os.path.join(read_path, generator_directory, dataset_directory, category_directory, file))
                
if __name__ == "__main__":
    resize_images(READ_PATH, WRITE_PATH, NEW_SIZE)