import cv2
import os
import numpy as np

READ_PATH = os.path.join(os.getcwd(), 'small_test', 'V2')
WRITE_PATH = os.path.join(os.getcwd(), 'small_test', 'V2', 'gray')

def convert_to_gray(read_path, write_path, **kwargs):
    '''
    Converts images to grayscale, assumes that images are stored in the following structure:
    ├── read_path
    |    ├── dataset_directory eg. train, test etc.
    |        ├── category_directory eg. cat, dog etc.
    '''
    for dataset_directory in os.listdir(read_path):    
        for category_directory in os.listdir(os.path.join(read_path, dataset_directory)):
            os.makedirs(os.path.join(write_path, dataset_directory, category_directory), exist_ok=True)        
            for file in os.listdir(os.path.join(read_path, dataset_directory, category_directory)):
                img = cv2.imread(os.path.join(read_path, dataset_directory, category_directory, file), cv2.IMREAD_GRAYSCALE)
                
                cv2.imwrite(os.path.join(write_path, dataset_directory, category_directory, file), img)
                
                silent = kwargs.get('silent', False)
                if not silent:
                    print("Done: ", os.path.join(dataset_directory, category_directory, file))
                
if __name__ == "__main__":
    convert_to_gray(READ_PATH, WRITE_PATH)