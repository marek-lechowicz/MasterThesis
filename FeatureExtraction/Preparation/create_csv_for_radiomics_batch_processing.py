import os
import pandas as pd
import cv2
import numpy as np

# Define paths
READ_PATH = os.path.join(os.getcwd(), 'data', 'genimage', 'RGB', 'imagenet_midjourney', 'red')
WRITE_PATH = os.path.join(os.getcwd(), 'data', 'batch_csv', 'artwork', 'RGB', 'imagenet_midjourney', 'red')

def create_csv_for_radiomics_batch_processing(read_path, write_path, mask_path=None, silent=False):
    '''
    Creates a csv file for batch processing with pyradiomics 
    (read more: https://pyradiomics.readthedocs.io/en/latest/usage.html?highlight=Batch#batch-mode).
    Categories are added as a separate column based on the directory. Assumes that images are stored in the following structure:
    ├── read_path
    |    ├── dataset_directory eg. train, test etc.
    |        ├── category_directory eg. cat, dog etc.
    '''
    
    # to keep track of progress
    all_files = sum([len(files) for r, d, files in os.walk(read_path)])
    file_counter = 0
    for dataset_directory in os.listdir(read_path):
        img_mask_dict = None
        for category_directory in os.listdir(os.path.join(read_path, dataset_directory)):
            for file in os.listdir(os.path.join(read_path, dataset_directory, category_directory)):
                img_path = os.path.join(read_path, dataset_directory, category_directory, file)
                
                # if no mask is passed make the mask covering the whole image
                # and save it named as mask{image_size}.jpg
                # before creating the mask, check if it already exists
                if mask_path is None:
                    mask_path = os.path.join(os.getcwd(), 'data', "masks")
                    os.makedirs(mask_path, exist_ok=True)
                    
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    mask_name = f"mask{img.shape[0]}x{img.shape[1]}.jpg"
                    mask_path = os.path.join(mask_path, mask_name)
                    
                    if not os.path.exists(os.path.join(mask_path, mask_name)): 
                        mask = np.ones_like(img)        
                        cv2.imwrite(mask_path, mask)
                
                if img_mask_dict is None:
                    img_mask_dict = {"Image": [img_path], "Mask": [mask_path], "Category": [category_directory]}
                else:
                    img_mask_dict["Image"].append(img_path)
                    img_mask_dict["Mask"].append(mask_path)
                    img_mask_dict["Category"].append(category_directory)
                
                if not silent:  
                    file_counter += 1
                    if (file_counter / all_files * 100) % 10 == 0:
                        print(f"Done ({file_counter / all_files * 100}%)")
     
            if not silent:   
                print(f" ############### Finished directory {os.path.join(dataset_directory, category_directory)} ############### ")
            
        img_mask_df = pd.DataFrame(img_mask_dict)
        os.makedirs(os.path.join(write_path, dataset_directory), exist_ok=True)
        img_mask_df.to_csv(os.path.join(write_path, dataset_directory, "batch_processing_info.csv"))

if __name__ == "__main__":
    for color in ['red', 'green', 'blue']:
        READ_PATH = os.path.join(os.getcwd(), 'data', 'artwork', 'RGB', color)
        WRITE_PATH = os.path.join(os.getcwd(), 'data', 'batch_csv', 'artwork', 'RGB', color)
        create_csv_for_radiomics_batch_processing(READ_PATH, WRITE_PATH)