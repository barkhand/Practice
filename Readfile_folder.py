import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Specify the folders
folder_paths = [
    "C:/Users/sansh/AI_TSAI/SAISAI_dataset/zendenkyou/mushroom_masked_r",
]

class ReadImage:
    def __init__(self,folder_paths):
        self.folder_paths=folder_paths

    def folder(self,path):
   

        for folder_path in folder_paths:
            
            img_arrays = []
            image_path_all=[] 
            
            # List all files in the folder
            all_files = os.listdir(folder_path)
            
            # Filter out the images (assuming they are all .jpg files)
            img_files = [f for f in all_files if f.endswith('.jpg')]
            
            # Construct full image paths
            img_paths = [os.path.join(folder_path, f) for f in img_files]
            image_path_all.extend(img_paths)
            # Load and preprocess images
            for img_path in img_paths:
                img_array = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
                img_array = preprocess_input(img_array)
                img_arrays.append(img_array) 
        # Convert the list of image arrays to a numpy array
        img_arrays = np.array(img_arrays)   
        return img_arrays, image_path_all

# Usage
#folder_path = "C:/path/to/folder"
#obj = ReadImage(folder_path)
#images = obj.folder()
#









        






    

        
