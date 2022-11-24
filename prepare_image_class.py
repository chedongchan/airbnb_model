import boto3
import requests
from PIL import Image
import os

class PrepareImages():
    
    def __init__(self) -> None:
        self.parent_image_folder_dir= "C:/Users/dongc/Desktop/Code/python/AiCore/airbnb/Modelling-Airbnb/airbnb-property-listings/images"
        pass

    def download_images(self):
        s3=boto3.resource('s3')
        my_bucket = s3.Bucket('airbnb-listing-data')

        for file in my_bucket.objects.all():
            print(file.key)

    def get_min_height(self):
        image_dimension_dict = {}
        full_image_folder_paths =[]
        final_image_file_paths=[]
        height_list= []
        dirs = os.listdir(self.parent_image_folder_dir)
        
        for folder in dirs:
            full_path = self.parent_image_folder_dir + '\\'+ folder
            full_image_folder_paths.append(full_path)
        for image_folder in full_image_folder_paths:
            subfolder_path = os.listdir(image_folder)
            for image_ID in subfolder_path:
                image_path = image_folder + '\\' + image_ID
                if '.png' in image_path:            
                    width,height = Image.open(image_path).size
                    image_dimension_dict.update({image_ID:[width,height]})
                    height_list.append(height)
                    final_image_file_paths.append(image_path)
        height_list.sort()
        min_height = height_list[0]
        print("This is the minimum height out of all the images.. resizing all images to a height of: " + str(min_height))
        # print(image_dimension_dict)
        return final_image_file_paths, min_height

    def resize_images(self,final_image_file_paths,min_height):
        files_to_delete=[]
        for image in final_image_file_paths:
            im = Image.open(image)
            if im.mode== 'RGB' and '_resized.png' not in image:
                im.thumbnail((100000,min_height))
                new_dir = os.path.splitext(image)[0] + '_resized.png'
                im.save(new_dir)
            else:
                files_to_delete.append(image)

        print('These file(s) were deleted since they were not of RGB type: ' + ','.join(files_to_delete))
        
        for file in files_to_delete:
            os.remove(file)
