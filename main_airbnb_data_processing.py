from tabular_data_class import TabularDataClean
from prepare_image_data import PrepareImages
# from modelling import *

def clean_tabular_data(dataframe, data_dir):
    dataframe_init = dataframe
    dataframe_init.read_file(data_dir)
    dataframe_init.get_num_rows()
    dataframe_init.make_alphabetical_list()
    dataframe_init.reannotate_id()
    dataframe_init.remove_rows_with_missing_ratings()
    dataframe_init.combine_description_strings()
    dataframe_init.set_default_feature_values()
    dataframe_init.save_data_as_cleaned_dataframe()
    # datasets = dataframe_init.load_aribnb("clean_tabular_data.csv")

   

def image_preparation(parent_image_folder_dir):
    image_dir =  PrepareImages()
    image_dir.parent_image_folder_dir=parent_image_folder_dir
    final_image_file_paths,min_height= image_dir.get_min_height()
    image_dir.resize_images(final_image_file_paths,min_height)

if __name__ == "__main__":
    data_dir = 'tabular_data/listing.csv'
    parent_image_folder_dir= "C:/Users/dongc/Desktop/Code/python/AiCore/airbnb/Modelling-Airbnb/airbnb-property-listings/images"
    dataframe_init = TabularDataClean()
    clean_tabular_data(dataframe_init,data_dir)

    # image_preparation(parent_image_folder_dir)
    



