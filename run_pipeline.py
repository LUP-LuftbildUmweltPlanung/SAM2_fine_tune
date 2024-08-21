from create_tiles_unet import split_raster
from predict import predict_and_save_tiles
from train import train_func

import time
import torch

# PARAMETERS
Create_tiles = False
Train = False
Predict = True


# Paths
image_path = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\unit_mosaic_fig\Hanau_2021\Hanau_2021_belaubt_tDOP_RGB_8bit_stack_50cm.tif"
mask_path = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\unit_mosaic_fig\Hanau_2021\hanau_2021_canopy_cover_binary_50cm.tif"
base_dir = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\SAM2_test_2"


# Tile creation parameters
#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image
patch_size = 400
patch_overlap = 0.2
max_empty = 0.2 # Maximum no data area in created image crops
split = [0.8, 0.2] # split the data into train & valid dataset


# Training parameters
base_dir = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\SAM2_test_2\trai" # Path where two folders: "img_tiles" & "mask_tiles" exist
model_path = r"C:\Users\QuadroRTX\Downloads\SAM2\model" # to save the trained model
user_defined_name = "canopy_model"
EPOCHS = 100 # the number of Img the model should go through them


# Prediction parameters
predict_path = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\SAM2_test\vali\img_tiles" # define the images path
predict_model = r"C:\Users\QuadroRTX\Downloads\SAM2\model\model_canopy_model_0.torch" # the path where the model saved and the name of the model "name.torch"
merge = True

def main():
    start_time = time.time()

    if torch.cuda.is_available():
        print("CUDA device is available.")
    else:
        print("No CUDA device available, running on CPU.")

    if Create_tiles:
        split_raster(
            path_to_raster=image_path,
            path_to_mask=mask_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            base_dir=base_dir,
            split=split,
            max_empty=0.2
        )

    if Train:
        train_func(
            base_dir=base_dir,
            epoch=EPOCHS,
            model_path=model_path,
            user_defined_name=user_defined_name
        )

    if Predict:
        predict_and_save_tiles(
            model_path=predict_model,
            input_folder=predict_path,
            merge=merge
        )

    end_time = time.time()
    print(f"Operation completed in {(end_time - start_time) / 60:.2f} minutes.")

if __name__ == '__main__':
    main()
