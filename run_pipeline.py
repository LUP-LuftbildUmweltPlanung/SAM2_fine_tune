from create_tiles_unet import split_raster
from predict import predict_and_save_tiles
from train import train_func
import time
import torch

# PARAMETERS
Create_tiles = False
Train = True
Predict = False


# Paths
image_path = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\Prora\rgbi_20cm_2015_resampled_cubic.tif"
mask_path = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\Prora\HCBUND3_attribute_name.tif"
base_dir = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\Prora\tiles"


# Tile creation parameters
#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image
patch_size = 400
patch_overlap = 0
max_empty = 0.2 # Maximum no data area in created image crops
split = [0.8, 0.2] # split the data into train & valid dataset
class_zero = True # Enable for seperating 0 prediction class from nodata


# Training parameters
base_dir_train = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\unet_tiles_rgb_8b_shadi\trai" # Path where two folders: "img_tiles" & "mask_tiles" exist
model_path = r"C:\Users\QuadroRTX\Downloads\SAM2\model" # to save the trained model
description = "canopy_model_sam2_fine_tune_all_data_15_epoch"
model_confg = "large" # 'large', 'base_plus', 'small', 'tiny'  which are  4 different pre-trained SAM 2 models
mode = "binary" # binary if the dataset is (0,1) classification, else #"multi-label"
LEARNING_RATE = 1e-5
EPOCHS = 3
VALID_SCENES = 'vali' # the name of the folder where the validation dataset, 'vali' or 'test'
accuracy_metric = 'loss' # "iou" or "loss
save_confusion_matrix = False # A boolean to enable or disable saving the confusion matrix table."



# Prediction parameters
predict_path = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\test_8bit\vali\img_tiles" # define the images path
predict_model = r"C:\Users\QuadroRTX\Downloads\SAM2\model\model_canopy_model_sam2_fine_tune_30_epoch_best.torch" # the path where the model saved and the name of the model "name.torch"
AOI = "Potsdam" # Area of Interest (AOI). This parameter is used to append the output TIFF file to define the city of the prediction data.
year = "1994" # The year of the prediction data. To append the output TIFF file to define the year.
validation_vision = True # Confusion matrix and classification report figures, Keep merge and regression False to work!
model_confg_predict = "large" # 'large', 'base_plus', 'small', 'tiny'  which are  4 different pre-trained SAM 2 models
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
            max_empty=max_empty,
            class_zero=class_zero
        )

    if Train:
        train_func(
            base_dir_train=base_dir_train,
            model_confg=model_confg,
            epoch=EPOCHS,
            LEARNING_RATE=LEARNING_RATE,
            model_path=model_path,
            description=description,
            mode=mode,
            class_zero=class_zero,
            VALID_SCENES=VALID_SCENES,
            accuracy_metric = accuracy_metric,
            save_confusion_matrix = save_confusion_matrix
        )

    if Predict:
        predict_and_save_tiles(
            input_folder=predict_path,
            model_path=predict_model,
            mode=mode,
            model_confg_predict=model_confg_predict,
            merge=merge,
            class_zero=class_zero,
            validation_vision=validation_vision,
            AOI=AOI,
            year=year,
        )

    end_time = time.time()
    print(f"Operation completed in {(end_time - start_time) / 60:.2f} minutes.")

if __name__ == '__main__':
    main()
