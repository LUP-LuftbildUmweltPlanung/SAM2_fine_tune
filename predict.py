
import numpy as np
import torch
import cv2
import os
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from osgeo import gdal

def read_image(image_path):
    """Read and resize image using Pillow."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert('RGB')  # Convert to RGB
    return np.array(img)

# Merge all the predicted file function
def merge_files(output_folder):
    """Merge all TIF files in the output folder into one."""
    # Find all TIF files in the output folder
    tif_files = glob.glob(os.path.join(output_folder, "*.tif"))

    # Print matched files for debugging
    print("Files to mosaic:", tif_files)

    # Filter out any .ovr files
    tif_files = [f for f in tif_files if not f.endswith('.ovr')]

    # If no files are found, raise an error
    if not tif_files:
        raise RuntimeError("No TIF files found for merging.")

    # Define the nodata value (can be adjusted as needed)
    nodata_value = None

    # Set GDAL warp options for creating the mosaic
    warp_options = gdal.WarpOptions(format="GTIFF", creationOptions=["COMPRESS=LZW", "TILED=YES"], dstNodata=nodata_value)

    # Define the output file path for the merged TIF (outside the output_folder)
    parent_folder = os.path.dirname(output_folder)
    output_file_name = os.path.basename(output_folder)
    output_file = os.path.join(parent_folder, output_file_name + "_merged.tif")

    # Perform the merge using GDAL Warp
    gdal.Warp(output_file, tif_files, options=warp_options)

    print(f"Merged file created at: {output_file}")

def predict_and_save_tiles(input_folder, model_path, merge=False):
    """Predict canopy cover area for all tiles in a folder and save the results."""

    # Automatically define paths to the SAM2 checkpoint and config files based on the current working directory
    current_dir = os.getcwd()  # Get the current working directory
    segment_anything_dir = os.path.join(current_dir, "environment")
    sam2_checkpoint = os.path.join(segment_anything_dir, "checkpoints", "sam2_hiera_large.pt")
    model_cfg = os.path.join(segment_anything_dir, "sam2_configs", "sam2_hiera_l.yaml")

    # Load model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load(model_path, map_location="cuda"))

    # Automatically create an output folder beside the input folder
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    # Get the parent directory of the input_folder
    parent_folder = os.path.dirname(input_folder)
    # Create the output_folder in the parent directory
    output_folder = os.path.join(parent_folder, f"{model_name}_predict_tiles")
    os.makedirs(output_folder, exist_ok=True)

    # Prediction loop
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for image_file in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_file)
            if not image_file.lower().endswith(('.tif', '.tiff')):
                continue

            # Read image and generate points
            image = read_image(image_path)

            # Predict masks for the entire image automatically by not passing any points
            with torch.no_grad():
                predictor.set_image(image)
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None
                )

            # Check if scores are 1-dimensional and handle accordingly
            if scores.ndim == 1:
                np_scores = scores
            else:
                np_scores = scores[:, 0]  # If 2D, take the first column (as previously assumed)

            # Convert scores to numpy if necessary
            if isinstance(np_scores, torch.Tensor):
                np_scores = np_scores.cpu().numpy()

            # Sort masks by scores
            sorted_indices = np.argsort(np_scores)[::-1]
            sorted_masks = masks[sorted_indices]

            # Stitch predicted masks into one binary segmentation mask (0 and 1)
            if sorted_masks.ndim == 3:  # Assuming masks are (N, H, W)
                seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
                occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
            else:
                raise ValueError("Unexpected mask dimensions: expected 3D array for masks")

            for i in range(sorted_masks.shape[0]):
                mask = sorted_masks[i].astype(bool)
                if (mask & occupancy_mask).sum() / mask.sum() > 0.15:
                    continue
                mask[occupancy_mask] = False
                seg_map[mask] = 1  # Ensure binary (0 and 1) values
                occupancy_mask |= mask

            # Save the segmentation mask as a TIF file in EPSG:25832
            output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + "_predicted.tif")

            with rasterio.open(image_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, 'EPSG:25832', src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': 'EPSG:25832',
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'count': 1,  # Ensure the output has a single band for binary mask
                    'dtype': 'uint8'  # Ensure the data type is uint8 (suitable for binary data)
                })

                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    reproject(
                        source=seg_map,
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs='EPSG:25832',
                        resampling=Resampling.nearest
                    )

    if merge:
        merge_files(output_folder)






