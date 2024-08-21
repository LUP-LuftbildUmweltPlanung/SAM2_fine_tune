import numpy as np
import torch
import os
import tifffile as tiff
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def train_func(base_dir, epoch, model_path, user_defined_name):
    # Automatically define paths to the SAM2 checkpoint and config files based on the current working directory
    current_dir = os.getcwd()  # Get the current working directory
    segment_anything_dir = os.path.join(current_dir, "environment")
    sam2_checkpoint = os.path.join(segment_anything_dir, "checkpoints", "sam2_hiera_large.pt")
    model_cfg = os.path.join(segment_anything_dir, "sam2_configs", "sam2_hiera_l.yaml")

    # Automatically define paths to image and mask tiles
    IMG_path = os.path.join(base_dir, "img_tiles")
    Mask_path = os.path.join(base_dir, "mask_tiles")

    # List of image files
    data = []
    for img_name in os.listdir(IMG_path):
        img_path = os.path.join(IMG_path, img_name)
        mask_path = os.path.join(Mask_path, img_name)
        data.append({"image": img_path, "mask": mask_path})

    current_index = 0

    def read_batch(data):
        nonlocal current_index

        ent = data[current_index]
        current_index = (current_index + 1) % len(data)

        Img = tiff.imread(ent["image"])

        if Img.shape[-1] == 4:
            Img = Img[:, :, :3]

        if Img.dtype == np.float32 or Img.dtype == np.int32:
            Img = ((Img - Img.min()) / (Img.max() - Img.min()) * 255).astype(np.uint8)

        ann_map = tiff.imread(ent["mask"])
        if ann_map.dtype == np.float32 or ann_map.dtype == np.int32:
            ann_map = ((ann_map - ann_map.min()) / (ann_map.max() - ann_map.min()) * 255).astype(np.uint8)

        inds = np.unique(ann_map)[1:]
        points = []
        masks = []
        for ind in inds:
            mask = (ann_map == ind).astype(np.uint8)
            masks.append(mask)
            coords = np.argwhere(mask > 0)
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([[yx[1], yx[0]]])

        return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

    # Load model using the automatically defined paths
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler()

    mean_iou = 0

    for itr in range(epoch):
        with torch.cuda.amp.autocast():
            image, mask, input_point, input_label = read_batch(data)
            if mask.shape[0] == 0:
                continue

            predictor.set_image(image)

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log(
                (1 - prd_mask) + 0.00001)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            predictor.model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if itr % 1000 == 0:
                file_name = "model_{}_{}.torch".format(user_defined_name, itr)
                torch.save(predictor.model.state_dict(), os.path.join(model_path, file_name))

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("step)", itr, "Accuracy(IOU)=", mean_iou)
