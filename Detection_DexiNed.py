import time
import os
import cv2
import numpy as np
import torch
import torch.optim as optim

from DexiNed_datasets import TestDataset
from DexiNed_model import DexiNed
from torch.utils.data import DataLoader
from DexiNed_image import save_image_batch_to_disk

isTesting = True


def UsingDexiNed(outputFolder):
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    checkpoint_path = '/checkpoints/10_model.pth'   # os.path.join(args.output_dir, args.train_data, args.checkpoint_data)
    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)

    dataset_val = TestDataset('/pic_in',
                              test_data='CLASSIC',
                              img_width=512,
                              img_height=512,
                              mean_bgr=[103.939, 116.779, 123.68],
                              test_list=None
                              )
    print("dataset_val: ")
    print(dataset_val)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=16)

    # output_dir = os.path.join(args.res_dir, args.train_data + "2" + args.test_data)
    test(checkpoint_path, dataloader_val, model, device, outputFolder)
    print('------------------- Test End -----------------------------')


def test(checkpoint_path, dataloader, model, device, output_dir):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()
    print("type: ")
    print(len(dataloader))
    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):

            images = sample_batched['images'].to(device)
            labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            # images = images[:, [2, 1, 0], :, :]

            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)

            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape)
            torch.cuda.empty_cache()

    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", 'CLASSIC', "dataset. *****")
    print("FPS: %f.4" % (len(dataloader) / total_duration))
