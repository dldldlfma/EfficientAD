#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium

from PIL import Image

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='capsule',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='output/1/trainings/mvtec_ad/capsule')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])


def load_teacher_norm_stats(model_dir):
    stats = torch.load(os.path.join(model_dir, "teacher_norm_stats.pth"), map_location="cpu")
    teacher_mean = stats['teacher_mean']  # shape [1, out_channels, 1, 1]
    teacher_std = stats['teacher_std']    # shape [1, out_channels, 1, 1]
    return teacher_mean, teacher_std

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    
    autoencoder = get_autoencoder(out_channels)
    
    teacher_state_dict = torch.load(config.weights + "/teacher_final.pth", map_location='cpu')
    teacher.load_state_dict(teacher_state_dict)
    
    student_state_dict = torch.load(config.weight + "/teacher_final.pth", map_location="cpu")
    student.load_state_dict(student_state_dict)
    
    autoencoder_state_dict = torch.load(config.weight + "/autoencoder_final.pth", map_location="cpu")
    autoencoder.load_state_dict(autoencoder_state_dict)
    
    teacher_mean, teacher_std = load_teacher_norm_stats(config.weight)
    
    
    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
    
    
    if on_gpu:
        image_st = image_st.cuda()
        image_ae = image_ae.cuda()
   

    # run intermediate evaluation
    teacher.eval()
    student.eval()
    autoencoder.eval()
    
    TEST_IMG_PATH = "./datasets/MVTec/capsule/test/crack/000.png"
    image = Image.open(TEST_IMG_PATH)
    image = default_transform(image)
    
    map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
    
    

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


if __name__ == '__main__':
    main()
