import torch
import clip
from torch.utils.data import Dataset
import torch.optim
import torchvision.io as io
import os
from torchvision.transforms import ToPILImage
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model, vit_preprocess = clip.load("ViT-B/32", device=device)


class MP4Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_paths = self._get_video_paths()

    def _get_video_paths(self):
        video_paths = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            for video_file in os.listdir(cls_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(cls_path, video_file)
                    video_paths.append((video_path, cls))
        return video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, cls = self.video_paths[idx]
        # 创建ToPILImage对象
        to_pil = ToPILImage()
        frames,_,_ = io.read_video(video_path, pts_unit='sec')
        # Combine every 8 frames into a sample
        samples = []
        labels = []
        for i in range(0, len(frames), 8):
            sample_frames = frames[i:i+8]
            framesfor5=[]
            if len(sample_frames) == 8:
                for frame in sample_frames:
                    frame = to_pil(frame.permute(2, 1, 0))
                    frame = vit_preprocess(frame).to(device)
                    frame_encoded = vit_model.encode_image(torch.unsqueeze(frame, 0))  # 编码图像
                    framesfor5.append(torch.squeeze(frame_encoded) ) # 将编码后的图像添加到样本中
                samples.append(torch.stack(framesfor5)) 
                labels.append(self.class_to_idx[cls])
        # 将 samples 转换为张量
        samples_tensor = torch.stack(samples)
        print(samples_tensor.shape)
        # Assign class label
        return samples_tensor, torch.stack(labels)