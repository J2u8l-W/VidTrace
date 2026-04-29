import torch
import clip
from torch.utils.data import Dataset
import torch.optim
import os
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model, vit_preprocess = clip.load("ViT-B/32", device=device)
L=8
def read_gif(gif_path):
    frames = []
    with Image.open(gif_path) as img:
        while True:
            try:
                img.seek(img.tell() + 1)
                frames.append(img.copy())
            except EOFError:
                break
    return frames
def gif2npy(root,L,outdir):
    os.makedirs(outdir,exist_ok=True)
    for mp4path in os.listdir(root):
        frames = read_gif(os.path.join(root,mp4path))
        to_pil = ToPILImage()
        # Combine every L frames into a sample
        for i in range(0, len(frames), L):
            sample_frames = frames[i:i+L]
            if len(sample_frames) < L:
                last_frame = sample_frames[-1]
                sample_frames += [last_frame]
                
            framesforL=[]
            if len(sample_frames) == L:
                for frame in sample_frames:
                    frame = vit_preprocess(frame).to(device)
                    frame_encoded = vit_model.encode_image(torch.unsqueeze(frame, 0))  # 编码图像
                    framesforL.append(torch.squeeze(frame_encoded).cpu().detach().numpy()) # 将编码后的图像添加到样本中
                print(np.array(framesforL).shape)
                output_file = os.path.join(outdir, mp4path.split('.')[0]+"_"+str(i)+".npy")
                np.save(output_file, framesforL)
                print("save ",mp4path+"_"+str(i)+".npy to ",outdir)
root="data/raw/Tune-a-Video/Tune-a-Video"
gif2npy(root,L,"data/npy/Tune-a-Video/0")
