import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

class LAIONAesthetic():
    def __init__(self, device, clip_model="vit_l_14"):
        """load the aethetic model"""
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError()
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        m.to(device)
        self.model = m

        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device=device
        )

        self.clip = model
        self.preprocess = preprocess
        self.device = device

    def preprocess_tensor(self, img_tensor):
        # img_tensor: [C, H, W] in range [0, 1]
        # Expected input size is [1, 3, 224, 224]

        # Ensure the tensor is float32
        img_tensor = img_tensor.float()

        # Add batch dimension if missing
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]

        # Resize the image to 224x224
        img_tensor = F.interpolate(
            img_tensor, size=(224, 224), mode='bicubic', align_corners=False
        )

        # Normalize the image
        # CLIP mean and std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

        img_tensor = (img_tensor - mean) / std

        return img_tensor  # [1, 3, 224, 224]

    def predict_from_pil(self, pil_image):
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.model(image_features)
        return prediction
    
    def predict_from_tensor(self, img_tensor):
        img_tensor = self.preprocess_tensor(img_tensor).to(self.device)
        image_features = self.clip.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prediction = self.model(image_features)
        return prediction