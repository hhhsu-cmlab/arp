import torch
from torch import nn
from torchvision import models
import torchvision.transforms as T

class ResnetEncoder(nn.Module):
    def __init__(self, model_cfg, device):
        super().__init__()

        if model_cfg.vision_encoder == "resnet18":
            self.resnet = models.resnet18(pretrained=True)
        elif model_cfg.resnet == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
        else:
            raise NotImplementedError(f"Unknown resnet version: {model_cfg.resnet}")
        
        self.resnet.eval()
        
        assert model_cfg.arp_cfg["n_embd"] % 2 == 0
        self.fc = nn.Linear(1000, model_cfg.arp_cfg["n_embd"] // 2)

        self.transform = T.Compose([
                        T.ToPILImage(),  # 將 Tensor 轉換為 PIL Image
                        T.Resize((224, 224)),  # 調整大小
                        T.ToTensor(),  # 將 PIL Image 轉換為 Tensor
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
        ])

        self.device = device
        
    def forward(self, wrist_camera_rgb_image, front_camera_rgb_image):
        '''
        wrist_camera_rgb_image: (batch_size, 224, 224, 3)
        front_camera_rgb_image: (batch_size, 224, 224, 3)

        Returns: visual feature of size (batch_size, 1, n_embd)
        '''
        batch_size = wrist_camera_rgb_image.shape[0]

        encoder_out=[]
        for i in range(batch_size):
            img = wrist_camera_rgb_image[i] # (224, 224, 3)
            img = img.permute(2, 0, 1)

            # print(img.shape)  # 確認 img 的形狀
            img = self.transform(img)  # 現在可以正常進行轉換

            img = img.unsqueeze(0)  # 增加批次維度

            img = img.half()  # 將輸入數據轉換為 float16

            with torch.no_grad():
                img = img.to(self.device)
                features = self.resnet(img)
                # print(features.shape) # 查看輸出形狀

                # NOTE: 
                # this self.fc should be trainable
                # However, the training before 12/22 21:00 is the following version                
                embedding_1 = self.fc(features)
            embedding_1 = embedding_1.squeeze(0)
            
            img = front_camera_rgb_image[i] # (224, 224, 3)
            img = img.permute(2, 0, 1)
            img = self.transform(img)  # 現在可以正常進行轉換
            img = img.unsqueeze(0)  # 增加批次維度

            img = img.half()  # 將輸入數據轉換為 float16

            with torch.no_grad():
                img = img.to(self.device)
                features = self.resnet(img)
                # print(features.shape) # 查看輸出形狀

                # NOTE: 
                # this self.fc should be trainable
                # However, the training before 12/22 21:00 is the following version  
                embedding_2 = self.fc(features)
            embedding_2 = embedding_2.squeeze(0)

            encoder_out.append(torch.cat((embedding_1, embedding_2), dim=0))

        encoder_out = torch.stack(encoder_out).unsqueeze(1).to(self.device) # (batch_size, 1, n_embd)

        return encoder_out

class DinoV2Encoder(nn.Module):
    def __init__(self, output_feature_dim, device):
        super().__init__()

        self.o_dim = output_feature_dim
        self.device = device

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((518, 518)),  # Resize to 518x518 (ViT L/14)
            T.ToTensor(),  # Convert to Tensor
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # DINOv2's normalization values
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
        self.encoder.eval()

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        assert output_feature_dim % 2 == 0
        self.fc1 = nn.Linear(1024, output_feature_dim // 2)
        self.fc2 = nn.Linear(1024, output_feature_dim // 2)
        
    def forward(self, wrist_camera_rgb_image, front_camera_rgb_image):
        '''
        wrist_camera_rgb_image: (batch_size, 224, 224, 3)
        front_camera_rgb_image: (batch_size, 224, 224, 3)

        Returns: visual feature of size (batch_size, 1, output_feature_dim)
        '''
        batch_size = wrist_camera_rgb_image.shape[0]

        encoder_out=[]
        for i in range(batch_size):
            img = wrist_camera_rgb_image[i] # (224, 224, 3)
            img = img.permute(2, 0, 1)

            img = img.unsqueeze(0)  # 增加批次維度

            img = img.half()  # 將輸入數據轉換為 float16

            with torch.no_grad():
                img = img.to(self.device)
                features = self.encoder(img)

            embedding_1 = self.fc1(features)
            embedding_1 = embedding_1.squeeze(0)
            
            img = front_camera_rgb_image[i] # (224, 224, 3)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)  # 增加批次維度

            img = img.half()  # 將輸入數據轉換為 float16

            with torch.no_grad():
                img = img.to(self.device)
                features = self.encoder(img)

            embedding_2 = self.fc2(features)
            embedding_2 = embedding_2.squeeze(0)

            encoder_out.append(torch.cat((embedding_1, embedding_2), dim=0))

        encoder_out = torch.stack(encoder_out).unsqueeze(1).to(self.device) # (batch_size, 1, n_embd)

        return encoder_out
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.fc1.train(mode)
        self.fc2.train(mode)
        self.encoder.eval()  # Keep encoder in eval mode (frozen)

    def eval(self):
        # Ensures both encoder and fc are in eval mode
        super().eval()
        self.fc1.eval()
        self.fc2.eval()
        self.encoder.eval()

    
