import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from model.enet import ENet
from model.attention_block import SENet, CBAM, SelfAttentionBlock
import torch
from torchvision import transforms
import albumentations
import matplotlib.pyplot as plt
import numpy as np
import cv2


class LungSegmentation:
    def __init__(self, checkpoint_path,
                       image_size = 320,
                       device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        #load model, checkpoint
        self.model = ENet(num_classes = 1,
                          attention_block= SENet,
                          ignore_attention=[])
        self.model.to(device)
        self.model.eval()
        self.load_checkpoints(checkpoint_path)

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
            transforms.Resize((image_size, image_size))
        ])
        self.image_size = image_size


    def predict(self, img_array, show = None):
        assert img_array is not None
        with torch.no_grad():
            img_tensor = self.preprocess(img_array)
            outputs = self.model(img_tensor)
        predicts = self.posprocess(outputs)
        return predicts

    def preprocess(self, img):
        self.org_w, self.org_h, _ = img.shape
        img_tensor = self.transform_test(img)
        img_tensor = img_tensor.to(self.device)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor

    def posprocess(self, outputs, threshold = 0.5):
        if isinstance(outputs, list):
            outputs = outputs[-1]
        predicts = torch.sigmoid(outputs)[0]
        predicts = predicts.cpu().numpy().transpose(1, 2, 0)[:,:,0]
        predicts = (predicts > threshold).astype(np.uint8)
        predicts = cv2.resize(predicts, (self.org_h, self.org_w))
        return predicts

    def visualize(self, img_array, predicts):
        img = cv2.resize(img_array[:,:,0])/255.0
        plt.imshow(np.hstack((img, predicts)), cmap = "gray")
        plt.show()

    def load_checkpoints(self, checkpoint_path):
        assert self.model is not None
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict["net"])


if __name__ == "__main__":
    segments = LungSegmentation(checkpoint_path = "checkpoint/SE_checkpoint_889", device="cpu")
    img_array = plt.imread("images/7aa6611b60c2e6115fdcdb7194e1f2_jumbo.jpg")
    predict = segments.predict(img_array)
    segments.visualize(img_array, predict)

def init_model(model_name):
    if model_name == "ENet":
        model = ENet( num_classes  = 1)
    elif model_name == "ENet_SE":
        model = ENet(num_classes = 1,
                            attention_block= SENet)
    elif model_name == "ENet_CBAM":
        model = ENet(num_classes = 1,
                            attention_block= CBAM)
    elif model_name == "ENet_CBAM":
        model = ENet(num_classes = 1,
                            attention_block= SelfAttentionBlock,
                            ignore_attention=[1, 2, 5, 6])
    else:
        model = None
    return model