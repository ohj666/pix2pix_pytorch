import torch
import cv2
from pix2pix.model import  Generator
import torch
import numpy as np
from pix2pix.preprocessing import  show_img
import torchvision
gen = Generator()
gen.load_state_dict(torch.load('generator_model.pth'))
img = cv2.imread(r"D:\over\new_email\new_email\traing_data\test\test\1.jpeg")
img = img[:,:256,:]
img = np.transpose(img, (2,0,1))
print(img.shape)
img = torch.tensor(img, dtype=torch.float)
img = torch.unsqueeze(img, 0)
result = torch.squeeze(gen(img).detach())
numpy_result = result.numpy()
numpy_result = np.transpose(numpy_result, (1, 2, 0))
cv2.imshow("result", numpy_result)

cv2.waitKey(-1)

