from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2

# the use of the data type "tensor"
# think about the two problems through transforms.ToTensor:
# 1.how to transforms?
# 2.why we need the data type "tensor"

# open image by inline function "Image.open()"
# absolute path D:\CodeProject\Pycharm\transforms_training\C-7.jpg

image_path = r"D:\CodeProject\Pycharm\transforms_training\C-7.jpg"
image = Image.open(image_path)
image1 = cv2.imread(image_path)

# use the function "SummaryWriter" to record the change of image, 
# and generate log
writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(image)
tensor_img1 = tensor_trans(image1)

writer.add_image("Tensor_image", tensor_img)
writer.close()

# look over the logs by the command "tensorboard --logdir= dir_name"
# if the port 6006 have no reaction, we can change the port by
# command "tensorboard --logdir=dir_name --port 6005(or any other
# port which can be used in browser)
