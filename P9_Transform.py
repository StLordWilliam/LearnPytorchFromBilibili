from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


img_path = "dataset/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("logs")


tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img", tensor_img)

writer.close()

print(tensor_img)
