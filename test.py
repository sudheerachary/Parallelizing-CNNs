import sys
import torch
from alexnet import *
model = alexnet(pretrained=False, progress=True, num_classes=1000)
x = torch.load(sys.argv[1])
model.load_state_dict(x["state_dict"])
model.eval()

from PIL import Image
from torchvision import transforms
input_image = Image.open(sys.argv[2])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
output = torch.nn.functional.softmax(output[0], dim=0)

top5 = torch.topk(output, k=5)

confidences = []; class_labels = []
for i in range(5):
    confidences.append(top5.values[i].item()*100)
    class_labels.append(top5.indices[i].item())

with open('labels.txt', 'r') as f:
    lines = f.readlines()

for i in range(5):
    class_label = lines[class_labels[i]].split(':')[1]
    print("{} ({})".format(class_label, confidences[i]))
