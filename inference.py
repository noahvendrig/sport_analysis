
# model = torch.hub.load(yolo_dir, 'custom',  source='local', path=os.path.join(weights_dir, "gelan-c.pt"), force_reload = True)
# model.eval()

# output = model(torch.rand(1, 3, 640, 640))  # dry run
# prediction = int(torch.max(output.data, 1)[1].numpy())
# print(prediction)


import cv2
import torchvision.transforms as transforms
from yolov9.models.common import non_max_suppression
import os
# import sys
import torch

from yolov9.models.yolo import Model

cwd = os.getcwd()
yolo_dir = os.path.join(cwd, "yolov9")
weights_dir = os.path.join(yolo_dir, "weights")

model = torch.hub.load(yolo_dir, 'custom',  source='local', path=os.path.join(weights_dir, "yolov9-e.pt"), force_reload = True)
model.eval()

image = cv2.imread(os.path.join(cwd, "data", "d3.jpg"))

# Convert BGR image to RGB image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, None, fx=0.3, fy=0.3) 
# image = cv2.resize(image, (1280, 1280)) 

# Define a transform to convert the image to torch tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Convert the image to Torch tensor
tensor = transform(image)

tensor = tensor[None, ...]

output = model(tensor)  # PyTorch expects input images to have four dimensions: batch, channels, height and width. You need to add a singleton "batch" dimension:

detections = output[0] # batch size 1
conf_thres = 0.25  # Confidence threshold
iou_thres = 0.45   # IoU threshold
results = non_max_suppression(detections, conf_thres, iou_thres)

results = results[0]  # Get results for the first (and only) image

# Convert to numpy array
results = results.cpu().numpy()

print(results)
print(model.names)
# print(list(map(lambda p: model.names[p], class_preds)))

def draw_detections(image, results, class_names):
    for x1, y1, x2, y2, conf, cls in results:
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw label
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)