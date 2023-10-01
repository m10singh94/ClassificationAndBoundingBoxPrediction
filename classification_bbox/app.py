"""
This file is the main file required by the Gradio Interface to run the demo.
"""

import gradio as gr
import os
import torch
from model import create_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["Cat", "Dog"]

# Create model
model = create_model(num_classes=len(class_names))

# Load saved weights
model.load_state_dict(
    torch.load(
        f="resnet_bbox_prediction_model.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

# Create predict function
def predict(image: torch.tensor,
            class_names,
            model: torch.nn.Module,
            transform):
  """Transforms and performs a prediction on img and returns prediction and time
  taken.

  Args:
    image: image tensor which is in the RGB format
    class_names: class names for labelling the prediction
    model: model to predict the label and bbox of the image
    transform: transform for the image before putting it into the model
  
  Returns:
    image: the annotated image
    bbox: bbox coordnaites
    time: prediction time in seconds
  """
  # getting height and width of the image
  h = image.shape[0]
  w = image.shape[1]

  # Start the timer
  start_time = timer()

  # Transform the target image and add a batch dimension
  image_resized = cv2.resize(image, (224, 224))
  image_resized = torch.tensor(image_resized)
  image_resized = image_resized.permute(2,0,1)
  image_transformed = transform(image_resized)

  # Put model into evaluation mode and turn on inference mode
  model.to("cpu")
  model.eval()
  with torch.inference_mode():
    image_transformed = image_transformed.unsqueeze(dim=0)
    bbox, label_logits = model(image_transformed)

  image_pred_probs = torch.softmax(label_logits, dim=1)
  target_image_pred_label = torch.argmax(image_pred_probs, dim=1)

  pred_time = round(timer() - start_time, 5)

  # Converting label code to label
  label = class_names[target_image_pred_label]
  title = f"{label} | {image_pred_probs.max()*100:.3f}%"

  # printing the image with the class, boundary box (bbox) and its probability
  (xmin, ymin, xmax, ymax) = bbox[0]
  startX = int(xmin * w)
  startY = int(ymin * h)
  endX = int(xmax * w)
  endY = int(ymax * h)

  font = cv2.FONT_HERSHEY_SIMPLEX
  color = (0,0,0)
  if h < 1000 and w < 1000:
    fontScale = 0.8
    thickness = 2
  else:
    fontScale = 5
    thickness = 12

  image = cv2.rectangle(image, (startX,startY), (endX,endY), (255,255,255), 2)
  image = cv2.rectangle(image,(startX,endY),(endX,endY+int(0.05*h)),(255,255,255),-1)
  image = cv2.putText(image,title,(startX+int(0.02*w),endY+int(0.045*h)),font,fontScale-0.2,color,thickness,cv2.LINE_AA)

  # plt.imshow(image)
  return image, (startX, startY, endX, endY), pred_time

### Gradio app ###

# Create title, description and article strings
title = "Cat Dog classification with Bounding Box"
description = "A resnet feature extractor computer vision model to classify images of cats and dogs with bounding box."
article = "Created at [GitHub repo](https://github.com/m10singh94/ClassificationAndBoundingBoxPrediction.git)."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"),
                    outputs=["image",
                             gr.Number(label="Prediction time (s)")],
                    # Create examples list from "examples/" directory
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
