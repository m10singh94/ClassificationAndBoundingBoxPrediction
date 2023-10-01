"""
Contains plot_loss_curves, download_random_img,
"""

import matplotlib.pyplot as plt
import requests
import torch
import cv2
from timeit import default_timer as timer


def plot_loss_curves(results):

  train_loss = results["train_loss"]
  test_loss = results["test_loss"]

  train_accuracy = results["train_acc"]
  test_accuracy = results["test_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))

  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_loss, label="train_loss", color='b')
  plt.plot(epochs, test_loss, label="test_loss", color='r')
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  # Plot accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_accuracy, label="train_accuracy", color='b')
  plt.plot(epochs, test_accuracy, label="test_accuracy", color='r')
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()

# ------------------------------------------------------------------------------

def download_random_img(data_path: str, image_link: str):
  """
  Takes in the image link and downloads it in the 'data_path'

  Args:
    image_link: link of teh image to be downloaded

  Returns:
    random_image_path: path of the downloaded image
  """

  random_image_path = data_path / "random_image.jpeg"
  if random_image_path.is_dir():
    print(f"{random_image_path} exists. Skipping Download")
  else:
    with open(random_image_path, "wb") as f:
      request = requests.get(image_link)
      print(f"Downloading image...")
      f.write(request.content)
      print(f"Image downloaded!")
  return random_image_path

# ------------------------------------------------------------------------------

def predict_random_img(data_path: str,
                      image_link: str,
                      model: torch.nn.Module,
                      class_names = None,
                      transform = None,
                      device: torch.device = None):
  """
  Downloads the image from the image link and infers the class of teh image
  with the probability of the prediction.

  Args:
    image_link: link of the image that you want to download ("image address")
    model: model with which prediction has to be done
    class_names: list of the possible categories for this classification
    transform: transformations needed for this image to be put inside the model
    device: "cuda" if available, otherwise "cpu"
  """
  # downloading the image
  image_filepath = download_random_img(data_path, image_link)
  print(f"random_image filepath: {image_filepath}")

  start_time = timer()
  # reading the image
  image = cv2.imread(str(image_filepath))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  h = image.shape[0]
  w = image.shape[1]

  image_resized = cv2.resize(image, (224, 224))
  image_resized = torch.tensor(image_resized)
  image_resized = image_resized.permute(2,0,1)
  image_transformed = transform(image_resized)

  # predicting the label and bbox
  model.to(device)
  model.eval()
  with torch.inference_mode():
    image_transformed = image_transformed.unsqueeze(dim=0)
    bbox, label_logits = model(image_transformed.to(device))

  image_pred_probs = torch.softmax(label_logits, dim=1)
  target_image_pred_label = torch.argmax(image_pred_probs, dim=1)

  end_time = timer()
  print(f"[INFO] Total predicting time: {end_time-start_time:.3f} seconds")

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

  plt.imshow(image)
