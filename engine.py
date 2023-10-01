"""
Contains train_step, test_step and train function to train the model.
"""

from typing import Tuple, Dict, List
import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               class_loss_fn: torch.nn.Module,
               bbox_loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

  # putting model in train mode
  model.train()

  # initialising train loss and accuracy
  train_loss, train_acc = 0, 0

  # loop through data loader data batches
  for batch, (X, y, bboxes) in enumerate(dataloader): # X=images, y=labels
      # Send data to target device
      X, y, bboxes = X.to(device), y.to(device), bboxes.to(device)

      bbox_pred, y_pred_logits = model(X)
      bbox_loss = bbox_loss_fn(bbox_pred, bboxes)
      class_loss = class_loss_fn(y_pred_logits, y)
      total_loss = bbox_loss + class_loss
      train_loss += total_loss.item()

      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

      y_pred_class = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred_class)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

# ------------------------------------------------------------------------------

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              class_loss_fn: torch.nn.Module,
              bbox_loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

  # Putting model to eval mode
  model.eval()

  # initialising test loss and accuracy
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y, bboxes) in enumerate(dataloader):
        X, y, bboxes = X.to(device), y.to(device), bboxes.to(device)

        bbox_pred, test_pred_logits = model(X)
        bbox_loss = bbox_loss_fn(bbox_pred, bboxes)
        class_loss = class_loss_fn(test_pred_logits, y)
        total_loss = bbox_loss + class_loss
        test_loss += total_loss.item()

        test_pred_labels = test_pred_logits.argmax(dim=1)
        test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

# ------------------------------------------------------------------------------

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          class_loss_fn: torch.nn.Module,
          bbox_loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:

  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          class_loss_fn=class_loss_fn,
                                          bbox_loss_fn=bbox_loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
                                          dataloader=test_dataloader,
                                          class_loss_fn=class_loss_fn,
                                          bbox_loss_fn=bbox_loss_fn,
                                          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results
