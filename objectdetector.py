from torch import nn

class ObjectDetector(nn.Module):
  def __init__(self, baseModel, numClasses):
    super(ObjectDetector, self).__init__()
    self.baseModel = baseModel
    self.numClasses = numClasses

    # regressor layer
    self.regressor = nn.Sequential(
      nn.Linear(baseModel.fc.in_features, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 4),
      nn.Sigmoid()
    )

    # classification layer
    self.classifier = nn.Sequential(
      nn.Linear(baseModel.fc.in_features, 512),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      nn.Linear(512, self.numClasses)
    )

    # The final step is to make the base model’s fully connected layer into an
    # Identity layer, which means it’ll mirror the outputs produced by the
    # convolution block right before it
    self.baseModel.fc = nn.Identity()

  def forward(self, x):
    features = self.baseModel(x)
    bbox = self.regressor(features)
    classLogits = self.classifier(features)
    return bbox, classLogits
