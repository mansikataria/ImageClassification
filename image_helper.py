import os
import matplotlib.pyplot as plt
import torch

def imageshow(data, title=None):
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")