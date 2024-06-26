

!wget http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
!unzip bbc-fulltext.zip

import os
import torch
from torch.utils.data import Dataset


class CustomTextDataset(Dataset):
    def __init__(self,root_dir,transform=None):
      self.root_dir = root_dir
      self.transform = transform


      self.data = self.create_dataset()
      self.length = self. __len__()
      self.docs = self.__getdocs__()
      self.texts = self.__getdocsText__()
      self.labels = self.__getdocsLabels__()

    def create_dataset(self):
      data = []
      for label in os.listdir(self.root_dir):             #listdir() returns a list containing the names of the entries in the directory given by path
          label_path = os.path.join(self.root_dir, label) #os.path.join() concatenate various path components with '/', apart from the last path component
          if os.path.isdir(label_path):                   #os.path.isdir() method in Python is used to check whether the specified path is an existing directory or not
            for filename in os.listdir(label_path):
              if filename.endswith(".txt"):
                 data.append((os.path.join(label_path, filename), label))
      return data

    def __len__(self):
      return len(self.data)

    def __getitem__(self,idx):
        txt_path, label = self.data[idx]

        with open(txt_path, 'r', encoding='utf-8') as f:
           content = f.read()

        if self.transform:
          content = self.transform(content)
        return content, label

    def __getdocs__(self):
       docs_text = []
       docs_labels = []

       for i in range(self.length):
        txt_path, label = self.data[i]

        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if self.transform:
            content = self.transform(content)

        docs_text.append(content)
        docs_labels.append(label)

       return docs_text, docs_labels


    def __getdocsText__(self):
      DocsText= []

      for i in range(self.length):
          txt_path, label = self.data[i]

          with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()

            DocsText.append(content)


      return DocsText


    def __getdocsLabels__(self):
      DocsLabels= []

      for i in range(self.length):
          txt_path, label = self.data[i]

          DocsLabels.append(label)

      return DocsLabels

DocsInMemory = CustomTextDataset('/content/bbc')
print(DocsInMemory.labels)
print(DocsInMemory.length)

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(DocsInMemory.texts,DocsInMemory.labels)

print(y_train)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english",max_features = 1000, decode_error = "ignore",use_idf=True)
vectorizer.fit(x_train)
x_train_vectorized = vectorizer.transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)
vectorizer.get_feature_names_out()

import torch
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train)
encoded_labels_test = label_encoder.fit_transform(y_test)


tensor_labels = torch.tensor(encoded_labels)
tensor_labels_test = torch.tensor(encoded_labels_test)


print("Original Class Labels:", y_train)
print("Encoded Tensor Labels:", tensor_labels)
print("Encoded Tensor Labels test:", tensor_labels_test)

print(x_train_vectorized)

tensor = torch.tensor( x_train_vectorized.toarray(), dtype=torch.float32)
tensor_test = torch.tensor( x_test_vectorized.toarray(), dtype=torch.float32)
print(tensor)
print(tensor.shape)

device = (
     "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(tensor.shape[1], 512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,5),
    )

  def forward(self,x):
   logits = self.linear_relu_stack(x)
   return logits

model = NeuralNetwork().to(device)
print(model)

logits = model(tensor)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

import torch.optim as optim
import torch.nn.functional as F


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

num_epochs = 2
for epoch in range(num_epochs):
  for inputs in tensor:
    optimizer.zero_grad()

    outputs = model(tensor)
    loss = criterion(outputs, tensor_labels)

    loss.backward()
    optimizer.step()

  print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

from sklearn.metrics import accuracy_score

model.eval()
with torch.no_grad():
    predictions = model(tensor_test)

softmax = torch.nn.Softmax(dim=1)
probs = softmax(predictions)
predicted_labels = torch.argmax(probs, dim=1)

accuracy = accuracy_score(tensor_labels_test, predicted_labels)
print(f'Accuracy: {accuracy:.5f}')
