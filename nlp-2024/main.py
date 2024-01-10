!wget http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
!unzip bbc-fulltext.zip


DocsInMemory = CustomTextDataset('/content/bbc')
print(DocsInMemory.labels)
print(DocsInMemory.length)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(DocsInMemory.texts,DocsInMemory.labels)



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



tensor = torch.tensor( x_train_vectorized.toarray(), dtype=torch.float32)
tensor_test = torch.tensor( x_test_vectorized.toarray(), dtype=torch.float32)

device = (
     "cpu"
)


model = NeuralNetwork().to(device)


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



from sklearn.metrics import accuracy_score

model.eval()
with torch.no_grad():
    predictions = model(tensor_test)

softmax = torch.nn.Softmax(dim=1)
probs = softmax(predictions)
predicted_labels = torch.argmax(probs, dim=1)

accuracy = accuracy_score(tensor_labels_test, predicted_labels)
print(f'Accuracy: {accuracy:.5f}')