import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load your data
'''combined_data = pd.read_csv('train.csv')

# Preprocess your data, including creating the 'pre' column
combined_data['pre'] = combined_data.apply(lambda row: ', '.join([row['pre requisite'], row['concept'], row['pre requisite taxonomy'], row['concept taxonomy']]), axis=1)
combined_data = combined_data[['label', 'pre']]'''

# Split the data into train and test sets
train_data, test_data = train_test_split(combined_data, test_size=0.1, random_state=42, stratify=combined_data['label'])

# Define a custom dataset class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['pre']
        label = self.data.iloc[idx]['label']

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }

# Set hyperparameters
batch_size = 64
max_length = 200
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = TextDataset(train_data, tokenizer, max_length)
test_dataset = TextDataset(test_data, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the BERT-BiGRU model
class BERTBiGRUClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes):
        super(BERTBiGRUClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.gru = nn.GRU(input_size=self.bert.config.hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        gru_out, _ = self.gru(hidden_states)
        logits = self.fc(gru_out[:, -1, :])
        return logits

# Set hyperparameters for training
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERTBiGRUClassifier('bert-base-uncased', hidden_size=64, num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}')

# Evaluation
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().numpy()

        outputs = model(input_ids, attention_mask)
        predictions.extend(torch.sigmoid(outputs).cpu().numpy())
        true_labels.extend(labels)

predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

# Calculate and print the classification report and confusion matrix
print(classification_report(true_labels, predictions,digits=3))
print(confusion_matrix(true_labels, predictions))