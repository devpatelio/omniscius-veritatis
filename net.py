
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeedForward(nn.Module):
    def __init__(self, num_classes, input_size, kernel_size=4):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc3 = nn.Linear(200, 100)        
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 20)
        self.fc7 = nn.Linear(20, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))

        
        return x


class RecurrentClassifier(nn.Module):
    def __init__(self, embedding_dim, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super(RecurrentClassifier, self).__init__()

        self.embedding = nn.Embedding(embedding_dim, input_size)
        self.rnn = nn.LSTM(input_size, 
                            hidden_size,
                            num_layers,
                            batch_first = True,
                            dropout=dropout)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x, (hidden, cell) = self.rnn(x)
        print(hidden.shape)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1, :, :]), dim=1))
        x = self.fc1(hidden)
        x = self.dropout(self.fc2(x))

        return x