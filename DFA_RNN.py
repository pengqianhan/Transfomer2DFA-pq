from Training_Functions import make_train_set_for_target,mixed_curriculum_train
from Tomita_Grammars import *
from Extraction import extract

from transformers import Trainer, TrainingArguments
from transformers import BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import trange

import torch
import torch.nn as nn
import transformers
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import BertConfig, BertForSequenceClassification, DistilBertForSequenceClassification, DistilBertConfig, AdamW
alphabet = "01"
tokenize = ord
vocab_size = 128
distil_model_config = DistilBertConfig(
    vocab_size=vocab_size,   
    hidden_dim=32,
    dim=32,
    n_heads=4
    # note: not updated
)
train_arg = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=200,              
    per_device_train_batch_size=1,   
    per_device_eval_batch_size=1,    
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,               
    learning_rate=5e-4,   
    logging_strategy='epoch',
    evaluation_strategy='epoch'
)
model_config = BertConfig(
    vocab_size=vocab_size, 
    hidden_size=32,  
    num_hidden_layers=2,  
    num_attention_heads=2,  
    intermediate_size=32,  
    num_labels=2,  
    pad_token_id=127 #anything unused !!
)

distil_model = DistilBertForSequenceClassification(distil_model_config)
model = BertForSequenceClassification(model_config)
target = tomita_4

class NumberDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        string, label = self.data[idx]
        binary_list = [tokenize(bit) for bit in string]
        return {
            "input_ids": torch.tensor(binary_list, dtype=torch.long), 
            "labels": torch.tensor(label, dtype=torch.long)
        }
    
class HiddenStateDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = rnn
        self.pre_classifier = nn.Linear(32, 32)
        self.classifier = nn.Linear(32, 2)
    def forward(self, x):
        x = self.embedding(x)
        _, x = self.rnn(x)
        x = self.pre_classifier(x)
        x = self.classifier(x)
        return x
    
class TransformerNetwork:
    def __init__(self, model, train_arg, vocab_size, alphabet, rnn_arg, target):
        """
        :param model: Transformer model
        :param train_arg: Training arguments for Transformer model
        :param alphabet: list of characters in the alphabet from 0 to 9. E.g. "01"
        :param rnn_arg: Arguments for RNN model, a dictionary
        :param target: Target language, a function
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rnn_train_data = None
        self.rnn_test_data = None
        self.target = target
        self.alphabet = alphabet
        self.model = model.to(self.device)
        self.train_arg = train_arg
        self.rnn_arg = rnn_arg
        # distil_model_config is ill implemented here
        rnn = nn.RNN(input_size=distil_model_config.dim, hidden_size=distil_model_config.dim, batch_first=True)
        self.rnn = RNNClassifier(vocab_size,distil_model_config.dim,rnn).to(self.device)
        self.data = list(make_train_set_for_target(self.target, alphabet).items())[1:]
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.train_dataset = NumberDataset(self.train_data)
        self.test_dataset = NumberDataset(self.test_data)
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        
    def classify_word(self, word):
        """
        :param word: a string of 0s and 1s
        :return: 1 if the word is in the target language, 0 otherwise
        """
        if len(word) == 0:
            return self.classify_state(torch.tensor([0 for _ in range(model_config.hidden_size)], dtype=torch.float32)).item()
        input = torch.tensor([tokenize(bit) for bit in word], dtype=torch.long).to(self.device)
        with torch.no_grad():
            return self.rnn(input).argmax().item()
    def model_classify_word(self, word):
        """
        :param word: a string of 0s and 1s
        :return: 1 if the word is in the target language, 0 otherwise
        """
        if len(word) == 0:
            return self.target('')
        input = torch.tensor([tokenize(bit) for bit in word], dtype=torch.long).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(input).logits.argmax().item()
    def train_transformer(self):
        """
        Train the transformer model
        """
        self.trainer = Trainer(
            model=self.model,
            args=self.train_arg,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
        )
        self.trainer.train()
        self.model.eval()
        correct = 0
        total = 0
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        for x in test_loader:
            input = x['input_ids'].to(self.device)
            label = x['labels']
            correct += self.model(input).logits.argmax() == label[0]
            total += 1
        print("Correct:" ,correct.item())
        print("Total:", total)
        self.trainer.evaluate()
    def train_rnn(self):
        """
        Train the RNN model, embedding is given by extract_embedding
        """
        # First load linear model from transformer
        def init_identity(m):
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize classifier in RNN
        assert (hasattr(transformer.model,'classifier')), 'Transformer has no classifier layer'
        self.rnn.classifier.load_state_dict(self.model.classifier.state_dict())
        if hasattr(transformer.model,'pre_classifier'):
            self.rnn.pre_classifier.load_state_dict(self.model.pre_classifier.state_dict())
        else:
            self.rnn.pre_classifier.apply(init_identity)
               
        # Make the training data
        self.rnn_train_data = []
        with torch.no_grad():
            for x in self.train_loader:
                input = x['input_ids'].to(self.device).squeeze(0)
                label = self.model(input.unsqueeze(0)).logits.argmax()
                self.rnn_train_data.append((input, label))
        self.rnn_train_data, self.rnn_test_data = train_test_split(self.rnn_train_data, test_size=0.2, random_state=42)
        rnn_train_dataset = HiddenStateDataset(self.rnn_train_data)
        rnn_test_dataset = HiddenStateDataset(self.rnn_test_data)
        rnn_train_loader = DataLoader(rnn_train_dataset, batch_size=1, shuffle=True)
        rnn_test_loader = DataLoader(rnn_test_dataset, batch_size=1, shuffle=False)
        # Train the self.rnn model
        self.rnn = self.rnn.to(self.device)
        optimizer = AdamW([{'params': self.rnn.embedding.parameters()},
                          {'params': self.rnn.rnn.parameters()}]
                          , lr=1e-3)
        for epoch in trange(200):
            for x in rnn_train_loader:
                input, label = x
                output = self.rnn(input)
                output = output.squeeze(0)
                loss = nn.CrossEntropyLoss()(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            for x in self.train_loader:
                # todo: add support for other bert
                bert_output = self.model.bert(x['input_ids'].to(transformer.device)).pooler_output
                input = x['input_ids'].to(self.device).squeeze(0)
                input = self.rnn.embedding(input)
                output = self.rnn.rnn(input)[1]
                output.reshape(bert_output.shape)
                loss_func = nn.L1Loss(reduction = 'mean')
                loss = loss_func(bert_output,output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Test the self.rnn model
        self.rnn.eval()
        correct = 0
        total = 0
        for x in rnn_test_loader:
            input, label = x
            output = self.rnn(input)
            correct += output.argmax() == label
            total += 1
        print("Correct:" ,correct.item())
        print("Total:", total)
    def classify_state(self,state):
        output = self.rnn.pre_classifier(state.to(self.device))
        output = self.rnn.classifier(output)
        # return the label
        return output.argmax()
    def get_first_RState(self):
        state = [0 for _ in range(model_config.hidden_size)]
        #print("0",state, self.classify_state(torch.tensor(state, dtype=torch.float32)))
        return state, self.target('') #self.classify_state(torch.tensor(state, dtype=torch.float32)).item()
    def get_next_RState(self, state, char):
        input = self.rnn.embedding.state_dict()['weight'][tokenize(char)]
        input = input.unsqueeze(0).unsqueeze(0)
        input = input.to(self.device)
        state = torch.tensor(state, dtype=torch.float32).view(1, 1, 32)
        state = state.to(self.device)
        output, next_state = self.rnn.rnn(input, state)
        return next_state.squeeze().tolist(), self.classify_state(next_state.squeeze()).item()
    

model = BertForSequenceClassification(model_config).from_pretrained('./results/checkpoint-550000')
transformer = TransformerNetwork(model, train_arg, vocab_size, alphabet, model_config, target)
#transformer.train_transformer()
transformer.model.eval()