import io
import pdb
import pickle
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import tensor
from torch.utils.data import DataLoader, Dataset

from sklearn.decomposition import PCA
import subprocess
from collections import Counter

import matplotlib.pyplot as plt

class WordVecDimReductionModel(nn.Module):
    def __init__(self,intput_dimension=300,output_dimension=100):
        super(WordVecDimReductionModel, self).__init__()
        hidden_dimension = 200
        # self.layers = nn.Sequential(nn.Linear(intput_dimension,hidden_dimension), nn.Sigmoid(), nn.Linear(hidden_dimension, output_dimension))
        self.layers = nn.Sequential(nn.Linear(intput_dimension, output_dimension))
    def forward(self, input):
        return self.layers(input)
        
class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size, word_vec_dimension, num_labels, embedding_layer, dropout_p, window_sizes=(3, 4, 5)):
        super(CnnTextClassifier, self).__init__()
        num_filters = 128
        # self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding = embedding_layer

        layer_list = []
        for window_size in window_sizes:
            layer_list.append(nn.Conv2d(1, num_filters, [window_size, word_vec_dimension], padding=(window_size - 1, 0)))
            # layer_list.append(nn.Dropout2d(dropout_p))
        self.convs = nn.ModuleList(layer_list)
        self.fc = nn.Linear(num_filters * len(window_sizes), num_labels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        input = self.embedding(input)          

        input = torch.unsqueeze(input, 1)       
        conv_out_list = []
        for conv in self.convs:
            x = F.relu(conv(input))        
            x = torch.squeeze(x, -1)  
            x = F.max_pool1d(x, x.size(2))  
            conv_out_list.append(x)
        conv_output = torch.cat(conv_out_list, 2)            

        conv_output = conv_output.view(conv_output.size(0), -1)       
        conv_output = self.fc(conv_output)             
        logits = self.dropout(conv_output)
        return logits

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    words = []
    vectors = []
    for line in tqdm.tqdm(fin,desc="Fasttext loading",total=n):
        tokens = line.rstrip().split(' ')
        vector = list(map(float, tokens[1:]))
        data[tokens[0]] = vector
        words.append(tokens[0])
        vectors.append(vector)
    mean_word_vec = list(np.mean(np.array(vectors),axis=0))
    return data, words, vectors, mean_word_vec

def get_word_vector_matrix(word2index, fasttext, word_vec_dimension, mean_word_vec):
    vocab_size = len(word2index)
    vocab_word_vectors = np.zeros((vocab_size, word_vec_dimension))
    words_found = 0
    for word, vec in fasttext.items():
        list(vec)
    for word,i in tqdm.tqdm(word2index.items(), total=vocab_size, desc="Creating vocab word vectors"):
        try: 
            vocab_word_vectors[i] = list(fasttext[word])
            words_found += 1
        except KeyError:
            vocab_word_vectors[i] = np.random.normal(scale=0.6, size=(word_vec_dimension, ))
            # vocab_word_vectors[i] = mean_word_vec
    with open("vocab_word_vectors_100d.pickle","wb") as f:
        pickle.dump(vocab_word_vectors,f)
    return vocab_word_vectors

def create_emb_layer(vocab_word_vectors, non_trainable=False):
    vocab_word_vectors = tensor(vocab_word_vectors)
    num_embeddings, embedding_dim = vocab_word_vectors.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': vocab_word_vectors})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

def plot(dev_set_labels, dev_set_sent_len):
    sent_len = [x[0] for x in sorted(dev_set_sent_len.items(),key=lambda x:x[0])]
    dev_set_sent_len = [x[1] for x in sorted(dev_set_sent_len.items(),key=lambda x:x[0])]
    plt.plot(sent_len, dev_set_sent_len,'g-')
    plt.xlabel("Validation set sentence length")
    plt.ylabel("Frequency")
    # plt.legend(loc="upper right")
    # plt.title("Loss values wrt epochs")
    plt.savefig("dev_set_sent_len.png")
    plt.close()

    sent_labels = [x[0] for x in sorted(dev_set_labels.items(),key=lambda x:x[0])]
    dev_set_labels = [x[1] for x in sorted(dev_set_labels.items(),key=lambda x:x[0])]
    plt.plot(sent_labels, dev_set_labels,'b--')
    plt.xlabel("Validation set sentence labels")
    plt.ylabel("Frequency")
    # plt.legend(loc="upper right")
    # plt.title("Loss values wrt epochs")
    plt.savefig("dev_set_sent_labels.png")
    plt.close()

def preprocess(file_type = "train"):

    dev_set_labels = Counter()
    dev_set_sent_len = Counter()

    data = open("topicclass/topicclass_"+str(file_type)+".txt","r",encoding="utf-8")
    all_labels = []
    all_text = []
    for line in data:
        label, text = line.split("|||")
        label = label.lower().strip()
        if "darama" in label:
            label = label.replace("darama","drama")
        if label not in label2index:
            label2index[label] = len(label2index)
        all_labels.append(label2index[label])
        text = text.split()
        # '''
        if file_type == "valid":
            dev_set_labels[label2index[label]] += 1
            dev_set_sent_len[len(text)] += 1
        # '''
        for index,word in enumerate(text):
            word = word.lower().strip()
            if file_type == "train":
                if word not in word2index:
                    word2index[word] = len(word2index)
                text[index] = word2index[word]
            else: #dev, test
                if word not in word2index:
                    text[index] = word2index["UNK"] #handling unknown words
                else:
                    text[index] = word2index[word]
        all_text.append(text)
    if file_type == "valid":
        plot(dev_set_labels, dev_set_sent_len)
        sys.exit(0)
    return all_labels, all_text

class CustomDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = np.array(labels)
        self.text = np.array(text)
    
    def __getitem__(self, index):
        return self.text[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

def collate_func(batch):
    max_sent_len = 0
    for text, _ in batch:
        max_sent_len = max(max_sent_len, len(text))
    batch_padded_text = []
    batch_labels = []
    word_vec_dimension = len(batch[0][0])
    for text, label in batch:
        padded_text = text + [0]*(max_sent_len-len(text))
        try:
            padded_text = np.array(padded_text).flatten()
        except:
            pdb.set_trace()
        batch_padded_text.append(padded_text)
        batch_labels.append(label)
    batch_padded_text = tensor(batch_padded_text)
    batch_labels = tensor(batch_labels)
    return batch_padded_text, batch_labels

def save_checkpoint(state, is_best_loss,is_best_accuracy, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename,"model_best_loss.pth.tar")
    if is_best_accuracy:
        shutil.copyfile(filename,"model_best_accuracy.pth.tar")

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.xavier_normal_(m.bias)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.xavier_normal_(m.bias)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":
    
    word2index = {"UNK":0}
    label2index = {}

    train_labels, train_text = preprocess("train")
    dev_labels, dev_text = preprocess("valid")
    test_labels, test_text = preprocess("test")

    assert len(train_labels)==len(train_text)
    assert (set(dev_labels) & set(train_labels)) == set(dev_labels)

    epochs = 1
    learning_rate = 0.001
    weight_decay = 0.00001
    output_size = len(set(train_labels))
    batch_size = 50
    dropout_p = 0.5

    num_labels = len(set(train_labels))
    vocab_size = len(word2index)
    word_vec_dimension = 300 #Ensure this is same dimension as pretrained vector length
    load_pretrained_vectors = True
    if load_pretrained_vectors:
        # fasttext, words, vectors, mean_word_vec = load_vectors("crawl-300d-2M.vec")
        # fasttext = word_vec_dim_reduction(fasttext, words, vectors, word_vec_dimension)
        # word_vector_matrix = get_word_vector_matrix(word2index, fasttext, word_vec_dimension, mean_word_vec)
        with open("vocab_word_vectors.pickle","rb") as f:
            word_vector_matrix = pickle.load(f)
        embedding_layer = create_emb_layer(word_vector_matrix, non_trainable=True)
    else:
        embedding_layer = torch.nn.Embedding(vocab_size, word_vec_dimension)

    #Sorting on sentence length
    # train_text = sorted(train_text,key = lambda x:len(x))

    train_dataset = CustomDataset(train_text, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func, num_workers=4, pin_memory=True)

    dev_dataset = CustomDataset(dev_text, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, collate_fn=collate_func)#, num_workers=4, pin_memory=True)

    test_dataset = CustomDataset(test_text, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_func)#, num_workers=4, pin_memory=True)

    num_train_batches = int(len(train_labels)/float(batch_size))
    num_dev_batches = int(len(dev_labels)/float(batch_size))
    num_test_batches = int(len(test_labels)/float(batch_size))

    model = CnnTextClassifier(vocab_size, word_vec_dimension, num_labels, embedding_layer, dropout_p)
    model.apply(init_weights)
    if torch.cuda.is_available():
        model = model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    # optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optim = torch.optim.SGD(model.parameters(), lr=0.01,weight_decay=0.000001,momentum=0.9,nesterov=True)

    print("\nTraining now\n")

    best_dev_accuracy = 0
    best_dev_loss = float('inf')

    epoch_dev_accuracies = []
    epoch_dev_loss = []
    epoch_dev_misclassified_label = Counter()
    epoch_dev_misclassified_sent_len = Counter()

    '''
    with open("proper_model_dev.pickle","rb") as f:
        epoch_dev_accuracies = pickle.load(f)
        epoch_dev_loss = pickle.load(f)
        epoch_dev_misclassified_label = pickle.load(f)
        epoch_dev_misclassified_sent_len = pickle.load(f)
    '''

    for epoch in tqdm.trange(epochs, ncols=100, desc="Epoch"):
        epoch_start_time = time.time()
        losses = []
        total = 0
        correct =0

        '''
        #Dividing learning rate by 10 every epoch_reduction_interval epochs
        epoch_reduction_interval = 5
        epoch_reduction_percent = 0.05
        if epoch%epoch_reduction_interval == 0:
            learn_rate = learn_rate * (1-epoch_reduction_percent)
            for param_group in optim.param_groups:
                param_group['lr'] = learn_rate
        '''
        
        model.train()

        for batch_num, data in enumerate(tqdm.tqdm(train_dataloader, total=num_train_batches, ncols=100, desc="Training")):
            text,label = data
            text, label = text.long(), label.long()
            if torch.cuda.is_available():
                text, label = text.cuda(), label.cuda()
            optim.zero_grad()  # Reset the gradients

            prediction = model(text)  # Feed forward
            loss = loss_fn(prediction, label)  # Compute losses
            loss.backward()  # Backpropagate the gradients
            losses.append(loss.cpu().data.numpy())
            
            optim.step()  # Update the network

            _,predicted = torch.max(prediction.data, 1)
            total += label.size(0)
            correct += (predicted.cpu().long() == data[1]).sum()

            # batch_accuracy = (predicted.cpu().long() == data[1]).sum().cpu().item() / label.size(0)
            # mean_loss = torch.mean(loss)
            # if batch_num % 500 == 0:
            #     tqdm.tqdm.write("Train Batch: " + str(batch_num) +" Loss: "+str(mean_loss.cpu().item())[:5]+" Accuracy: "+str(batch_accuracy))
                # sys.stdout.write("\nTrain batch Loss:"+str(mean_loss.cpu().item())[:5]+" Accuracy:"+str(batch_accuracy))
                # sys.stdout.write('\x1b[2K')
                # sys.stdout.write('\x1b[1A')
            # sys.stdout.flush()

        epoch_end_time = time.time()
        accuracy = (100.0 * correct)/float(total)
        sys.stdout.write("\nTrain Epoch {} Loss: {:.4f} Time {} Accuracy {}".format(epoch, np.asscalar(np.mean(losses)),epoch_end_time-epoch_start_time,accuracy))
        sys.stdout.flush()

        losses = []
        total = 0
        correct = 0
        dev_start_time = time.time()

        model.eval()

        for batch_num, data in enumerate(tqdm.tqdm(dev_dataloader, total=num_dev_batches, ncols=100, desc="Validation")):
            text,label = data
            text, label = text.long(), label.long()
            if torch.cuda.is_available():
                text, label = text.cuda(), label.cuda()
            prediction = model(text)  # Feed forward
            loss = loss_fn(prediction, label)
            losses.append(loss.cpu().data.numpy())

            _,predicted = torch.max(prediction.data, 1)
            total += label.size(0)
            correct += (predicted.cpu().long() == data[1]).sum()
            if predicted.cpu().long() != data[1]:
                epoch_dev_misclassified_label[data[1].cpu().item()] += 1
                epoch_dev_misclassified_sent_len[len(text[0])] += 1

            # batch_accuracy = (predicted.cpu().long() == data[1]).sum().cpu().item() / label.size(0)
            # mean_loss = torch.mean(loss)
            # if batch_num % 500 == 0:
            #     tqdm.tqdm.write("Dev Batch: " + str(batch_num) +" Loss: "+str(mean_loss.cpu().item())[:5]+" Accuracy: "+str(batch_accuracy))
                # sys.stdout.write("\nDev batch Loss:"+str(mean_loss.cpu().item())[:5]+" Accuracy:"+str(batch_accuracy))
                # sys.stdout.write('\x1b[2K')
                # sys.stdout.write('\x1b[1A')
            # sys.stdout.flush()

        dev_end_time = time.time()
        dev_accuracy = (100.0 * correct)/float(total)
        dev_loss = np.asscalar(np.mean(losses))
        epoch_dev_loss.append(dev_loss)
        epoch_dev_accuracies.append(dev_accuracy)
        sys.stdout.write("\nDev Epoch {} Loss {} Time {} Accuracy {}".format(epoch, dev_loss,dev_end_time-dev_start_time,dev_accuracy))
        sys.stdout.flush()

        is_best_loss = False
        is_best_accuracy = False
        if dev_loss<best_dev_loss:
            best_dev_loss = dev_loss
            print("\n Best Dev Loss ",best_dev_loss)
            is_best_loss = True
        if dev_accuracy>best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            print("\n Best Dev Accuracy ",best_dev_accuracy.cpu().item())
            is_best_accuracy = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_dev_accuracy,
            'optimizer' : optim.state_dict(),
        }, is_best_loss,is_best_accuracy)

    for saved_model in ["model_best_accuracy.pth.tar","model_best_loss.pth.tar"]:
        torch_dict = torch.load(saved_model)
        model.load_state_dict(torch_dict["state_dict"])
        model.eval()
        fout = open("predictions_"+str("_".join(saved_model.split(".")[0].split("_")[1:]))+".csv",'w')
        fout.write("id,label\n")
        batch_num = 0
        for batch in tqdm.tqdm(test_dataloader, total=num_test_batches, ncols=100, desc="Test "):
        # for batch_num, data in enumerate(test_data_loader):
            text,_ = batch
            text = text.long()
            if torch.cuda.is_available():
                text = text.cuda()
            prediction = model(text)  # Feed forward
            _,predicted = torch.max(prediction.data, 1)
            # predicted_labels.extend(list(predicted.long()))
            fout.write(str(batch_num)+","+str(predicted.cpu().long().item())+"\n")
            batch_num += 1
        fout.close()
   
    with open("proper_model_dev.pickle","wb") as f:
        pickle.dump(epoch_dev_accuracies,f)
        pickle.dump(epoch_dev_loss,f)
        pickle.dump(epoch_dev_misclassified_label,f)
        pickle.dump(epoch_dev_misclassified_sent_len,f)

    with open("proper_model_dev.pickle","rb") as f:
        epoch_dev_accuracies = pickle.load(f)
        epoch_dev_loss = pickle.load(f)
        epoch_dev_misclassified_label = pickle.load(f)
        epoch_dev_misclassified_sent_len = pickle.load(f)

    plt.plot(range(epochs), epoch_dev_loss,'b-',label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.legend(loc="upper right")
    # plt.title("Loss values wrt epochs")
    plt.savefig("loss.png")
    plt.close()

    plt.plot(range(epochs), epoch_dev_accuracies,'r-',label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    # plt.title("Loss values wrt epochs")
    plt.savefig("accuracy.png")
    plt.close()
    
    sent_len_min = min(epoch_dev_misclassified_sent_len.keys())
    sent_len_max = max(epoch_dev_misclassified_sent_len.keys())
    sent_len = [x[0] for x in sorted(epoch_dev_misclassified_sent_len.items(),key=lambda x:x[0])]
    epoch_dev_misclassified_sent_len = [x[1] for x in sorted(epoch_dev_misclassified_sent_len.items(),key=lambda x:x[0])]
    plt.plot(sent_len, epoch_dev_misclassified_sent_len,'g-')
    plt.xlabel("Sentence length")
    plt.ylabel("Frequency")
    # plt.legend(loc="upper right")
    # plt.title("Loss values wrt epochs")
    plt.savefig("proper_miss_sent_len.png")
    plt.close()

    index2labels = dict(zip(list(label2index.values()),list(label2index.keys())))
    sent_labels = [x[0] for x in sorted(epoch_dev_misclassified_label.items(),key=lambda x:x[0])]
    epoch_dev_misclassified_label = [x[1] for x in sorted(epoch_dev_misclassified_label.items(),key=lambda x:x[0])]
    plt.plot(sent_labels, epoch_dev_misclassified_label,'b--')
    plt.xlabel("Sentence labels")
    plt.ylabel("Frequency")
    # plt.legend(loc="upper right")
    # plt.title("Loss values wrt epochs")
    plt.savefig("proper_miss_labels.png")
    plt.close()
