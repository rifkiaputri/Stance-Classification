import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
from torch.autograd import Variable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

feature_size = 5000
input_vec_size = (feature_size * 2 + 1) + 8 + 44
hidden_size = 100
batch_size = 500
drop_out = 0.4
num_class = 4

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_vec_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = x.view(-1, input_vec_size)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.dropout(x, p = drop_out, training=self.training)
        return self.l2(x)

def runNN():
    from utils.dataset import DataSet
    import utils.score as sc

    import sentiment_feature as s_f
    import count_feature as c_f
    import tfidf_feature as t_f

    #make train, competition_test data prepared
    train_dataset = DataSet(name = 'train', path = './')
    test_dataset = DataSet(name = 'competition_test', path = './')
    
    train_x = []
    train_y = []
    for idx in range(0, len(train_dataset.stances)):
        train_x.append({'headline':train_dataset.stances[idx]['Headline'], 'article':train_dataset.articles[train_dataset.stances[idx]['Body ID']]})
        train_y.append(train_dataset.stances[idx]['Stance'])
    
    test_x = []
    test_y = []
    for idx in range(0, len(test_dataset.stances)):
        test_x.append({'headline':test_dataset.stances[idx]['Headline'], 'article':test_dataset.articles[test_dataset.stances[idx]['Body ID']]})
        test_y.append(test_dataset.stances[idx]['Stance'])

    def get_heads_bodies_from_x(x):
        heads = []
        bodies = []
        for idx in range(0, len(x)):
            heads.append(x[idx]['headline'])
            bodies.append(x[idx]['article'])
        return heads, bodies
    
    train_heads, train_bodies = get_heads_bodies_from_x(train_x)
    test_heads, test_bodies = get_heads_bodies_from_x(test_x)

    label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
    train_y_one_hot = []
    for ins in train_y:
        train_y_one_hot.append(label_ref[ins])
    test_y_one_hot = []
    for ins in test_y:
        test_y_one_hot.append(label_ref[ins])

    #generate features for "train" dataset
    print('Generate features from train dataset')
    if not t_f.check_tfidf_feature_exist('train'):
        print('generate and load tfidf feature')
        tfidf_vectorizer = t_f.tfidfVectorizer_init(train_heads, train_bodies, test_heads, test_bodies)
        t_f.tfidf_feature_generate('train', tfidf_vectorizer, train_x)
        train_tfidf_feature = t_f.tfidf_feature_read('train')
    else:
        print('load tfidf feature')
        train_tfidf_feature = t_f.tfidf_feature_read('train')

    
    if not s_f.check_sentiment_feature_exist('train'):
        print('generate and load sentiment feature')
        train_sentiment_feature = s_f.sentiment_feature_generate('train', train_heads, train_bodies)
    else:
        print('load sentiment feature')
        train_sentiment_feature = s_f.sentiment_feature_read('train')
    
    
    if not c_f.check_count_feature_exist('train'):
        print('generate and load overlapping feature')
        train_count_feature = c_f.count_feature_generate('train', train_heads, train_bodies)
    else:
        print('load count feature')
        train_count_feature = c_f.count_feature_read('train')
    
    #generate features for "competition_test" dataset
    print('Generate features from competition test dataset')
    if not t_f.check_tfidf_feature_exist('test'):
        print('generate and load tfidf feature')
        tfidf_vectorizer = t_f.tfidfVectorizer_init(train_heads, train_bodies, test_heads, test_bodies)
        t_f.tfidf_feature_generate('train', tfidf_vectorizer, train_x)
        train_tfidf_feature = t_f.tfidf_feature_read('train')
    else:
        print('load tfidf feature')
        test_tfidf_feature = t_f.tfidf_feature_read('test')
    
    if not s_f.check_sentiment_feature_exist('test'):
        print('generate and load sentiment feature')
        test_sentiment_feature = s_f.sentiment_feature_generate('test', test_heads, test_bodies)
    else:
        print('load sentiment feature')
        test_sentiment_feature = s_f.sentiment_feature_read('test')
    
    
    if not c_f.check_count_feature_exist('test'):
        print('generate and load count feature')
        test_count_feature = c_f.count_feature_generate('test', test_heads, test_bodies)
    else:
        print('load count feature')
        test_count_feature = c_f.count_feature_read('test')
    
    #add codes to run nerual net
    #prepare 'train' dataset
    tensor_x_train_tfidf = torch.from_numpy(train_tfidf_feature)
    tensor_x_train_sentiment = torch.from_numpy(train_sentiment_feature)
    tensor_x_train_count = torch.from_numpy(train_count_feature)
    tensor_x = torch.cat((tensor_x_train_tfidf, tensor_x_train_sentiment, tensor_x_train_count), dim=1)
    #tensor_x = torch.cat((tensor_x_train_tfidf, tensor_x_train_sentiment), dim=1)
    tensor_y = torch.from_numpy(np.asarray(train_y_one_hot))
    train_dataset = data_utils.TensorDataset(tensor_x, tensor_y)
    #train_dataset = data_utils.TensorDataset(tensor_x_train_tfidf, tensor_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

    #prepare 'competition_test' dataset
    tensor_x_test_tfidf = torch.from_numpy(test_tfidf_feature)
    tensor_x_test_sentiment = torch.from_numpy(test_sentiment_feature)
    tensor_x_test_count = torch.from_numpy(test_count_feature)
    tensor_x_test = torch.cat((tensor_x_test_tfidf, tensor_x_test_sentiment, tensor_x_test_count), dim=1)
    #tensor_x_test = torch.cat((tensor_x_test_tfidf, tensor_x_test_sentiment), dim=1)
    tensor_y_test = torch.from_numpy(np.asarray(test_y_one_hot))
    test_dataset = data_utils.TensorDataset(tensor_x_test, tensor_y_test)
    #test_dataset = data_utils.TensorDataset(tensor_x_test_tfidf, tensor_y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data).float()
            target = Variable(target).type(torch.LongTensor)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data[0]))

    def test():
        pred = []

        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            #label_t = torch.tensor([train_dataset.get_label_id(l) for l in label], dtype=torch.long, device=device)
            data = Variable(data, volatile=True).float()
            target = Variable(target).type(torch.LongTensor)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).data[0]
            # get the index of the max
            tmp = output.data.max(1, keepdim=True)[1]
            pred = pred + tmp.cpu().numpy().tolist()
            #gold = gold + label_t.cpu().numpy().tolist()
            correct += tmp.eq(target.data.view_as(tmp)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

        return pred

    for epoch in range(1, 90):
        train(epoch)
    torch.save(model, './nlp_project/model.pt')
    pred = test()

    temp_pred = []
    for e in pred:
        temp_pred.append(e[0])
    sc.report_score([sc.LABELS[e] for e in test_y_one_hot], [sc.LABELS[e] for e in temp_pred])
    return pred

if __name__ == '__main__':
    pred = runNN()
