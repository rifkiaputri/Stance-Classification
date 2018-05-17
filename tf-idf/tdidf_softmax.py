
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
import torch.utils.data as data_utils
from torch.autograd import Variable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]
headline_list = []
headline_body_id_list = []
body_article_list = []
body_art_id_list = []
train_y = []
train_x = []

headline_list_test = []
headline_body_id_list_test = []
body_article_list_test = []
body_art_id_list_test = []
test_y = []
test_x = []

feature_size = 5000
input_vec_size = (feature_size * 2) + 1
hidden_size = 100
batch_size = 500
drop_out = 0.4
num_class = 4


# In[3]:


tf = open('train_stances.csv', 'r', encoding='utf-8')
rdr = csv.reader(tf)
for (head,bodyid,stance) in rdr:
    if(head == 'Headline'):
        continue
    headline_list.append(head)
    headline_body_id_list.append(bodyid)
    train_y.append(stance)
tf.close() 

tf = open('train_bodies.csv', 'r', encoding='utf-8')
rdr = csv.reader(tf)
count = 0
for (bodyid,article) in rdr:
    if(bodyid == 'Body ID'):
        continue
    body_article_list.append(article)
    body_art_id_list.append(bodyid)
tf.close() 

for idx in range(0, len(headline_list)):
    train_x.append({'headline' : headline_list[idx], 'article' : body_article_list[body_art_id_list.index(headline_body_id_list[idx])]})


# In[51]:


tf = open('competition_test_stances.csv', 'r', encoding='utf-8')
rdr = csv.reader(tf)
for (head,bodyid,stance) in rdr:
    if(head == 'Headline'):
        continue
    headline_list_test.append(head)
    headline_body_id_list_test.append(bodyid)
    test_y.append(stance)
tf.close()

tf = open('competition_test_bodies.csv', 'r', encoding='utf-8')
rdr = csv.reader(tf)
count = 0
for (bodyid,article) in rdr:
    if(bodyid == 'Body ID'):
        continue
    body_article_list_test.append(article)
    body_art_id_list_test.append(bodyid)
tf.close() 

for idx in range(0, len(headline_list_test)):
    test_x.append({'headline' : headline_list_test[idx], 'article' : body_article_list_test[body_art_id_list_test.index(headline_body_id_list_test[idx])]})


# In[8]:


tfidf_vectorizer = TfidfVectorizer(max_features=feature_size, stop_words=stop_words).fit(headline_list + body_article_list + headline_list_test + body_article_list_test)

def tfidf_vector_fun(idx):
    head_vec = tfidf_vectorizer.transform([train_x[idx]['headline']]).toarray()
    body_vec = tfidf_vectorizer.transform([train_x[idx]['article']]).toarray()
    cos_sim = cosine_similarity(head_vec, body_vec)[0].reshape(1, 1)
    return head_vec[0], cos_sim[0], body_vec[0]

def tfidf_vector_fun_test(idx):
    head_vec = tfidf_vectorizer.transform([test_x[idx]['headline']]).toarray()
    body_vec = tfidf_vectorizer.transform([test_x[idx]['article']]).toarray()
    cos_sim = cosine_similarity(head_vec, body_vec)[0].reshape(1, 1)
    return head_vec[0], cos_sim[0], body_vec[0]


# In[17]:


head, sim, body = tfidf_vector_fun(0)
nn_input = np.concatenate((head, sim, body))

for ins_idx in range(1,len(train_x)):

    if (ins_idx % 500 == 0):
        print(str(ins_idx) + ':' + str(len(train_x)) )
        np.savetxt('D:/nlp_project/nn_input' + str(ins_idx) + '.out', nn_input)
        head, sim, body = tfidf_vector_fun(ins_idx)
        nn_input = np.concatenate((head, sim, body))
        continue
    head, sim, body = tfidf_vector_fun(ins_idx)
    nn_input = np.vstack((nn_input,np.concatenate((head, sim, body))))

np.savetxt('D:/nlp_project/nn_input' + str(len(train_x) - 1) + '.out', nn_input)


# In[59]:


head, sim, body = tfidf_vector_fun_test(0)
nn_test = np.concatenate((head, sim, body))

for ins_idx in range(1,len(test_x)):
    if (ins_idx % batch_size == 0):
        print(str(ins_idx) + ':' + str(len(test_x)) )
        np.savetxt('D:/nlp_project/nn_test' + str(ins_idx) + '.out', nn_test)
        head, sim, body = tfidf_vector_fun_test(ins_idx)
        nn_test = np.concatenate((head, sim, body))
        continue
    head, sim, body = tfidf_vector_fun_test(ins_idx)
    nn_test = np.vstack((nn_test,np.concatenate((head, sim, body))))

np.savetxt('D:/nlp_project/nn_test' + str(len(test_x) - 1) + '.out', nn_test)


# In[14]:


nn_input = np.genfromtxt('D:/nlp_project/train/nn_input500.out')

for idx in range (1, int(len(train_x) / batch_size)):
    data = np.genfromtxt('D:/nlp_project/train/nn_input' + str((idx + 1) * batch_size) + '.out')
    nn_input = np.vstack((nn_input,data))

data = np.genfromtxt('D:/nlp_project/train/nn_input' + str(len(train_x) - 1) + '.out')
nn_input = np.vstack((nn_input,data))

label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
train_y_one_hot = []
for ins in train_y:
    train_y_one_hot.append(label_ref[ins])
    
tensor_x = torch.from_numpy(nn_input)
tensor_y = torch.from_numpy(np.asarray(train_y_one_hot))
train_dataset = data_utils.TensorDataset(tensor_x, tensor_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)


# In[63]:


nn_test = np.genfromtxt('D:/nlp_project/test/nn_test500.out')

for idx in range (1, int(len(test_x) / batch_size)):
    print(idx)
    data = np.genfromtxt('D:/nlp_project/test/nn_test' + str((idx + 1) * batch_size) + '.out')
    nn_test = np.vstack((nn_test,data))
    print(nn_test.shape)

data = np.genfromtxt('D:/nlp_project/test/nn_test' + str(len(test_x) - 1) + '.out')
nn_test = np.vstack((nn_test,data))

test_y_one_hot = []
for ins in test_y:
    test_y_one_hot.append(label_ref[ins])

tensor_x_test = torch.from_numpy(nn_test)
tensor_y_test = torch.from_numpy(np.asarray(test_y_one_hot))
test_dataset = data_utils.TensorDataset(tensor_x_test, tensor_y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# In[26]:


import torch.nn.functional as F
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

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[29]:


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


# In[65]:


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = Variable(data, volatile=True).float()
        target = Variable(target).type(torch.LongTensor)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))


# In[30]:


for epoch in range(1, 90):
    train(epoch)
torch.save(model, 'D:/nlp_project/model.pt')


# In[66]:


test()

