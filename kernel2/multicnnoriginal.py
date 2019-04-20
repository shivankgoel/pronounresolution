from dependencies import *
from loaddata import *


class multichannelcnnmodel(nn.Module):
	def __init__(self):
		super().__init__()
		self.dropout1 = nn.Dropout(p=0.5)
		self.dropout2 = nn.Dropout(p=0.5)
		self.conv1 = nn.Conv1d(300, 10, 3)
		#output is 10*9
		self.pool1 = nn.MaxPool1d(9,1)
		#output is 10*1
		self.conv2 = nn.Conv1d(300,10,5)
		#output is 10*7
		self.pool2 = nn.MaxPool1d(7, 1)
		#output is 10*1acc
		self.fc2 = nn.Linear(45,10)
		self.bn1 = nn.BatchNorm1d(80)
		self.fc3 = nn.Linear(80,60)
		self.fc4 = nn.Linear(60,3)
	def forward(self,embeddings,positions):
		a_s = [self.conv1(self.dropout1(embedding.transpose(1, 2))) for embedding in embeddings]
		a = [self.pool1(t).transpose(1,2).flatten(1) for t in a_s]
		b_s = [self.conv2(self.dropout2(embedding.transpose(1, 2))) for embedding in embeddings]
		b = [self.pool2(t).transpose(1, 2).flatten(1) for t in b_s]
		e1 = [torch.cat((a[i],b[i]),1) for i in range(3)]
		e1 = [e.flatten(1) for e in e1]
		x1 = torch.cat(tuple([e for e in e1]),1)
		x2 = torch.cat(tuple([self.fc2(position) for position in positions]),1)
		x = F.relu(torch.cat((x1,x2),1))
		xnorm = self.bn1(x)
		hiddenlayer = F.relu(self.fc3(xnorm))
		outs = F.softmax(self.fc4(hiddenlayer),dim=1)
		return outs


dev_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_dev,a_emb_dev,b_emb_dev,pa_pos_dev,pb_pos_dev]]
train_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_tra,a_emb_tra,b_emb_tra,pa_pos_tra,pb_pos_tra]]
test_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_test,a_emb_test,b_emb_test,pa_pos_test,pb_pos_test]]

y_tra = torch.tensor(y_tra)
y_dev = torch.tensor(y_dev)
y_test = torch.tensor(y_test)


model = multichannelcnnmodel()
opt = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

num_epochs = 20

for i in range(num_epochs):
	opt.zero_grad()
	outputs = model([train_data[0],train_data[1],train_data[2]],[train_data[3],train_data[4]])
	loss = criterion(outputs, y_tra)
	loss.backward()
	opt.step()
	if i%10==0:
		print('Epoch: %4f Loss : %4f'%(i,loss))


model.eval()
dev_outputs = model([dev_data[0],dev_data[1],dev_data[2]],[dev_data[3],dev_data[4]])
predicted_labels = torch.max(dev_outputs,1)[1]
acc = np.sum(np.array(predicted_labels) == y_dev.values)/len(y_dev)




#https://www.kaggle.com/keyit92/coref-by-mlp-cnn-coattention/data
