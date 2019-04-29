from dependencies import *
# from loaddata import *


class mlpmodel(nn.Module):
	def __init__(self):
		super().__init__()
		self.dropout1 = nn.Dropout(p=0.5)
		self.fc1 = nn.Linear(11*300,10)
		self.fc2 = nn.Linear(45,10)
		self.bn1 = nn.BatchNorm1d(50)
		self.fc3 = nn.Linear(50,60)
		self.fc4 = nn.Linear(60,3)
	def forward(self,embeddings,positions):
		e1 = [self.dropout1(embedding) for embedding in embeddings]
		e1 = [e.flatten(1) for e in e1]
		x1 = torch.cat(tuple([self.fc1(e) for e in e1]),1)
		x2 = torch.cat(tuple([self.fc2(position) for position in positions]),1)
		x = F.relu(torch.cat((x1,x2),1))
		xnorm = self.bn1(x)
		hiddenlayer = F.relu(self.fc3(xnorm))
		outs = F.softmax(self.fc4(hiddenlayer),dim=1)
		return outs


'''
Test data

a = torch.tensor(p_emb_dev,dtype=torch.float)
b = torch.tensor(a_emb_dev,dtype=torch.float)
c = torch.tensor(b_emb_dev,dtype=torch.float)
d = torch.tensor(pa_pos_dev,dtype=torch.float)
e = torch.tensor(pb_pos_dev,dtype=torch.float)

outs = mod([a,b,c],[d,e])
'''

data = dict()

with open(data_save_path, 'rb') as f:
   data = pickle.load(f)
f.close()

p_emb_tra = data['p_emb_tra']
p_emb_dev = data['p_emb_dev']
p_emb_test = data['p_emb_test']

a_emb_tra = data['a_emb_tra']
a_emb_dev = data['a_emb_dev']
a_emb_test = data['a_emb_test']

b_emb_tra = data['b_emb_tra']
b_emb_dev = data['b_emb_dev']
b_emb_test = data['b_emb_test']

pa_pos_tra = data['pa_pos_tra']
pa_pos_dev = data['pa_pos_dev']
pa_pos_test = data['pa_pos_test']

pb_pos_tra = data['pb_pos_tra']
pb_pos_dev = data['pb_pos_dev']
pb_pos_test = data['pb_pos_test']

y_tra = data['y_tra']
y_dev = data['y_dev']
y_test = data['y_test']


dev_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_dev,a_emb_dev,b_emb_dev,pa_pos_dev,pb_pos_dev]]
train_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_tra,a_emb_tra,b_emb_tra,pa_pos_tra,pb_pos_tra]]
test_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_test,a_emb_test,b_emb_test,pa_pos_test,pb_pos_test]]

y_tra = torch.tensor(y_tra)
y_dev = torch.tensor(y_dev)
y_test = torch.tensor(y_test)



model = mlpmodel()
opt = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

num_epochs = 1000

for i in range(num_epochs):
	opt.zero_grad()
	outputs = model([train_data[0],train_data[1],train_data[2]],[train_data[3],train_data[4]])
	loss = criterion(outputs, y_tra)
	loss.backward()
	opt.step()
	if i%10==0:
		print('Epoch: %4f Loss : %4f'%(i,loss))


model.eval()
tra_outputs = model([train_data[0],train_data[1],train_data[2]],[train_data[3],train_data[4]])
dev_outputs = model([dev_data[0],dev_data[1],dev_data[2]],[dev_data[3],dev_data[4]])
test_outputs = model([test_data[0],test_data[1],test_data[2]],[test_data[3],test_data[4]])

predicted_labels_tra = torch.max(tra_outputs,1)[1]
predicted_labels_dev = torch.max(dev_outputs,1)[1]
predicted_labels_test = torch.max(test_outputs,1)[1]

predvals_tra = np.array(predicted_labels_tra)
predvals_dev = np.array(predicted_labels_dev)
predvals_test = np.array(predicted_labels_test)

truevals_tra = np.array(y_tra)
truevals_dev = np.array(y_dev)
truevals_test = np.array(y_test)

pred_dic_tra = dict()
pred_dic_dev = dict()
pred_dic_test = dict()

pred_dic_tra['prediction'] = predvals_tra
pred_dic_dev['prediction'] = predvals_dev
pred_dic_test['prediction'] = predvals_test

pred_dic_tra['label'] = truevals_tra
pred_dic_dev['label'] = truevals_dev
pred_dic_test['label'] = truevals_test


with open(result_save_path + '_train.pickle', 'wb') as f:
   pickle.dump(pred_dic_tra, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

with open(result_save_path + '_dev.pickle', 'wb') as f:
   pickle.dump(pred_dic_dev, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

with open(result_save_path + '_test.pickle', 'wb') as f:
   pickle.dump(pred_dic_test, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()




#savethemodel
# def save_checkpoint(filename, **states):
#    torch.save(states, filename)
# utils.save_checkpoint('model_mlp.pth.tar', **{'model' : model.state_dict(), 'opt' : opt.state_dict()})



