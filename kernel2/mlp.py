from dependencies import *
from loaddata import *


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


model = mlpmodel()
opt = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


'''
Test data

a = torch.tensor(p_emb_dev,dtype=torch.float)
b = torch.tensor(a_emb_dev,dtype=torch.float)
c = torch.tensor(b_emb_dev,dtype=torch.float)
d = torch.tensor(pa_pos_dev,dtype=torch.float)
e = torch.tensor(pb_pos_dev,dtype=torch.float)

outs = mod([a,b,c],[d,e])
'''


dev_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_dev,a_emb_dev,b_emb_dev,pa_pos_dev,pb_pos_dev]]
train_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_tra,a_emb_tra,b_emb_tra,pa_pos_tra,pb_pos_tra]]
test_data = [torch.tensor(x,dtype=torch.float) for x in [p_emb_test,a_emb_test,b_emb_test,pa_pos_test,pb_pos_test]]



#savethemodel
# def save_checkpoint(filename, **states):
#    torch.save(states, filename)

# utils.save_checkpoint('model_mlp.pth.tar', **{'model' : model.state_dict(), 'opt' : opt.state_dict()})



