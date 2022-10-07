from torch import nn

class Classifier(nn.Module):
	def __init__(self, input_dim, num_classes):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1,  padding = 'valid')
		self.act = nn.ReLU()
		self.drop = nn.Dropout(p=0.2)
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3))
		self.drop2 = nn.Dropout(p=0.2)
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
		self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3))
		self.drop3 = nn.Dropout(p=0.2)
		self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
		self.flatten = nn.Flatten()
		self.linear1 = nn.Linear(144, 32)
		self.linear2 = nn.Linear(32, num_classes)
		self.act2 = nn.Softmax()

	def forward(self, x, training = False):
		x = self.conv1(x)
		x = self.act(x)
		if training:
			x = self.drop(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.act(x)
		if training:
			x = self.drop2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.act(x)
		x = self.drop3(x)
		if training:
			x = self.pool3(x)
		x = self.flatten(x)
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.act2(x)
		return x