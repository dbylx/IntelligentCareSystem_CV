# -*- coding: utf-8 -*-
'''
train mnist

image is grayscale with 28*28 size.
'''

# 导入包
from lenet import LeNet
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# 全局常量
LR = 0.01
BATCH_SIZE = 128
EPOCHS = 10

# 全局变量
accuracy_plot_path = 'plots/accuracy.png'
loss_plot_path = 'plots/loss.png'
output_model_path = 'models/mnist.hdf5'

################################################
# 第一部分：数据预处理
# grab the MNIST dataset
print('[INFO] 下载数据集...')
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# matrix shape should be: num_samples x rows x columns x depth
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# scale data to the range of [0,1]
trainData = trainData.astype('float32') / 255.0
testData = testData.astype('float32') / 255.0

# transform the training and testing labels into vectors
#in the range [0, classes]
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)


################################################3
# 第二部分：创建并训练模型
# initialize the optimizer and model
print('[INFO] 编译模型...')
opt = SGD(lr = LR)
model = LeNet.build(28,28,10,'',1)
model.compile(loss = 'categorical_crossentropy',
              optimizer=opt, metrics = ['accuracy'])

# train model
print('[INFO] 训练模型...')
H = model.fit(trainData, trainLabels,
              validation_data=(testData, testLabels),
              batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1)


################################################
# 第三部分：评估模型

# 画出accuracy曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, EPOCHS+1), H.history["acc"], label="train_acc")
plt.plot(np.arange(1, EPOCHS+1), H.history["val_acc"],label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(accuracy_plot_path)

# 画出loss曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1,EPOCHS+1),H.history["loss"], label="train_loss")
plt.plot(np.arange(1,EPOCHS+1),H.history["val_loss"],label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(loss_plot_path)

# 打印分类报告
# show accuracy on the testing set
print('[INFO] 评估模型...')
predictions = model.predict(testData, batch_size=32)
print(classification_report(testLabels.argmax(axis=1),
	                        predictions.argmax(axis=1),
                            target_names=[str(i) for i in range(10)]))


################################################
# 第四部分：保存模型
model.save(output_model_path)
