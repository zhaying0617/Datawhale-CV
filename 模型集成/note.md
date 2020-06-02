集成学习方法
-----
集成学习目的——提高预测精度  
集成学习方法——Stacking Bagging Boosting  
注 若硬件设备不允许——留出法    追求精度——交叉验证法  
   
深度学习中的集成学些
-----
Dropout——在每次训练批次中，随机让一部分的节点停止工作；但在预测过程中让所有的节点都工作  
<img width="500" height="280" src="https://github.com/zhaying0617/Datawhale-CV/blob/master/img-storage/图片5.png">  
加入Dropout的网络结构

    # 定义模型
    class SVHN_Model1(nn.Module):
        def __init__(self):
            super(SVHN_Model1, self).__init__()
            # CNN提取特征模块
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.MaxPool2d(2),
            )
            self.fc1 = nn.Linear(32*3*7, 11)
            self.fc2 = nn.Linear(32*3*7, 11)
            self.fc3 = nn.Linear(32*3*7, 11)
            self.fc4 = nn.Linear(32*3*7, 11)
            self.fc5 = nn.Linear(32*3*7, 11)
            self.fc6 = nn.Linear(32*3*7, 11)
        def forward(self, img):
            feat = self.cnn(img)
            feat = feat.view(feat.shape[0], -1)
            c1 = self.fc1(feat)
            c2 = self.fc2(feat)
            c3 = self.fc3(feat)
            c4 = self.fc4(feat)
            c5 = self.fc5(feat)
            c6 = self.fc6(feat)
            return c1, c2, c3, c4, c5, c6
            
TTA——测试集数据扩增（Test Time Augmentation);在训练和预测时都可以用；对同一个样本预测三次后对三次结果求平均

    def predict(test_loader, model, tta=10):
       model.eval()
       test_pred_tta = None
       # TTA 次数
       for _ in range(tta):
           test_pred = []
           with torch.no_grad():
              for i, (input, target) in enumerate(test_loader):
                  c0, c1, c2, c3, c4, c5 = model(data[0])
                  output = np.concatenate([c0.data.numpy(), c1.data.numpy(),
                     c2.data.numpy(), c3.data.numpy(),
                     c4.data.numpy(), c5.data.numpy()], axis=1)
           test_pred = np.vstack(test_pred)
           if test_pred_tta is None:
              test_pred_tta = test_pred
           else:
              test_pred_tta += test_pred
       return test_pred_tta
test_pred.append(output)
