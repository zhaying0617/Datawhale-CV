构造验证集
-----
训练集-用于训练和调整参数  
验证集-用于验证模型精度和调整模型超参数  
测试集-验证模型的泛化能力  

划分本地验证集:  
   留出法 Hold-Out  
   交叉验证法 Cross Validation  
   自助采样法 BootStrap-适用于数据量较小的情况  
   <img width="400" height="200" src="https://github.com/zhaying0617/Datawhale-CV/blob/master/img-storage/图片4.png">  


模型的训练和验证
-----
构造训练集与验证集

    #构造训练集
    train_path = glob.glob('C:/Users/zhayi/Desktop/kaggle/CV/mchar_train/*.png')
    train_path.sort()
    train_json = json.load(open('C:/Users/zhayi/Desktop/kaggle/CV/train.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    print(len(train_path), len(train_label))

    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                    transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
        batch_size=40,
        shuffle=True,
        num_workers=0,
    )
    
    #构造验证集
    val_path = glob.glob('C:/Users/zhayi/Desktop/kaggle/CV/mchar_val/*.png')
    val_path.sort()
    val_json = json.load(open('C:/Users/zhayi/Desktop/kaggle/CV/val.json'))
    val_label = [val_json[x]['label'] for x in val_json]
    print(len(val_path), len(val_label))

    val_loader = torch.utils.data.DataLoader(
        SVHNDataset(val_path, val_label,
                    transforms.Compose([
                    transforms.Resize((60, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
        batch_size=40,
        shuffle=False,
        num_workers=0,
     )      


每轮进行训练和验证

    #进行训练
    def train(train_loader, model, criterion, optimizer):
    # 切换模型为训练模式
      model.train()
      train_loss = []
    
        for i, (input, target) in enumerate(train_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            
               c0, c1, c2, c3, c4 = model(input)
               target=target.long()
               loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3]) + \
               criterion(c4, target[:, 4])
               #         loss /= 6
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               if i % 100 == 0:
                  print(loss.item())
        
               train_loss.append(loss.item())
        return np.mean(train_loss)

    #进行验证
    def validate(val_loader, model, criterion):
        # 切换模型为预测模型
      model.eval()
      val_loss = []
       # 不记录模型梯度信息
      with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                if use_cuda:
                    input = input.cuda()
                    target = target.cuda()
                
                c0, c1, c2, c3, c4 = model(input)
                target=target.long()
                loss = criterion(c0, target[:, 0]) + \
                       criterion(c1, target[:, 1]) + \
                       criterion(c2, target[:, 2]) + \
                       criterion(c3, target[:, 3]) + \
                       criterion(c4, target[:, 4])
                #loss /= 6
                val_loss.append(loss.item())
      return np.mean(val_loss)



