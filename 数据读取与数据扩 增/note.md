图像读取
-------
Pillow, OpenCV    


Pillow
-------
提供了常见的图像读取和处理工作，且可与ipython notebook无缝集成  
官方文档    https://pillow.readthedocs.io/en/stable/  
导入Pillow库:                          from PIL import Image   
读取图片:                              im =Image.open(cat.jpg')  
应用模糊滤镜:                          im2 = im.filter(ImageFilter.BLUR) im2.save('blur.jpg', 'jpeg')  
打开一个ipg文件，注意是当前路径:         im = Image.open('cat.jpg') im.thumbnail((w//2, h//2)) im.save('thumbnail.jpg', 'jpeg') 


OpenCV
-------
计算机视觉，数字图像处理，机器视觉。比Pillow强大，但学习成本也高  

官方文档    https://opencv.org/  

导入OpenCV库:                          import cv2     
读取图片:                              img = cv2.imread('cat.jpg')    
OpenCV默认颜色通道顺序为BRG，转化一下:   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
<img width="150" height="150" src="https://github.com/zhaying0617/Datawhale-CV/blob/master/img-storage/图片1.png">    
转化为灰度图:                           img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
<img width="150" height="150" src="https://github.com/zhaying0617/Datawhale-CV/blob/master/img-storage/图片2.png">  
边缘检测:                               edges = cv2.Canny(img, 30, 70) cv2.imwrite('canny.jpg', edges)  
<img width="150" height="150" src="https://github.com/zhaying0617/Datawhale-CV/blob/master/img-storage/图片3.png">  


数据扩增方法
-------
图像颜色，尺寸，形态，空间，像素等角度进行变化  

transforms.CenterCrop:             对图片中心进行裁剪    
transforms.ColorJitter:            对图像颜色的对比度，饱和度，零度进行变换    
transforms.FiveCrop:               对图像四个角和中心进行裁剪得到五分图像    
transforms.Grayscale:              对图像进行灰度变化  
transforms.Pad:                    使用固定值进行像素填充  
transforms.RandomAffine:           随机仿射变化  
transforms.RandomCrop:             随机区域裁剪   
transforms.RandomHorizontalFlip:   随机水平翻转  
transforms.RandomRotation:         随机旋转  
transforms.RandomVerticalFlip:     随机垂直翻转  

常用的数据扩增库
torchvision:       可无缝与torch集成，但扩增方法种类少且速度中等         https://github.com/pytorch/vision    
imgaug:            第三方数据扩增库，提供多样数据扩增方法且速度较快       https://github.com/aleju/imgaug  
albumentations:    第三方数据扩增库，提供多样数据扩增方法且速度较快       https://albumentations.readthedocs.io 


Pytorch读取数据  
-------
Dataset:对数据集的封装及提供索引方式，并对数据样本进行读取  

    import os, sys, glob, shutil, json  
    import cv2  
    from PIL import Image  
    import numpy as np  
    import torch  
    from torch.utils.data.dataset import Dataset  
    import torchvision.transforms as transforms  

    class SVHNDataset(Dataset):

      def __init__(self, img_path, img_label, transform=None):
          self.img_path = img_path
          self.img_label = img_label
          if transform is not None:
            self.transform = transform
          else:
            self.transform = None


      def __getitem__(self, index):
         img = Image.open(self.img_path[index]).convert('RGB')
         if self.transform is not None:
            img = self.transform(img)
            lbl = np.array(self.img_label[index], dtype=np.int)
            lbl = list(lbl) + (5 - len(lbl)) * [10]
            return img, torch.from_numpy(np.array(lbl[:5]))


      def __len__(self):
         return len(self.img_path)


    train_path = glob.glob('C:/Users/zhayi/Desktop/kaggle/CV/mchar_train/*.png')
    train_path.sort()
    train_json = json.load(open('C:/Users/zhayi/Desktop/kaggle/CV/train.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    
    data = SVHNDataset(train_path, train_label,  
              transforms.Compose([  
                 #缩放到固定尺⼨  
                 transforms.Resize((64, 128)),  
                 #随机颜⾊色变换  
                 transforms.ColorJitter(0.2, 0.2, 0.2),  
                 #加⼊随机旋转  
                 transforms.RandomRotation(5),  
              ]))  
          
DataLoder:在定义好的Dataset基础上构建DataLoder，对Dataset封装，提供批量读取并迭代

    class SVHNDataset(Dataset):

      def __init__(self, img_path, img_label, transform=None):
          self.img_path = img_path
          self.img_label = img_label
          if transform is not None:
            self.transform = transform
          else:
            self.transform = None


      def __getitem__(self, index):
         img = Image.open(self.img_path[index]).convert('RGB')
         if self.transform is not None:
            img = self.transform(img)
            lbl = np.array(self.img_label[index], dtype=np.int)
            lbl = list(lbl) + (5 - len(lbl)) * [10]
            return img, torch.from_numpy(np.array(lbl[:5]))


      def __len__(self):
         return len(self.img_path)


    train_path = glob.glob('C:/Users/zhayi/Desktop/kaggle/CV/mchar_train/*.png')
    train_path.sort()
    train_json = json.load(open('C:/Users/zhayi/Desktop/kaggle/CV/train.json'))
    train_label = [train_json[x]['label'] for x in train_json]

    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                  transforms.Compose([
                  transforms.Resize((64, 128)),
                  transforms.ColorJitter(0.3, 0.3, 0.2),
                  transforms.RandomRotation(5),
                   transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])),
      batch_size=10, # 每批样本个数
      shuffle=False, # 是否打乱顺序
      num_workers=10, # 读取的线程个数
    )



