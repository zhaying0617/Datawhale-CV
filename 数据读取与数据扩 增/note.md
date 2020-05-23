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



