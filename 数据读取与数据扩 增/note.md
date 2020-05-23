图像读取
-------
Pillow, OpenCV    

Pillow
-------
提供了常见的图像读取和处理工作，且可与ipython notebook无缝集成  
官方文档    https://pillow.readthedocs.io/en/stable/  
导入Pillow库                         from PIL import Image   
读取图片                             im =Image.open(cat.jpg')  
应用模糊滤镜                         im2 = im.filter(ImageFilter.BLUR) im2.save('blur.jpg', 'jpeg')  
打开一个ipg文件，注意是当前路径        im = Image.open('cat.jpg') im.thumbnail((w//2, h//2)) im.save('thumbnail.jpg', 'jpeg')    

OpenCV
-------
计算机视觉，数字图像处理，机器视觉。比Pillow强大，但学习成本也高
官方文档    https://opencv.org/ 
导入OpenCV库                           import cv2   
读取图片                               img = cv2.imread('cat.jpg')  
OpenCV默认颜色通道顺序为BRG，转化一下    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
转化为灰度图                           img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
边缘检测                               edges = cv2.Canny(img, 30, 70) cv2.imwrite('canny.jpg', edges)
