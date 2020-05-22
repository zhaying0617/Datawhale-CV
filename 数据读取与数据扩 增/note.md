图像读取    
Pillow, OpenCV    

Pillow  
提供了常见的图像读取和处理工作，且可与ipython notebook无缝集成  
官方文档    https://pillow.readthedocs.io/en/stable/  
导入Pillow库                         from PIL import Image   
读取图片                             im =Image.open(cat.jpg')  
应用模糊滤镜                         im2 = im.filter(ImageFilter.BLUR) im2.save('blur.jpg', 'jpeg')  
打开一个ipg文件，注意是当前路径        im = Image.open('cat.jpg') im.thumbnail((w//2, h//2)) im.save('thumbnail.jpg', 'jpeg')    
