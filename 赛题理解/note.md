零基础入门CV-街道字符识别概要  
  计算机视觉中字符识别，预测街道字符编码
  
数据下载  
  训练集-http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.zip  http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json  
  验证集-http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.zip  http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json  
  测试集-http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_test_a.zip    
  提交样本-http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_sample_submit_A.csv  
  
数据标签  
top-左上角坐标X  
height-字符高度  
left-左上角坐标y  
width-字符宽度    
label-字符编码  
<img width="150" height="150" src="https://github.com/zhaying0617/Datawhale-CV/blob/master/微信图片_20200519233219.png">  

评测指标  
score=编码识别正确的数量/测试集图片数量  

读取数据（代码）  
import json  
import cv2  
import matplotlib.pyplot as plt  
train_json = json.load(open('C:/Users/zhayi/Desktop/kaggle/CV/train.json'))  
def parse_json(d):  
    arr = np.array([d['top'], d['height'], d['left'], d['width'], d['label']])  
    arr = arr.astype(int)  
    return arr  
image_dir='C:/Users/zhayi/Desktop/kaggle/CV/mchar_train/000000.png'  
img = cv2.imread(image_dir)  
arr = parse_json(train_json['000000.png'])  
plt.figure(figsize=(10, 10))  
plt.subplot(1, arr.shape[1]+1, 1)  
plt.imshow(img)  
plt.xticks([]); plt.yticks([])  
for idx in range(arr.shape[1]):  
    plt.subplot(1, arr.shape[1]+1, idx+2)  
    plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])  
    plt.title(arr[4, idx])  
    plt.xticks([]); plt.yticks([])      
<img width="150" height="150" src=" https://github.com/zhaying0617/Datawhale-CV/blob/master/img-storage/微信图片_20200520202205.png">  
   


        
