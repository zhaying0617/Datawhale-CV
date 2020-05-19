import json  
import cv2  
import matplotlib.pyplot as plt  
train_json = json.load(open('C:/Users/zhayi/Desktop/kaggle/CV/train.json'))
image_dir='C:/Users/zhayi/Desktop/kaggle/CV/mchar_train/000000.png'  
img = cv2.imread(image_dir)  
arr = parse_json(train_json['000000.png'])  
def parse_json(d):  
    arr = np.array([d['top'], d['height'], d['left'], d['width'], d['label']])  
    arr = arr.astype(int)  
    return arr  

plt.figure(figsize=(10, 10))  
plt.subplot(1, arr.shape[1]+1, 1)  
plt.imshow(img)  
plt.xticks([]); plt.yticks([])  
for idx in range(arr.shape[1]):  
    plt.subplot(1, arr.shape[1]+1, idx+2)  
    plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])  
    plt.title(arr[4, idx])  
    plt.xticks([]); plt.yticks([])  
