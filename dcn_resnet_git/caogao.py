from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('../../data/line_detect/JPEGImages/14.bmp').convert('RGB')
image,_ ,_  = image.split()
plt.imshow(image) # 显示图片
# plt.axis('off') # 不显示坐标轴
plt.show()