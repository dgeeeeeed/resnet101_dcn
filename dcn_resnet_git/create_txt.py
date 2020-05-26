import os
names = os.listdir('../../data/line_detect/JPEGImages/')  #路径
i=0  #用于统计文件数量是否正确，不会写到文件里
train_val = open('train.txt','w')
for name in names:
    index = name.rfind('.')
    name = name[:index]
    train_val.write(name+'\n')
    i=i+1
print(i)

# path1 = '../../data/line_detect/img/'
# path2 = '../../data/line_detect/mask/'