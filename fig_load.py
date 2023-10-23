import numpy as np
import skimage as ski

pic_url = []
with open('Face_recognition/dev_urls.txt') as f:
    for i in f.readlines():
        pic_url.append(i.strip('\r\n'))

urls = []
name=[]
location=[]
for s in pic_url:
    n, _,_, url, loc, _ = s.split()
    urls.append(url)
    name.append(n)
    location.append(loc)

# 写入到文件里面
with open('url.data', 'w') as f:
    for i in urls:
        f.write(i)
        f.write('\n')

import urllib.request as request
import socket
import os

# 在同级目录新建文件夹存图片
# os.mkdir('./img')

# 为请求增加一下头
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
headers = ('User-Agent', user_agent)
opener = request.build_opener()
opener.addheaders = [headers]
request.install_opener(opener)

# 设定一下无响应时间，防止有的坏图片长时间没办法下载下来
timeout = 10
socket.setdefaulttimeout(timeout)

# 从文件里面读urls
urls = []
with open('./url.data') as f:
    for i in f.readlines():
        if i != '':
            urls.append(i)
        else:
            pass
# 通过urllibs的requests获取所有的图片
count = 1
# bad_url = []
for i,url in enumerate(urls):
    url.rstrip('\n')
    try:
        pic = request.urlretrieve(url, 'Face_recognition/train_img/%d.jpg' % count)
        temp_img=ski.io.imread('Face_recognition/train_img/%d.jpg' % count)
        loc = location[i]
        loc=np.array(loc.split(','))
        temp_img=temp_img[int(loc[1]):int(loc[3]),int(loc[0]):int(loc[2]),:]
        ski.io.imsave('Face_recognition/train_img/%d.jpg' % count,temp_img)
        print('pic %d' % count)
        count += 1
    except Exception as e:
        print(Exception, ':', e)
        # bad_url.append(url)
        name[i]=0
        
    print('\n')
print('got all photos that can be got')

new_name=[]
for i in range(len(name)):
    if(name[i] =='Abhishek'):
        new_name.append(0)
    if(name[i] =='Alex'):
        new_name.append(1)
    if(name[i] =='Ali'):
        new_name.append(2)
    if(name[i] =='Alyssa'):
        new_name.append(3)
    if(name[i] =='Anderson'):
        new_name.append(4)
    if(name[i] =='Anna'):
        new_name.append(5)
    if(name[i] =='Audrey'):
        new_name.append(6)
    if(name[i]=='Barack'):
        new_name.append(7)
new_name = np.array(new_name)
np.save('Face_recognition/train_name',new_name)

# # 把没有抓取到的urls保存起来
# with open('train_name.data', 'w') as f:
#     for i in new_name:
#         f.write(i)
#         f.write('\n')
#     print('saved sucessful name')