import cv2
import numpy as np
import math
import sys
import pygame
import _thread

video_path = "J:\\EAC-Datasets\\videos\\0413-1.mp4" # 视频路径
save_path = "J:\\EAC-Datasets\\images\\" # 保存路径

w_num = 8 # 显示数量
h_num = 4
pic_w = 320 # 每帧显示大小
pic_h = 180
each = 40 # 隔几帧输出一帧

save_str = "0414_" # 保存文件名前缀
save_start = 0 # 保存起始序号

vc=cv2.VideoCapture(video_path)

if vc.isOpened():
    success, image=vc.read()
else:
    success=False
    print("视频无法打开")
    sys.exit()

pygame.init()
screen = pygame.display.set_mode((w_num*pic_w,h_num*pic_h))

imgs = []
imgs_source = []
def fresh_images():
    global success
    global imgs_source
    global imgs
    imgs = []
    imgs_source = []
    count = 0
    while success:
        success, image = vc.read()
        count += 1
        if not success:
            break
        if count % each == 0:
            imgs_source.append(image)
            image = cv2.resize(image,(pic_w,pic_h))
            imgs.append(image)
            if(len(imgs)>=w_num*h_num):
                break
    img_lines = []
    for i in range(math.ceil(len(imgs)/h_num)):
        end = min(i*h_num+h_num,len(imgs))
        imgs_line = imgs[i*h_num:end]
        for l in range(h_num-len(imgs_line)):
            imgs_line.append(np.full_like(imgs_line[0],112))
        img_lines.append(np.vstack(imgs_line))
    cv2.imencode('.jpg', np.concatenate(img_lines, axis=1))[1].tofile("temp.jpg")
    image = pygame.image.load("temp.jpg")
    screen.blit(image,(0,0))
    return imgs

had_save = []
fresh_images()
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sys.exit()
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                if success:
                    had_save = []
                    screen.fill((112,112,112))
                    fresh_images()
                else:
                    print("Video end")
                    sys.exit()
        if e.type == pygame.MOUSEBUTTONDOWN:
            x,y = e.pos
            x = x//pic_w
            y = y//pic_h
            pygame.draw.rect(screen,(0,200,100),[x*pic_w,y*pic_h,pic_w,pic_h],3)
            had_save.append([x,y])
            save_img = imgs_source[x*h_num+y]
            path = save_path + save_str + str(save_start) + ".jpg"
            cv2.imwrite(path, save_img)
            print("save: " + path)
            save_start+=1
    pygame.display.flip()



