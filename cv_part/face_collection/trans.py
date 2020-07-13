import cv2
import queue
import os
import numpy as np
from threading import Thread
import datetime,_thread
import subprocess as sp
import time
from oldcare.facial.faceutildlib import FaceUtil

import argparse
from oldcare.facial import FaceUtil
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import shutil
import time



limit_time = 2
# 使用线程锁，防止线程死锁
mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()


rtmpUrl = "rtmp://39.97.107.19:1935/rtmplive"

faceutil = FaceUtil()

action_list = ['blink', 'open_mouth', 'smile', 'rise_head', 'bow_head',
               'look_left', 'look_right']
action_map = {'blink': '请眨眼', 'open_mouth': '请张嘴',
              'smile': '请笑一笑', 'rise_head': '请抬头',
              'bow_head': '请低头', 'look_left': '请看左边',
              'look_right': '请看右边'}


command=['ffmpeg',

                '-y',

                '-f', 'rawvideo',

                '-vcodec','rawvideo',

                '-pix_fmt', 'bgr24',

                '-s', "{}x{}".format(640, 480),# 图片分辨率

                '-r', str(4.0),# 视频帧率

                '-i', '-',

                '-c:v', 'libx264',

                '-pix_fmt', 'yuv420p',

                '-preset', 'ultrafast',

                '-f', 'flv',

                rtmpUrl]

 

def Video():

# 调用相机拍图的函数

    vid = cv2.VideoCapture(0)

    if not vid.isOpened():

        raise IOError("Couldn't open webcam or video")

    while (vid.isOpened()):

        return_value, frame = vid.read()

 

        # 原始图片推入队列中

        frame_queue.put(frame)
        frame_queue.get() if frame_queue.qsize() > 1 else time.sleep(0.01)
 

 

def push_frame():

    # 推流函数

    accum_time = 0

    curr_fps = 0

    fps = "FPS: ??"

 #   prev_time = time()

 

    # 防止多线程时 command 未被设置

    while True:
        print('command lenth',len(command))
        if len(command) > 0:
           

            # 管道配置，其中用到管道

            p = sp.Popen(command, stdin=sp.PIPE)

            break

 

    while True:
        if frame_queue.empty() != True:
            counter = 0
            for action in action_list:

                action_name = action_map[action]

                counter = 1
                for i in range(15):
                    print('%s-%d' % (action_name, i))
                    img_OpenCV = frame_queue.get()

                    img_OpenCV = cv2.flip(img_OpenCV, 1)
                    origin_img = img_OpenCV.copy()  # 保存时使用

                    face_location_list = faceutil.get_face_location(img_OpenCV)
                    for (left, top, right, bottom) in face_location_list:
                        cv2.rectangle(img_OpenCV, (left, top),
                                      (right, bottom), (0, 0, 255), 2)

                    img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV,
                                                           cv2.COLOR_BGR2RGB))

                    draw = ImageDraw.Draw(img_PIL)
                    draw.text((int(img_OpenCV.shape[1] / 2), 30), action_name,
                              font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40),
                              fill=(255, 0, 0))  # linux

                    # 转换回OpenCV格式
                    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),
                                              cv2.COLOR_RGB2BGR)

                    image_name = os.path.join("images", "202",
                                              action + '_' + str(counter) + '.jpg')
                    print(image_name)
                    cv2.imwrite(image_name, origin_img)


                    cv2.imshow('Collecting Faces', img_OpenCV)  # show the image

                    # image_name = os.path.join(args['imagedir'], args['id'],
                    #                           action + '_' + str(counter) + '.jpg')
                    # cv2.imwrite(image_name, origin_img)
                    # Press 'ESC' for exiting video
                    p.stdin.write(img_OpenCV.tostring())
                    k = cv2.waitKey(100) & 0xff
                    if k == 27:
                        break

            break;













            # 你处理图片的代码

            # 将图片从队列中取出来做处理，然后再通过管道推送到服务器上

            # 增加画面帧率

        
          

            # 将处理后的图片通过管道推送到服务器上



 

def run():

     #使用两个线程处理

 

    thread1 = Thread(target=Video,)

    thread1.start()

    thread2 = Thread(target=push_frame,)

    thread2.start()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()


