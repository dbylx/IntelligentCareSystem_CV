import cv2
import queue
import os
import numpy as np
from threading import Thread
import datetime,_thread
import subprocess as sp
import time
from oldcare.facial.faceutildlib import FaceUtil
limit_time = 2
# 使用线程锁，防止线程死锁
mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()

rtmpUrl="rtmp://192.168.37.5:1935/rtmplive/lxf"

faceutil = FaceUtil()

command=['ffmpeg',

                '-y',

                '-f', 'rawvideo',

                '-vcodec','rawvideo',

                '-pix_fmt', 'bgr24',

                '-s', "{}x{}".format(640, 480),# 图片分辨率

                '-r', str(25.0),# 视频帧率

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

            while True:
                counter += 1
                image = frame_queue.get()
                if counter <= 10:  # 放弃前10帧
                    continue
                image = cv2.flip(image, 1)

                if error == 1:
                    end_time = time.time()
                    difference = end_time - start_time
                    print(difference)
                    if difference >= limit_time:
                        error = 0

                face_location_list = faceutil.get_face_location(image)
                for (left, top, right, bottom) in face_location_list:
                    cv2.rectangle(image, (left, top), (right, bottom),
                                  (0, 0, 255), 2)

                p.stdin.write(image.tostring())
                # cv2.imshow('Collecting Faces', image)  # show the image
                # # Press 'ESC' for exiting video
                # k = cv2.waitKey(100) & 0xff
                # if k == 27:
                #     break

                face_count = len(face_location_list)
                if error == 0 and face_count == 0:  # 没有检测到人脸
                    print('[WARNING] 没有检测到人脸')
                    # audioplayer.play_audio(os.path.join(audio_dir,'no_face_detected.mp3'))
                    error = 1
                    start_time = time.time()
                elif error == 0 and face_count == 1:  # 可以开始采集图像了
                    print('[INFO] 可以开始采集图像了')
                    # audioplayer.play_audio(os.path.join(audio_dir,'start_image_capturing.mp3'))
                    break
                elif error == 0 and face_count > 1:  # 检测到多张人脸
                    print('[WARNING] 检测到多张人脸')
                    # audioplayer.play_audio(os.path.join(audio_dir, 'multi_faces_detected.mp3'))
                    error = 1
                    start_time = time.time()
                else:
                    pass
















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


