import cv2
import queue
import os

import imutils
import numpy as np
from threading import Thread
import datetime, _thread
import subprocess as sp
import time

import argparse
from oldcare.facial import FaceUtil
from PIL import Image, ImageDraw, ImageFont
from oldcare.utils import fileassistant
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import time
import numpy as np
import os
import imutils
import subprocess


VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ANGLE = 20
# 全局常量

limit_time = 2
# 使用线程锁，防止线程死锁
mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()

rtmpUrl = "rtmp://39.97.107.19:1935/rtmplive"

# "rtmp://39.97.107.19:1935/rtmplive"


command = ['ffmpeg',

           '-y',

           '-f', 'rawvideo',

           '-vcodec', 'rawvideo',

           '-pix_fmt', 'bgr24',

           '-s', "{}x{}".format(640, 480),  # 图片分辨率

           '-r', str(13.0),  # 视频帧率

           '-i', '-',

           '-c:v', 'libx264',

           '-pix_fmt', 'yuv420p',

           '-preset', 'ultrafast',

           '-f', 'flv',

           rtmpUrl]


def Video():
    # 调用相机拍图的函数

    vid = cv2.VideoCapture(0)
    time.sleep(2);
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    while (vid.isOpened()):
        return_value, frame = vid.read()

        # 原始图片推入队列中

        frame_queue.put(frame)
        frame_queue.get() if frame_queue.qsize() > 1 else time.sleep(0.01)


def push_frame():
    # 推流函数
    facial_recognition_model_path = 'info/face_recognition_hog.pickle'  # jian ce ren lian
    facial_expression_model_path = 'models/face_expression_seven_class.hdf5'  # fenxi qinggan

    output_stranger_path = 'supervision/strangers'
    output_smile_path = 'supervision/smile'

    people_info_path = 'info/people_info.csv'
    facial_expression_info_path = 'info/facial_expression_info_seven_class.csv'
    faceutil = FaceUtil(facial_recognition_model_path)
    facial_expression_model = load_model(facial_expression_model_path, compile=False)
    # 初始化人脸识别模型

    # your python path
    python_path = '/root/anaconda3/envs/tensorflow/bin/python'

    # 全局常量
    FACIAL_EXPRESSION_TARGET_WIDTH = 48
    FACIAL_EXPRESSION_TARGET_HEIGHT = 48

    ANGLE = 20

    # 得到 ID->姓名的map 、 ID->职位类型的map、
    # 摄像头ID->摄像头名字的map、表情ID->表情名字的map
    id_card_to_name, id_card_to_type = fileassistant.get_people_info(
        people_info_path)
    facial_expression_id_to_name = fileassistant.get_facial_expression_info(
        facial_expression_info_path)

    # 控制陌生人检测
    strangers_timing = 0  # 计时开始
    strangers_start_time = 0  # 开始时间
    strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.

    # 控制微笑检测
    facial_expression_timing = 0  # 计时开始
    facial_expression_start_time = 0  # 开始时间
    facial_expression_limit_time = 2  # if >= 2 seconds, he/she is smiling

    accum_time = 0

    curr_fps = 0

    fps = "FPS: ??"

    #   prev_time = time()

    # 防止多线程时 command 未被设置

    while True:
        print('command lenth', len(command))
        if len(command) > 0:
            # 管道配置，其中用到管道

            p = sp.Popen(command, stdin=sp.PIPE)

            break

    while True:
        faceutil = FaceUtil(facial_recognition_model_path)
        facial_expression_model = load_model(facial_expression_model_path, compile=False)
        if frame_queue.empty() != True:
            counter = 0

            while True:
                counter += 1
                image = frame_queue.get()


                image=cv2.flip(image, 1)

                image = imutils.resize(image, width=VIDEO_WIDTH,
                                       height=VIDEO_HEIGHT)  # 压缩，加快识别速度

                # if counter%10!=0:
                # 	cv2.imshow("Checking Strangers and Ole People's Face Expression",
                # 			   image)
                #
                # 	# Press 'ESC' for exiting video
                # 	k = cv2.waitKey(1) & 0xff
                # 	continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

                # if True:
                # 	cv2.imshow("Checking Strangers and Ole People's Face Expression",
                # 			   gray)
                # 	continue

                face_location_list, names = faceutil.get_face_location_and_name(
                    image)

                # 得到画面的四分之一位置和四分之三位置，并垂直划线
                one_fourth_image_center = (int(VIDEO_WIDTH / 4),
                                           int(VIDEO_HEIGHT / 4))
                three_fourth_image_center = (int(VIDEO_WIDTH / 4 * 3),
                                             int(VIDEO_HEIGHT / 4 * 3))

                cv2.line(image, (one_fourth_image_center[0], 0),
                         (one_fourth_image_center[0], VIDEO_HEIGHT),
                         (0, 255, 255), 1)
                cv2.line(image, (three_fourth_image_center[0], 0),
                         (three_fourth_image_center[0], VIDEO_HEIGHT),
                         (0, 255, 255), 1)

                # 处理每一张识别到的人脸
                for ((left, top, right, bottom), name) in zip(face_location_list,
                                                              names):

                    # 将人脸框出来
                    rectangle_color = (0, 0, 255)
                    if id_card_to_type[name] == 'old_people':
                        rectangle_color = (0, 0, 128)
                    elif id_card_to_type[name] == 'employee':
                        rectangle_color = (255, 0, 0)
                    elif id_card_to_type[name] == 'volunteer':
                        rectangle_color = (0, 255, 0)
                    else:
                        pass
                    cv2.rectangle(image, (left, top), (right, bottom),
                                  rectangle_color, 2)

                    # 陌生人检测逻辑
                    if 'Unknown' in names:  # alert
                        if strangers_timing == 0:  # just start timing
                            strangers_timing = 1
                            strangers_start_time = time.time()
                        else:  # already started timing
                            strangers_end_time = time.time()
                            difference = strangers_end_time - strangers_start_time

                            current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                         time.localtime(time.time()))

                            if difference < strangers_limit_time:
                                print('[INFO] %s, 房间, 陌生人仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                            else:  # strangers appear
                                event_desc = '陌生人出现!!!'
                                event_location = '房间'
                                print('[EVENT] %s, 房间, 陌生人出现!!!' % (current_time))
                                cv2.imwrite(os.path.join(output_stranger_path,
                                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                            image)  # snapshot
                                pic_path = os.path.join(output_stranger_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S')))
                                # insert into database
                                command1 = '%s inserting.py --event_desc %s --event_type 2 --event_location %s --pic_path %s' % (
                                python_path, event_desc, event_location,pic_path)
                                p2 = subprocess.Popen(command1, shell=True)

                                # 开始陌生人追踪
                                unknown_face_center = (int((right + left) / 2),
                                                       int((top + bottom) / 2))

                                cv2.circle(image, (unknown_face_center[0],
                                                   unknown_face_center[1]), 4, (0, 255, 0), -1)

                                direction = ''
                                # face locates too left, servo need to turn right,
                                # so that face turn right as well
                                if unknown_face_center[0] < one_fourth_image_center[0]:
                                    direction = 'right'
                                elif unknown_face_center[0] > three_fourth_image_center[0]:
                                    direction = 'left'

                                # adjust to servo
                                if direction:
                                    print('%d-摄像头需要 turn %s %d 度' % (counter,
                                                                     direction, ANGLE))

                    else:  # everything is ok
                        strangers_timing = 0

                    # 表情检测逻辑
                    # 如果不是陌生人，且对象是老人
                    if name != 'Unknown' and id_card_to_type[name] == 'old_people':
                        # 表情检测逻辑
                        roi = gray[top:bottom, left:right]
                        roi = cv2.resize(roi, (FACIAL_EXPRESSION_TARGET_WIDTH,
                                               FACIAL_EXPRESSION_TARGET_HEIGHT))
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        # determine facial expression
                        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

                        emotion_value_list = facial_expression_model.predict(roi)[0]

                        facial_expression_label = emotions[np.argmax(emotion_value_list)]

                        if facial_expression_label == 'Happy':  # alert
                            if facial_expression_timing == 0:  # just start timing
                                facial_expression_timing = 1
                                facial_expression_start_time = time.time()
                            else:  # already started timing
                                facial_expression_end_time = time.time()
                                difference = facial_expression_end_time - facial_expression_start_time

                                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                             time.localtime(time.time()))
                                if difference < facial_expression_limit_time:
                                    print('[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (
                                    current_time, id_card_to_name[name], difference))
                                else:  # he/she is really smiling
                                    event_desc = '%s正在笑' % (id_card_to_name[name])
                                    event_location = '房间'
                                    print('[EVENT] %s, 房间, %s正在笑.' % (current_time, id_card_to_name[name]))
                                    pic_path = os.path.join(output_smile_path,'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S')))
                                    cv2.imwrite(pic_path,image)  # snapshot

                                    # insert into database
                                    command1 = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d --pic_path %s' % (
                                    python_path, event_desc, event_location, int(name),pic_path)
                                    p2 = subprocess.Popen(command1, shell=True)

                        else:  # everything is ok
                            facial_expression_timing = 0

                    else:  # 如果是陌生人，则不检测表情
                        facial_expression_label = ''

                    # 人脸识别和表情识别都结束后，把表情和人名写上
                    # (同时处理中文显示问题)
                    img_PIL = Image.fromarray(cv2.cvtColor(image,
                                                           cv2.COLOR_BGR2RGB))

                    draw = ImageDraw.Draw(img_PIL)
                    final_label = id_card_to_name[name] + ': ' + facial_expression_id_to_name[facial_expression_label] if facial_expression_label else id_card_to_name[name]
                    draw.text((left, top - 30), final_label,
                              font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40),
                              fill=(255, 0, 0))  # linux

                    # 转换回OpenCV格式
                    image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

                # show our detected faces along with smiling/not smiling labels
                p.stdin.write(image.tostring())
                cv2.imshow("Checking Strangers and Ole People's Face Expression",
                           image)

                # Press 'ESC' for exiting video
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break


def run():
    # 使用两个线程处理

    thread1 = Thread(target=Video, )

    thread1.start()
    time.sleep(2)

    thread2 = Thread(target=push_frame, )

    thread2.start()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()


