import cv2
import queue
import os
import numpy as np
from threading import Thread
import datetime,_thread
import subprocess as sp
import time
from oldcare.facial.faceutildlib import FaceUtil
from oldcare.facial import FaceUtil
from scipy.spatial import distance as dist
from oldcare.utils import fileassistant
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import imutils
import numpy as np
import argparse
import subprocess


limit_time = 2
# 全局变量






mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()

rtmpUrl = "rtmp://39.97.107.19:1935/rtmplive"



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
    camera_turned = 0
    curr_fps = 0

    fps = "FPS: ??"

 #   prev_time = time()

    output_activity_path = 'supervision/activity'
    model_path = 'models/face_recognition_hog.pickle'
    people_info_path = 'info/people_info.csv'

    # your python path
    python_path = '/root/anaconda3/envs/tensorflow/bin/python'

    # 全局常量
    FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480
    ANGLE = 20
    ACTUAL_DISTANCE_LIMIT = 100  # cm

    # 得到 ID->姓名的map 、 ID->职位类型的map
    id_card_to_name, id_card_to_type = fileassistant.get_people_info(
        people_info_path)
    faceutil = FaceUtil(model_path)

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
                image = imutils.resize(image,
                                       width=VIDEO_WIDTH,
                                       height=VIDEO_HEIGHT)  # 压缩，加快识别速度

                face_location_list, names = faceutil.get_face_location_and_name(
                    image)

                # 得到画面的四分之一位置和四分之三位置，并垂直划线
                one_sixth_image_center = (int(VIDEO_WIDTH / 6), int(VIDEO_HEIGHT / 6))
                five_sixth_image_center = (int(VIDEO_WIDTH / 6 * 5),
                                           int(VIDEO_HEIGHT / 6 * 5))

                cv2.line(image, (one_sixth_image_center[0], 0),
                         (one_sixth_image_center[0], VIDEO_HEIGHT),
                         (0, 255, 255), 1)
                cv2.line(image, (five_sixth_image_center[0], 0),
                         (five_sixth_image_center[0], VIDEO_HEIGHT),
                         (0, 255, 255), 1)

                people_type_list = list(set([id_card_to_type[i] for i in names]))

                volunteer_name_direction_dict = {}
                volunteer_centroids = []
                old_people_centroids = []
                old_people_name = []

                # loop over the face bounding boxes
                for ((left, top, right, bottom), name) in zip(face_location_list,
                                                              names):  # 处理单个人

                    person_type = id_card_to_type[name]
                    # 将人脸框出来
                    rectangle_color = (0, 0, 255)
                    if person_type == 'old_people':
                        rectangle_color = (0, 0, 128)
                    elif person_type == 'employee':
                        rectangle_color = (255, 0, 0)
                    elif person_type == 'volunteer':
                        rectangle_color = (0, 255, 0)
                    else:
                        pass
                    cv2.rectangle(image, (left, top), (right, bottom),
                                  rectangle_color, 2)

                    if 'volunteer' not in people_type_list:  # 如果无义工则跳出循环
                        continue

                    if person_type == 'volunteer':  # 如果检测到有义工存在
                        # 获得义工位置
                        volunteer_face_center = (int((right + left) / 2),
                                                 int((top + bottom) / 2))
                        volunteer_centroids.append(volunteer_face_center)

                        cv2.circle(image,
                                   (volunteer_face_center[0],
                                    volunteer_face_center[1]),
                                   8, (255, 0, 0), -1)

                        adjust_direction = ''
                        # face locates too left, servo need to turn right,
                        # so that face turn right as well
                        if volunteer_face_center[0] < one_sixth_image_center[0]:
                            adjust_direction = 'right'
                        elif volunteer_face_center[0] > five_sixth_image_center[0]:
                            adjust_direction = 'left'

                        volunteer_name_direction_dict[name] = adjust_direction

                    elif person_type == 'old_people':  # 如果没有发现义工
                        old_people_face_center = (int((right + left) / 2),
                                                  int((top + bottom) / 2))
                        old_people_centroids.append(old_people_face_center)
                        old_people_name.append(name)

                        cv2.circle(image,
                                   (old_people_face_center[0],
                                    old_people_face_center[1]),
                                   4, (0, 255, 0), -1)
                    else:
                        pass

                    # 把人名写上 (同时处理中文显示问题)
                    img_PIL = Image.fromarray(cv2.cvtColor(image,
                                                           cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_PIL)
                    final_label = id_card_to_name[name]
                    draw.text((left, top - 30), final_label,
                              font=ImageFont.truetype('NotoSansCJK-Black.ttc',
                                                      40), fill=(255, 0, 0))  # linux
                    # 转换回OpenCV格式
                    image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

                # 义工追踪逻辑
                if 'volunteer' in people_type_list:
                    volunteer_adjust_direction_list = list(
                        volunteer_name_direction_dict.values())
                    # 有的义工恰好在范围内，所以不需要调整舵机
                    if '' in volunteer_adjust_direction_list:
                        print('%d-有义工恰好在可见范围内，摄像头不需要转动'
                              % (counter))
                    else:
                        adjust_direction = volunteer_adjust_direction_list[0]
                        camera_turned = 1
                        print('%d-摄像头需要 turn %s %d 度' % (counter,
                                                         adjust_direction, ANGLE))

                # 在义工和老人之间划线
                if camera_turned == 0:
                    for i in volunteer_centroids:
                        for j_index, j in enumerate(old_people_centroids):
                            pixel_distance = dist.euclidean(i, j)
                            face_pixel_width = sum([i[2] - i[0] for i
                                                    in face_location_list]) / len(face_location_list)
                            pixel_per_metric = face_pixel_width / FACE_ACTUAL_WIDTH
                            actual_distance = pixel_distance / pixel_per_metric

                            if actual_distance < ACTUAL_DISTANCE_LIMIT:
                                cv2.line(image, (int(i[0]), int(i[1])),
                                         (int(j[0]), int(j[1])), (255, 0, 255), 2)
                                label = 'distance: %dcm' % (actual_distance)
                                cv2.putText(image, label, (image.shape[1] - 150, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 2)

                                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                             time.localtime(time.time()))
                                event_desc = '%s正在与义工交互' % (id_card_to_name[old_people_name[j_index]])
                                event_location = '房间桌子'
                                print(
                                    '[EVENT] %s, 房间桌子, %s 正在与义工交互.' % (current_time, id_card_to_name[old_people_name[j_index]]))
                                cv2.imwrite(os.path.join(output_activity_path,
                                                         'snapshot_%s.jpg'
                                                         % (time.strftime('%Y%m%d_%H%M%S'))), image)
                                pic_path = os.path.join(output_activity_path,'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S')))
                                # insert into database
                                command1 = '%s inserting.py --event_desc %s --event_type 1 --event_location %s --old_people_id  %d --pic_path %s' % (
                                python_path, event_desc, event_location, int(name),pic_path)
                                p2 = subprocess.Popen(command1, shell=True)

                            # show our detected faces along with smiling/not smiling labels

                p.stdin.write(image.tostring())
                cv2.imshow("Checking Volunteer's Activities", image)

                # Press 'ESC' for exiting video
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break



 

def run():

     #使用两个线程处理

 

    thread1 = Thread(target=Video,)

    thread1.start()

    thread2 = Thread(target=push_frame,)

    thread2.start()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()


