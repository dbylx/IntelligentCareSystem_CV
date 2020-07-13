# -*- coding: utf-8 -*-
'''
将事件插入数据库主程序

用法：

'''
import requests
import json
import datetime
import argparse

from urllib3 import encode_multipart_formdata

f = open('allowinsertdatabase.txt','r')
content = f.read()
f.close()
allow = content[11:12]

if allow == '1': # 如果允许插入
    
    f = open('allowinsertdatabase.txt','w')
    f.write('is_allowed=0')
    f.close()
    
    print('准备插入数据库')
    
    # 传入参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-ed", "--event_desc", required=False, 
                    default = '', help="")
    ap.add_argument("-et", "--event_type", required=False, 
                    default = '', help="")
    ap.add_argument("-el", "--event_location", required=False, 
                    default = '', help="")
    ap.add_argument("-epi", "--old_people_id", required=False, 
                    default = '', help="")
    ap.add_argument("-egg", "--pic_path", required=False,
                    default='', help="")

    args = vars(ap.parse_args())
    
    event_desc = args['event_desc']
    event_type = int(args['event_type']) if args['event_type'] else None
    event_location = args['event_location']
    old_people_id = int(args['old_people_id']) if args['old_people_id'] else None
    pic_path = (args['pic_path']) if args['pic_path'] else None
    print(pic_path)
    event_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    payload = {
               'eventDesc':event_desc,
               'envetType':event_type,
               'eventDate':event_date,
               'eventLocation':event_location,
               'oldPerson':old_people_id}
    
    print('调用插入事件数据的API')

    url = "http://123.56.92.168:8888/event/add"
    headers = {'Content-Type': 'application/json'}
    print(url)
    print(payload, u'数据类型:', type(payload))
    print(json.dumps(payload))
    res = requests.put(url, headers=headers, data=json.dumps(payload))
    print(str(res.text))
    pic_map = json.loads(str(res.text))
    print("path is " + pic_path)
    print("coid is "+ pic_map['data'])

    files = {'picture': (pic_map['data'] + '.png', open(pic_path, 'rb').read(), 'false')}
    encode_data = encode_multipart_formdata(files)

    headers1 = {'content-type': encode_data[1], 'id': pic_map['data']}
    data = encode_data[0]

    res1 = requests.post('http://123.56.92.168:8888/event/picture', headers=headers1, data=data)
    print(res1.text)






    print('插入成功')
    
else:
    print('just pass')

