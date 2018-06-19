import os
import re
import time
import shutil
import requests
import subprocess

import cv2
import ctypes
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from setting_for_web_request import Global

G = Global

'''JonYonv 1943'''
'''2018-05-09 stable version'''
"""2018-05-15 fileIO"""
'''2015-05-16 two tf model, class Info'''
'''2018-05-18 1201 get_post_result() request_post()'''
'''2018-06-17 use face_match_snap subprocess.Popen(), to cancel id2name_dict'''
'''2018-06-18 time_stamp, format_time'''


def queue_img_put(q_put, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))

    while True:
        is_opened, frame = cap.read()
        q_put.put(frame) if is_opened else None
        q_put.get() if q_put.qsize() > 1 else None


def queue_img_get(q_get, model_path, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO) if window_name else None
    csv_path = os.path.join(G.img_origin_dir, '0_image_info_for_xml.csv')

    [os.mkdir(dirt) for dirt in (G.img_origin_dir, G.img_result_dir) if not os.path.exists(dirt)]  # check dir

    print("||| Load TensorFlow Model:", model_path)
    pbtxt_name = [f for f in os.listdir(model_path) if f.find('.pbtxt') >= 0][0]
    with open(os.path.join(model_path, pbtxt_name), 'r') as f:  # get label_dict
        lines = f.readlines()
        ids = [int(line.split(':')[-1]) for line in lines if line.find('id:') > 0]
        names = [line.split(':')[-1][2:-2] for line in lines if line.find('name:') > 0]
        label_dict = dict(zip(ids, names))
    print("||| label_dict:", label_dict)

    timer, time1 = time.time(), time.time()
    while True:
        (origin_img, label_id, score, boxes) = q_get.get()
        img = np.copy(origin_img)
        box_info = deal_with_tf_output_data(boxes, score, label_id, label_dict)

        '''report and draw'''
        timer, time1 = time1, time.time()
        print("||| Ave time: %0.2f" % (time1 - timer))
        if len(box_info) > 0:
            print(box_info[:, -2:])
            img = draw_box(box_info, img)
        else:
            print("||| box_info: Null")
        (cv2.imshow(window_name, img), cv2.waitKey(1)) if window_name else None

        '''image save'''
        time_now = time.time()
        img_format_time = "%s%s" % (time.strftime("%Y%m%d%H%M%S", time.localtime(time_now)),
                                    str(time_now % 1.0)[2:4])

        (hat_result_idx, cloth_result_idx) = get_post_result(box_info) if len(box_info) > 0 else (2, 2)
        img_name = "%s-%s-%s.jpg" % (img_format_time, hat_result_idx, cloth_result_idx)

        save_pwd = os.path.join(G.img_result_dir, img_name)
        cv2.imwrite(save_pwd, img)  # save TF result image

        save_org = os.path.join(G.img_origin_dir, img_name)
        cv2.imwrite(save_org, origin_img)  # save origin image

        with open(csv_path, 'a+') as f:  # save the csv
            # line = ["%s.jpg" % img_format_time]
            line = [img_name, ]
            for (box, score, label) in box_info:
                line.extend([label, str(score)])
                line.extend([str(number) for number in box])
            f.write("%s\n" % ','.join(line))

        '''web requests post'''
        request_post(hat_result_idx, cloth_result_idx, img_format_time, save_pwd)


def face_match_snap():
    previous_working_directory = os.getcwd()
    face_snap_cwd = "/home/cb/pycode/tooth/HK_SDK/psdatacall_demo"
    face_snap_dir = os.path.join(face_snap_cwd, "face_snap")
    os.chdir("/home/cb/pycode/tooth/HK_SDK/psdatacall_demo")
    out = subprocess.Popen("python3 face_match_snap.py", stdout=subprocess.PIPE, shell=True)
    os.chdir(previous_working_directory)

    print("||| face snap ready")
    while True:
        file_name = out.stdout.readline().decode('utf-8')

        if file_name.find('CPP match face:') == 0:
            file_name = file_name[len("CPP match face: "):-1]
            time_stamp, face_id, name, score = file_name.split('_')

            try:
                # format_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(int(time_stamp)))
                format_time = list(time_stamp)
                [format_time.insert(i, '-') for i in (12, 10, 8, 6, 4)]
                format_time = ''.join(format_time)

                data = {
                    "name": face_id, "score": score[2:4], "datetime": format_time,
                    "authorize": "未授权" if face_id == '00000000' else "已授权",  # id '00000000' means stranger
                    "image_path": os.path.join(face_snap_dir, file_name),
                }
                requests.post('http://0.0.0.0:8008/faceresult', json=data)
            except Exception as error:
                print("||| Error:", error)
                time.sleep(0.5)

            print("||| Face_match_snap:", time_stamp, face_id, name, score)
        else:
            print(file_name)


def tf_model(origin_img_q, result_img_q, model_path, gpu_memory_limit):
    timer = time.time()

    '''loading the TensorFlow model'''
    ckpt_pwd = os.path.join(model_path, 'frozen_inference_graph.pb')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_pwd, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    '''run the TensorFlow model'''
    with detection_graph.as_default():
        '''limit the GPU Memory'''
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_limit), )
        with tf.Session(graph=detection_graph, config=tf_config) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

            print("||| Loading TF model time:", time.time() - timer)

            is_opened = True
            while is_opened:
                while origin_img_q.qsize() == 0:
                    time.sleep(0.1)
                origin_img = origin_img_q.get()  # one tf model

                img = np.copy(origin_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                (boxes, score, label_id, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: img[np.newaxis, :, :, :]},
                )

                result_img_q.put((origin_img, label_id, score, boxes))


def deal_with_tf_output_data(boxes, scores, label_id, label_dict, score_thresh=0.8, max_boxes_to_draw=32, ):
    """
    :return: [box, score, label]
    box = (y_min, x_min, y_max, x_max), score = float(), label = str()
    """
    custom_score_dict = G.custom_score_dict

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    label_id = np.squeeze(label_id).astype(np.int32)

    box_info = []
    for (box, score, idx) in zip(boxes, scores, label_id):
        label = label_dict[idx] if idx in label_dict.keys() else 'N/A'
        if score > custom_score_dict.setdefault(label, score_thresh):
            box_info.append([box, score, label])

    max_len = min(max_boxes_to_draw, len(boxes))

    '''sort by score'''
    box_info = np.array(box_info)
    if len(box_info.shape) > 1:  # check
        argsort_key = box_info[:, 1]
        box_info = box_info[np.argsort(argsort_key, axis=0)]
        box_info = box_info[:max_len]
    else:
        box_info = []
    return box_info


def draw_box(box_info, img):
    y_len, x_len = img.shape[:2]

    for box, score, label in box_info:

        '''draw rectangles'''
        y_min, x_min, y_max, x_max = box
        pt1 = (int(x_min * x_len), int(y_min * y_len))
        pt2 = (int(x_max * x_len), int(y_max * y_len))

        if label not in G.labels_pass_dict:
            cv2.rectangle(img, pt1, pt2, (0, 0, 255), thickness=8)
        elif G.show_green_box:  # for debug, show ban label
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=8)
    return img


def request_post(hat_result_idx, cloth_result_idx, img_time, img_pwd):
    try:

        hat_result = G.labels_hat_dict['idx_to_str_dict'][hat_result_idx]
        cloth_result = G.labels_cloth_dict['idx_to_str_dict'][cloth_result_idx]

        print("||| hat:", hat_result_idx, 'cloth:', cloth_result_idx)
        data = {
            'hat_result': hat_result,
            'cloth_result': cloth_result,
            "datetime": img_time,
            "image_path": img_pwd,
        }

        requests.post('http://0.0.0.0:8008/cameraresult', json=data)
    except Exception as error:
        print("|||", error)


def get_post_result(box_info):
    labels = set(box_info[:, 2])

    hat_result_idx = 2  # 0: False(wrong), 1:True(pass), 2:Null(nothing)
    if labels & G.labels_hat_dict.keys():
        for label in labels:
            if label in G.labels_hat_dict.keys():
                hat_result_idx = G.labels_hat_dict[label]
                if hat_result_idx == 0:
                    break

    cloth_result_idx = 2  # 0: False(wrong), 1:True(pass), 2:Null(nothing)
    if labels & G.labels_cloth_dict.keys():
        for label in labels:
            if label in G.labels_cloth_dict.keys():
                cloth_result_idx = G.labels_cloth_dict[label]
                if cloth_result_idx == 0:
                    break
    return hat_result_idx, cloth_result_idx


def run():
    mp.set_start_method(method='spawn')

    origin_img_q = mp.Queue(maxsize=2)
    result_img_q = mp.Queue(maxsize=4)

    processes = [
        # mp.Process(target=face_match_snap, args=()),
        mp.Process(target=queue_img_put, args=(origin_img_q, G.user_name, G.user_pwd, G.camera_ip_l[0])),
        mp.Process(target=tf_model, args=(origin_img_q, result_img_q, G.model_path, G.gpu_memory_limit)),
        mp.Process(target=queue_img_get, args=(result_img_q, G.model_path, G.gui_name)),
    ]

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run()
