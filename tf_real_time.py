import os
import time
import requests
import subprocess

import cv2
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from setting_for_real_time import Global

G = Global

'''JonYonv 1943'''
'''2018-05-09 stable version'''
"""2018-05-15 fileIO"""
'''2015-05-16 two tf model, class Info'''
'''2018-05-18 get_post_result() request_post()'''
'''2018-06-17 use face_match_snap subprocess.Popen(), to cancel id2name_dict'''
'''2018-06-18 time_stamp, format_time'''
'''2018-06-18 real time, delete the web request'''


def queue_img_put(q_put, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))

    while True:
        is_opened, frame = cap.read()
        q_put.put(frame) if is_opened else None
        q_put.get() if q_put.qsize() > 1 else None


def queue_img_get(q_get, model_path, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO) if window_name else None

    os.mkdir(G.img_origin_dir) if not os.path.exists(G.img_origin_dir) else None  # check dir
    csv_path = os.path.join(G.img_origin_dir, '0_image_info_for_xml.csv')

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
        print(box_info[:, -2:], end='') if len(box_info) > 0 else print("||| box_info: Null", end='')
        print("||| Ave time: %0.2f" % (time1 - timer))

        img = draw_box(box_info, img)
        (cv2.imshow(window_name, img), cv2.waitKey(1)) if window_name else None

        '''image save'''
        time_now = time.time()
        img_name = "%s%s.jpg" % (time.strftime("%Y%m%d%H%M%S", time.localtime(time_now)), str(time_now % 1.0)[2:4])
        img_path = os.path.join(G.img_origin_dir, img_name)
        cv2.imwrite(img_path, origin_img)  # save origin image

        with open(csv_path, 'a+') as f:  # save the csv with the tf_result_information
            line = [img_name, ]
            for (box, score, label) in box_info:
                line.extend([label, str(score)])
                line.extend([str(number) for number in box])
            f.write("%s\n" % ','.join(line))


def tf_model(origin_img_q, result_img_q, model_path, gpu_memory_limit):
    timer = time.time()
    gpu_limit_rate = gpu_memory_limit / int(os.popen('nvidia-smi').readlines()[8].split('/')[2].split('MiB')[0])

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
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_limit_rate), )
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


def deal_with_tf_output_data(boxes, scores, label_id, label_dict):
    boxes, scores, label_id = np.squeeze(boxes), np.squeeze(scores), np.squeeze(label_id).astype(np.int32)

    box_info = []
    for (box, score, idx) in zip(boxes, scores, label_id):
        label = label_dict[idx] if idx in label_dict.keys() else 'N/A'
        if score > G.custom_score_dict.setdefault(label, G.default_score_thresh):
            box_info.append([box, score, label])

    '''sort by score'''
    box_info = np.array(box_info)
    box_info = box_info[np.argsort(box_info[:, 1], axis=0)] if len(box_info.shape) > 1 else []
    return box_info  # [box, score, label], box = (y_min, x_min, y_max, x_max), score = float(), label = str()


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


def run():
    mp.set_start_method(method='spawn')

    origin_img_q = mp.Queue(maxsize=2)
    result_img_q = mp.Queue(maxsize=4)

    processes = [
        mp.Process(target=queue_img_put, args=(origin_img_q, G.user_name, G.user_pwd, G.camera_ip_l[0])),
        mp.Process(target=tf_model, args=(origin_img_q, result_img_q, G.model_path, G.gpu_memory_limit)),
        mp.Process(target=queue_img_get, args=(result_img_q, G.model_path, G.gui_name)),
    ]

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run()
