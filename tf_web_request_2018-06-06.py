import os
import re
import time
import shutil

import cv2
import ctypes
import requests
import numpy as np
import tensorflow as tf
import multiprocessing as mp

'''JonYonv 1943'''
'''2018-05-09 stable version'''
"""2018-05-15 fileIO"""
'''2015-05-16 two tf model, class Info'''
'''2018-05-18 1201 get_post_result() request_post()'''


class Info(object):
    gpu_memory_limit = 0.6
    show_gui = True
    delete = False  # delete or move

    user_name, user_pwd = "admin", "!QAZ2wsx3edc"
    camera_ip_l = [
        # "192.168.1.64",
        # "192.168.1.164",
        # "192.168.1.165",

        # "192.168.1.166",
        # "192.168.1.167",
        # "192.168.1.168",
        "192.168.1.169",
        # "192.168.1.170",
    ]

    model_path_l = [
        # 'hat_worker_rcnn_detection_graph_16801_0520',
        # 'hat_worker_rcnn_detection_graph_16801_0523'
        'hat_worker_rcnn_detection_graph_16801_0524'
    ]

    img_result_dir = os.path.join(os.getcwd(), "tensorflow_result")
    img_origin_dir = os.path.join(os.getcwd(), "tensorflow_origin")
    img_backup_dir = os.path.join(os.getcwd(), "tensorflow_backup")
    csv_pwd = os.path.join(img_origin_dir, '0_image_info_for_xml.csv')

    custom_score_dict = {  # min_score belong to area[0.5, 1.0]
        'people': 0.55,
    }

    labels_pass_dict = {
        "worker_ok": 1,
        "cloth_r": 1,
        "cloth_t2_r": 1,
        "worker_t1_ok": 1,
        "worker_t2_ok": 1,

        'safety_hat_on': 1,
        "people_in_hat": 1,
        'safety_hat_ok': 1,
        'hat': 1,
    }

    labels_hat_dict = {
        'idx_to_str_dict': {0: "安全帽佩戴错误",
                            1: "安全帽佩戴正确",
                            2: "", },

        # "people": 0,
        # "people_in_hat_w": 0,
        # "people_in_n_hat": 0,
        # 'safety_hat_alter': 0,
        "head": 0,

        # "safety_hat_on": 1,
        # "people_in_hat": 1,
        'safety_hat_ok': 1,
    }

    labels_cloth_dict = {  # labels_suit_dict
        'idx_to_str_dict': {0: "工作服着装错误",
                            1: "工作服着装正确",
                            2: "", },

        "worker_t1_alert": 0,
        "worker_t2_alert": 0,
        "worker_alert": 0,
        #
        "worker_t1_ok": 1,
        "worker_t2_ok": 1,
        "worker_ok": 1,
    }


info = Info()


def get_gpu_memory_usage():
    line = os.popen('nvidia-smi').readlines()[8]
    line = line.split('|')[2]
    gpu_usage, gpu_tota = re.sub('\D', ' ', line).split()
    gpu_usage_rate = int(gpu_usage) / int(gpu_tota)
    return gpu_usage_rate


def face_match_snap():
    previous_working_directory = os.getcwd()
    os.chdir("/home/cb/pycode/tooth/HK_SDK/psdatacall_demo")
    h_dll = ctypes.cdll.LoadLibrary("/home/cb/pycode/tooth/HK_SDK/psdatacall_demo/getpsdata.so")
    h_dll.FaceDetectAndContrast(b"192.168.1.164", 8000, b"admin", b"!QAZ2wsx3edc")
    os.chdir(previous_working_directory)
    print("||| face snap ready")

    id2name_dict = {
        "00000000": "未知人员",
        "11111234": "郑佳豪",
        "22221234": "邱基盛",
    }

    '''establish environment'''
    face_snap_dir = "/home/cb/pycode/tooth/HK_SDK/psdatacall_demo/face_snap"
    os.mkdir(face_snap_dir) if not os.path.exists(face_snap_dir) else None

    face_snap_backup_dir = face_snap_dir + '_backup'
    shutil.rmtree(face_snap_backup_dir, ignore_errors=True)  # remove backup dir
    os.mkdir(face_snap_backup_dir) if not os.path.exists(face_snap_backup_dir) else None

    face_snap_file = None
    while True:
        time.sleep(0.1)  # directory scan gaps

        '''post'''
        face_snap_dir_l = [os.path.join(face_snap_dir, f) for f in os.listdir(face_snap_dir)]

        if len(face_snap_dir_l) == 0:  # check
            time.sleep(1)
            continue

        face_snap_dir_l.sort(key=lambda f: os.path.getmtime(f))  # sort by file time

        if time.time() - os.path.getmtime(face_snap_dir_l[-1]) > 4:  # avoid to report the old image
            time.sleep(1)
            continue

        try:
            if face_snap_file != face_snap_dir_l[-1]:  # new image?
                face_snap_file = face_snap_dir_l[-1]
                _time, face_id, score = face_snap_file[:-4].split('-')
                time_stamp = os.path.getmtime(os.path.join(face_snap_dir, face_snap_file))
                format_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time_stamp))

                data = {
                    "name": id2name_dict[face_id],
                    "score": score[2:4],
                    "datetime": format_time,
                    "authorize": "未授权" if face_id == '00000000' else "已授权",  # id '00000000' means stranger
                    "image_path": os.path.join(face_snap_dir, face_snap_file),
                }
                requests.post('http://0.0.0.0:8008/faceresult', json=data)
        except Exception as error:
            print("|||", error)
            time.sleep(1)

        '''auto move'''
        if len(face_snap_dir_l) > 64:
            [shutil.move(f, face_snap_backup_dir) for f in face_snap_dir_l[:-16]]  # dir_l had sorted


def draw_box(box_info, img):
    y_len, x_len = img.shape[:2]

    for box, score, label in box_info:

        '''draw rectangles'''
        y_min, x_min, y_max, x_max = box
        pt1 = (int(x_min * x_len), int(y_min * y_len))
        pt2 = (int(x_max * x_len), int(y_max * y_len))

        if label not in info.labels_pass_dict:
            cv2.rectangle(img, pt1, pt2, (0, 0, 255), thickness=8)
        # else:  # for debug, show ban label
        #     cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=8)
    return img


def video_capture(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    # cap = cv2.VideoCapture("test.mp4")

    is_opened, frame = cap.read()

    '''loop'''
    while is_opened:
        is_opened, frame = cap.read()
        frame = cv2.resize(frame, (704, 432)) if channel == 3 else frame
        q.put([is_opened, frame])
        while q.qsize() >= 2:
            q.get()

        # cap.read()
        # cap.read()
        # q.put([is_opened, frame])
        # while q.qsize() > 1:
        #     time.sleep(0.1)



def id_to_label_dict(pbtxt_pwd):
    d = {}
    with open(pbtxt_pwd, 'r') as f:
        line = f.readline()
        while line:
            if line == "item {\n":
                key = int(f.readline().split(':')[-1])
                value = f.readline().split(':')[-1][2:-2]
                d[key] = value
            line = f.readline()
    return d


def tf_model(q_img_l, q_data, model_pwd, q_lock):
    timer = time.time()

    '''loading the TensorFlow model'''
    ckpt_pwd = os.path.join(model_pwd, 'frozen_inference_graph.pb')

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
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=info.gpu_memory_limit), )
        with tf.Session(graph=detection_graph, config=tf_config) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

            print("||| Loading TF model time:", time.time() - timer)

            is_opened = True
            while is_opened:
                for camera_num, q_img in enumerate(q_img_l):

                    # '''custom lock, keep the two model get the same image'''
                    # frame, model_pwd_get = None, None
                    # if model_pwd == info.model_path_l[0]:
                    #     while q_lock.qsize() != 0:
                    #         time.sleep(0.1)
                    #     else:
                    #         (is_opened, frame) = q_img.get()
                    #         q_lock.put(frame)
                    # elif model_pwd == info.model_path_l[1]:
                    #     while q_lock.qsize() != 1:
                    #         time.sleep(0.1)
                    #     else:
                    #         frame = q_lock.get()

                    while q_img.qsize() == 0:
                        time.sleep(0.1)
                    (is_opened, frame) = q_img.get()  # one tf model

                    image = np.copy(frame)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    (boxes, score, label_id, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image[np.newaxis, :, :, :]},
                    )

                    q_data.put((frame, label_id, score, boxes, camera_num))


def deal_with_tf_output_data(boxes, scores, label_id, label_dict, score_thresh=0.8, max_boxes_to_draw=32, ):
    """
    :return: [box, score, label]
    box = (y_min, x_min, y_max, x_max), score = float(), label = str()
    """
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    label_id = np.squeeze(label_id).astype(np.int32)

    box_info = []
    for (box, score, idx) in zip(boxes, scores, label_id):
        label = label_dict[idx] if idx in label_dict.keys() else 'N/A'
        if score > info.custom_score_dict.setdefault(label, score_thresh):
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


def get_post_result(box_info):
    labels = set(box_info[:, 2])

    hat_result_idx = 2  # 0: False(wrong), 1:True(pass), 2:Null(nothing)
    if labels & info.labels_hat_dict.keys():
        for label in labels:
            if label in info.labels_hat_dict.keys():
                hat_result_idx = info.labels_hat_dict[label]
                if hat_result_idx == 0:
                    break

    cloth_result_idx = 2  # 0: False(wrong), 1:True(pass), 2:Null(nothing)
    if labels & info.labels_cloth_dict.keys():
        for label in labels:
            if label in info.labels_cloth_dict.keys():
                cloth_result_idx = info.labels_cloth_dict[label]
                if cloth_result_idx == 0:
                    break
    return hat_result_idx, cloth_result_idx


def request_post(hat_result_idx, cloth_result_idx, img_time, img_pwd, camera_num):
    try:

        hat_result = info.labels_hat_dict['idx_to_str_dict'][hat_result_idx]
        cloth_result = info.labels_cloth_dict['idx_to_str_dict'][cloth_result_idx]

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


def tf_data(q_data_l, model_pwd_l, show_gui):
    label_dict_l = []
    for model_path in model_pwd_l:
        print("||| Load TensorFlow Model:", model_path)
        pbtxt_pwd = [f for f in os.listdir(model_path) if f.find('.pbtxt') >= 0][0]
        labels_pwd = os.path.join(model_path, pbtxt_pwd)
        label_dict = id_to_label_dict(labels_pwd)
        label_dict_l.append(label_dict)

    cv2.namedWindow(tf_data.__name__, flags=cv2.WINDOW_FREERATIO) if show_gui else None

    while not os.path.exists(info.csv_pwd):
        time.sleep(0.1)

    timer = time.time()
    while True:
        box_info_all = []
        # box_info_all = [(0, 0, 0, 0), 0, 'Null']
        frame = []
        camera_num = None
        for (q_data, label_dict) in zip(q_data_l, label_dict_l):
            (frame, label_id, score, boxes, camera_num) = q_data.get()
            # cv2.imshow('test', frame)
            # cv2.waitKey(1)

            box_info = deal_with_tf_output_data(boxes, score, label_id, label_dict)
            for item in box_info:
                box_info_all.append(item)

        if len(box_info_all) == 0:
            timer = time.time()
            img_draw = frame
            print("||| Info: tf_data() len(box_info_all) is 0")
        else:
            box_info_all = np.array(box_info_all)

            img_draw = np.copy(frame)
            img_draw = draw_box(box_info_all, img_draw)

            '''image save'''
            time_now = time.time()
            img_format_time = "%s%s" % (time.strftime("%Y%m%d%H%M%S", time.localtime(time_now)),
                                        str(time_now % 1.0)[2:4])

            hat_result_idx, cloth_result_idx = get_post_result(box_info_all)
            img_name = "%s-%s-%s.jpg" % (img_format_time, hat_result_idx, cloth_result_idx)

            save_pwd = os.path.join(info.img_result_dir, img_name)
            cv2.imwrite(save_pwd, img_draw)  # save TF result image

            save_org = os.path.join(info.img_origin_dir, img_name)
            cv2.imwrite(save_org, frame)  # save origin image

            with open(info.csv_pwd, 'a+') as f:  # save the csv
                # line = ["%s.jpg" % img_format_time]
                line = [img_name, ]
                for (box, score, label) in box_info_all:
                    line.extend([label, str(score)])
                    line.extend([str(number) for number in box])
                f.write("%s\n" % ','.join(line))


            '''web requests post'''
            request_post(hat_result_idx, cloth_result_idx, img_format_time, save_pwd, camera_num)

            '''report'''
            print("||| Ave time: %0.2f" % (time.time() - timer))
            print(box_info_all[:, 1:3], end='\n\n') if box_info_all is not None else print("||| box_info: NULL")
            timer = time.time()

        (cv2.imshow(tf_data.__name__, img_draw), cv2.waitKey(1)) if show_gui else None


def dir_manager(delete, min_file=16, max_file=64):
    dir_manager_l = [
        info.img_result_dir,
        info.img_origin_dir,
        # info.img_backup_dir,
        info.csv_pwd,
    ]

    for dirt in dir_manager_l:
        if os.path.exists(dirt):
            print("||| dirt exists:", dirt)
        else:
            os.mkdir(dirt) if dirt.find('.') < 0 else os.mknod(dirt)
            print("||| mkdir or mkfifo:", dirt)

    print("||| %s: Delete file" % dir_manager.__name__) if delete \
        else print("||| %s: Move to *_backup" % dir_manager.__name__)
    while True:
        time.sleep(1943)
        # dir_l = [f for f in os.listdir(info.img_result_dir) if f[-4:] == '.jpg']
        # if len(dir_l) > max_file:
        #     file_l = [os.path.join(info.img_result_dir, f) for f in sorted(dir_l)[:-min_file]]
        #     [os.remove(f) for f in file_l] if delete \
        #         else [shutil.move(os.path.join(info.img_result_dir, f),
        #                           os.path.join(info.img_backup_dir, f)) for f in file_l]
        #
        #     time.sleep(4)


def main():
    mp.set_start_method(method='spawn')

    queue_data_l = [mp.Queue(maxsize=16) for _ in info.model_path_l]
    queue_img_l = [mp.Queue(maxsize=16) for _ in info.camera_ip_l]
    queue_lock = mp.Queue(maxsize=1)

    process_l = [
        mp.Process(target=dir_manager, args=(info.delete,)),
        mp.Process(target=face_match_snap, args=()),
        mp.Process(target=tf_data, args=(queue_data_l, info.model_path_l, info.show_gui)),
    ]
    process_l.extend([mp.Process(target=video_capture, args=(queue, info.user_name, info.user_pwd, cam_ip))
                      for (queue, cam_ip) in zip(queue_img_l, info.camera_ip_l)])
    process_l.extend([mp.Process(target=tf_model, args=(queue_img_l, queue_data, model_path, queue_lock))
                      for (queue_data, model_path) in zip(queue_data_l, info.model_path_l)])

    for process in process_l:
        process.start()
    for process in process_l:
        process.join()
    pass


if __name__ == '__main__':
    main()
pass
