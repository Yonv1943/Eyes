import os
import time
import shutil

# import cv2
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

import numpy as np
import tensorflow as tf
import multiprocessing as mp
from setting_for_web_request import Global

'''JonYonv 1943'''
'''2018-05-09 stable version'''
"""2018-05-15 fileIO"""
'''2015-05-16 two tf model, class Info'''
'''2018-05-23 Daemon'''

G = Global()


def tf_sess(q_img, q_res, model_pwd):
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
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=G.gpu_memory_limit), )
        with tf.Session(graph=detection_graph, config=tf_config) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

            print("||| Loading TF model time:", time.time() - timer)

            while True:
                while q_img.qsize() == 0:
                    time.sleep(0.1)
                (img, img_name) = q_img.get()  # one tf model

                # image = np.copy(img)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.asarray(img, dtype=np.uint8)

                (boxes, score, label_id, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image[np.newaxis, :, :, :]},
                )

                q_res.put((img, img_name, label_id, score, boxes))


def tf_data(q_res, q_fun, model_pwd, show_gui):
    # cv2.namedWindow(tf_data.__name__, flags=cv2.WINDOW_FREERATIO) if show_gui else None

    print("||| Load TensorFlow Model:", model_pwd)
    pbtxt_name = [f for f in os.listdir(model_pwd) if f.find('.pbtxt') >= 0][0]
    pbtxt_pwd = os.path.join(model_pwd, pbtxt_name)

    label_dict = {}  # get label_dict
    with open(pbtxt_pwd, 'r') as f:
        line = f.readline()
        while line:
            if line == "item {\n":
                key = int(f.readline().split(':')[-1])
                value = f.readline().split(':')[-1][2:-2]
                label_dict[key] = value
            line = f.readline()

    while True:
        (img, img_name, label_id, score, boxes) = q_res.get()

        box_info = []  # [label_name, score, x_min, y_min, x_max, y_max]
        box_info.extend(get_box_info(boxes, score, label_id, label_dict))
        box_info = np.array(box_info)

        # img_draw = np.copy(img)
        # img_draw = draw_box(box_info, img_draw)
        img_draw = draw_box(box_info, img)

        hat_result_idx, cloth_result_idx = get_post_result(box_info)
        img_name_with_label = "%s_%s_%s.jpg" % (img_name[:-4], hat_result_idx, cloth_result_idx)

        # q_fun.put([cv2.imwrite, (os.path.join(G.img_result_dir, img_name_with_label), img_draw)])  # save result_image
        q_fun.put([img_draw.save, (os.path.join(G.img_result_dir, img_name_with_label),)])  # save result_image

        q_fun.put([shutil.move, (os.path.join(G.img_origin_dir, img_name),
                                 os.path.join(G.img_backup_dir, img_name))])  # move origin_image
        # q_fun.put([os.remove, (os.path.join(G.img_origin_dir, img_name))])  # remove origin_image

        # (cv2.imshow(tf_data.__name__, img_draw), cv2.waitKey(1)) if show_gui else None
        print("||| Info:", box_info[:, 0:2].flatten() if len(box_info) != 0 else [])


def get_box_info(boxes, scores, label_id, label_dict, score_thresh=0.8, max_boxes_to_draw=32, ):
    """
    boxes: [(y_min, x_min, y_max, x_max), ...]
    :return: np.array([label_name, score, y_min, x_min, y_max, x_max], dtype=np.str)
    """
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    label_id = np.squeeze(label_id).astype(np.int32)

    box_info = []
    for (box, score, idx) in zip(boxes, scores, label_id):
        y_min, x_min, y_max, x_max = box
        label = label_dict[idx] if idx in label_dict.keys() else 'N/A'
        if score > G.custom_score_dict.setdefault(label, score_thresh):
            box_info.append([label, score, y_min, x_min, y_max, x_max])

    max_len = min(max_boxes_to_draw, len(boxes))

    '''sort by score'''
    box_info = np.array(box_info)
    argsort_key = box_info[:, 1] if len(box_info) != 0 else []
    box_info = box_info[np.argsort(argsort_key, axis=0)]
    box_info = box_info[:max_len].tolist()
    return box_info


def get_post_result(box_info):
    labels = set(box_info[:, 2]) if len(box_info) != 0 else set()

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


def draw_box(box_info, img):
    # y_len, x_len = img.shape[:2]
    x_len, y_len = img.size

    for i, (label, score, y_min, x_min, y_max, x_max) in enumerate(box_info):
        y_min, x_min, y_max, x_max = [float(s) for s in (y_min, x_min, y_max, x_max)]
        '''draw rectangles'''
        pt1 = (int(x_min * x_len), int(y_min * y_len))
        pt2 = (int(x_max * x_len), int(y_max * y_len))

        img_draw = ImageDraw.Draw(img)
        if label not in G.labels_pass_dict:
            # cv2.rectangle(img, pt1, pt2, (0, 0, 255), thickness=8)
            img_draw.rectangle([pt1, pt2], fill=None, outline=(255, 0, 0))
        else:  # for debug, show ban label
            # cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=8)
            img_draw.rectangle([pt1, pt2], fill=None, outline=(0, 255, 0))
    return img


def io_task(q_img, q_fun, src):
    pwd_l = [
        G.img_origin_dir,
        G.img_result_dir,
        G.img_backup_dir,
        G.img_error_dir,
    ]

    [os.makedirs(pwd) for pwd in pwd_l if not os.path.exists(pwd)]

    file_oldest = None

    while True:
        file_l = os.listdir(src)
        file_l.sort(key=lambda f: os.path.getmtime(os.path.join(src, f)))  # sort by file time

        if len(file_l) != 0 and file_oldest != file_l[0]:
            file_oldest = file_l[0]

            img_name = file_oldest
            # img = cv2.imread(os.path.join(src, img_name))

            image_open = True
            img = None
            try:
                img = Image.open(os.path.join(src, img_name))
            except OSError:
                image_open = False

            # if not isinstance(img, np.ndarray):  # check image
            if not image_open:  # check image
                q_fun.put([shutil.move, (os.path.join(G.img_origin_dir, img_name),
                                         os.path.join(G.img_error_dir, img_name))])  # move origin_image
                print("||| Move Broken Image:", img_name)
                print("||| Type:", type(img))
            else:
                q_img.put([img, img_name])

        while q_fun.qsize() > 0:
            (fun, args) = q_fun.get()
            fun(*args) if isinstance(args, tuple) else fun(args)

        time.sleep(1)


def main():
    # Startup Application Preferences
    # python3 tf_model.py
    # [(time.sleep(1), print("||| Start after:", i)) for i in range(4, 0, -1)]

    mp.set_start_method(method='spawn')

    queue_res = mp.Queue(maxsize=16)  # result, (img, img_name,label_id, score, boxes)
    queue_img = mp.Queue(maxsize=16)  # image, (img, img_name)
    queue_fun = mp.Queue()  # function(args), (func, args)

    process_l = [
        mp.Process(target=io_task, args=(queue_img, queue_fun, G.img_origin_dir)),
        mp.Process(target=tf_sess, args=(queue_img, queue_res, G.model_path)),
        mp.Process(target=tf_data, args=(queue_res, queue_fun, G.model_path, G.gui_name)),
    ]

    # for process in process_l:
    #     process.daemon = True
    #     process.start()
    #
    # while True:
    #     [time.sleep(1) if process.is_alive() else process.start() for process in process_l]
    [p.start() for p in process_l]
    [p.join() for p in process_l]


if __name__ == '__main__':
    main()
pass
