class Global(object):
    gui_name = "Safety Suit 0524"  # None means not GUI
    gpu_memory_limit = 4096  # MiB, min GPU memory may be 1.5G, but 1.5G would impact the detection effect
    show_green_box = True  # debug
    delete = False  # delete or move

    model_path = 'hat_worker_rcnn_detection_graph_16801_0524'

    img_origin_dir = "tf_origin"
    img_result_dir = "tf_result"
    img_backup_dir = "tf_backup"
    img_error_dir = "tf_errors"

    labels_pass_dict = {
        'safety_hat_ok': 1,
        "worker_t2_ok": 1,
    }

    labels_hat_dict = {
        'idx_to_str_dict': {0: "安全帽佩戴错误",
                            1: "安全帽佩戴正确",
                            2: "", },
        "head": 0,
        'safety_hat_alter': 0,
        'safety_hat_ok': 1,
    }

    labels_cloth_dict = {  # labels_suit_dict
        'idx_to_str_dict': {0: "工作服着装错误",
                            1: "工作服着装正确",
                            2: "", },
        'worker_alert': 0,
        "worker_t2_alert": 0,
        "worker_t2_ok": 1,
    }

    default_score_thresh = 0.8
    custom_score_dict = {  # min_score belong to area[0.5, 1.0]
        'safety_hat_alter': 0.9,
    }

    user_name, user_pwd = "admin", "password"
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
