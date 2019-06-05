# encoding:utf-8

class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        self.backbone = 'resnet18'
        self.class_num = 20
        self.fpn_config = dict(
            classes_num=20, 
            anchor_num=3, 
            middle_channel=256,
            use_pyramid_iter=[2, 3, 4], 
            upsample_scale=2, 
            upsample_mode='nearest'
        )
        self.yolo_layer_config_list = [
            dict(
                stride=8,
                anchor_list=[[10,13],  [16,30],  [33,23]],
                grid_num=[52, 52],
                iou_thresh=0.125,
                device='cuda:0'

            ),
            dict(
                stride=16,
                anchor_list=[[30,61],  [62,45],  [59,119]],
                grid_num=[26, 26],
                iou_thresh=0.125,
                device='cuda:0'

            ),
            dict(
                stride=32,
                anchor_list=[[116,90],  [156,198],  [373,326]],
                grid_num=[13, 13],
                iou_thresh=0.125,
                device='cuda:0'

            ),
        ]

        self.train_config = dict(
            lr_adjust_map= {
                1: 0.001,
                85: 0.0001,
                135: 0.00001
            },

            lbd_map = {
                'k': 10.39,  # loss multiple
                'xy': 0.1367,  # xy loss fraction
                'wh': 0.01057,  # wh loss fraction
                'cls': 0.01181,  # cls loss fraction
                'conf': 0.8409,  # conf loss fraction
                'lr0': 0.001028,  # initial learning rate
                'lrf': -3.441,  # final learning rate = lr0 * (10 ** lrf)
                'momentum': 0.9127,  # SGD momentum
                'weight_decay': 0.0004841,  # optimizer weight decay
            },


            gpu_ids= [0],
            worker_num=4,
            batch_size=16,
            epoch_num=1500,
            resume_from_path=None,
            resume_epoch=0,
            train_txt_path="datasets/little_train.txt",
            test_txt_path="datasets/2007_test.txt",
            log_name="trainLog",
            base_save_path="/home/ubuntu/project/YOLO_V3/save_weights/resnet18_results",
            vis_log_path="/home/ubuntu/project/YOLO_V3/save_weights/resnet18_results/vis_log.log",

        )

        self.eval_config = dict(
            gpu_ids= [0],
            worker_num=0,
            batch_size=1,
            test_txt_path="datasets/train.txt",
            load_from_path="save_weights/resnet18_results/resnet18_last.pth"
        )

