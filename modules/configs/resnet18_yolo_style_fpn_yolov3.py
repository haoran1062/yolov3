# encoding:utf-8

class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        self.class_num = 20
        self.backbone_type='resnet18',
        self.backbone_config = dict(
            layer_st=2,
            layer_ed=5,
        ),
        self.fpn_config = dict(
            classes_num=self.class_num, 
            anchor_num=3,
            single_fpn_in_channels=[512, 1024, 512, 1024], 
            single_fpn_out_channels=[512, 1024, 512, 1024],
            kernel_sizes=[1, 3, 1, 3, 1],
            middle_channel=512,
            in_channels=[128, 256, 512], 
            upsample_scale=2, 
            upsample_mode='nearest'
        )
        self.yolo_layer_config_list = [
            
            dict(
                stride=32,
                class_num=20,
                obj_lbd=1,
                noobj_lbd=1,
                anchor_list=[[116,90],  [156,198],  [373,326]],
                iou_thresh=0.1287,
                device='cuda:0'

            ),
            dict(
                stride=16,
                class_num=20,
                obj_lbd=1,
                noobj_lbd=1,
                anchor_list=[[30,61],  [62,45],  [59,119]],
                iou_thresh=0.1287,
                device='cuda:0'

            ),
            dict(
                stride=8,
                class_num=20,
                obj_lbd=1,
                noobj_lbd=1,
                anchor_list=[[10,13],  [16,30],  [33,23]],
                iou_thresh=0.1287,
                device='cuda:0'

            ),
        ]

        self.train_config = dict(
            add_lr=0.00001*10,
            stop_down_iter=100//10,
            lr_adjust_map= {
                1: 0.001,
                185: 0.0001,
                335: 0.00001
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
            worker_num=0,
            batch_size=16,
            epoch_num=15000,
            show_img_iter_during_train=1,
            resume_from_path=None,
            resume_epoch=0,
            # train_txt_path="/data/datasets/yolo_txt/top300_truth_till_20190624.txt",
            train_txt_path="datasets/2007_train.txt",
            test_txt_path="datasets/2007_train.txt",
            log_name="trainLog",
            base_save_path="/data/temp/resnet18_results",
            vis_log_path="/data/temp/resnet18_results/vis_log.log",

        )

        self.eval_config = dict(
            gpu_ids= [0],
            worker_num=1,
            batch_size=32,
            test_txt_path="datasets/little_train.txt",
            load_from_path="/data/temp/resnet18_results/resnet18_last.pth"
        )

        self.name_path = 'datasets/voc.names'

