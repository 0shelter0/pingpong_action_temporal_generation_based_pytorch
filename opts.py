import argparse
import time


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
   
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.001)#0.001
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=30)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--step_size',
        type=int,
        default=15)# orignal=7
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1) # default=0.1
    parser.add_argument(
        '--num_heads',
        type=int,
        default=4)
    parser.add_argument(
        '--att_layers',
        type=int,
        default=2) #

    parser.add_argument(
        '--d_model',
        type=int,
        default=512,
        help='this is for transformer encoder d_model.')
    
    parser.add_argument(
        '--d_ff',
        type=int,
        default=512,
        help='this is for hidden dimension of FFN in transformer encoder.')

    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="data/activitynet_annotations/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="data/activitynet_annotations/anet_anno_action.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="data/activitynet_feature_cuhk/")

    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=400)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.5)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.9)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="./output/evaluation_result.jpg")

    
    # ============xzq=============
    time_str=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name='BMN_%s'%(time_str)
    log_path='log/log_%s.txt'%exp_name
    parser.add_argument(
        '--log_path',
        type=str,
        default=log_path)

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=f'checkpoint-{time_str}')

    parser.add_argument(
        '--exp_info',
        type=str,
        default='')

    args = parser.parse_args()

    return args

