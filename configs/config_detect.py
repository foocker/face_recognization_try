# config_arc.py

cfg = {
    'name': 'FaceBoxes',
    #'min_dim': 1024,
    #'feature_maps': [[32, 32], [16, 16], [8, 8]],
    # 'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'loc_five_weight': 4.0,
    'gpu_train': True,
    'confidence_threshold': 0.05,
    'nms_threshold': 0.3,
    'top_k':3000,
    'keep_top_k': 400,
    'trained_model':'./weights/Final_FaceBoxes.pth',
    'v_path': './data/videos/Friendsone.mp4',
    'sv_path': './data/videos/Friendsold_detect_coords_48.avi',
    'Onet_weights':'./weights/Onet_140.pt'
    
}
