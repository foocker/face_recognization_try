import argparse
from configs.config_arc import get_config
from mtcnn import MTCNN
from apis.learner import face_learner
from utils.utils_arc import load_facebank, prepare_facebank
from data.prepare_valdata import get_val_dataset
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face evaluation')
    parser.add_argument("-p", "--valimg_dir", help="imgs dir for evaluation ./XX/val/", default='./data/faces_emore/val/', type=str)
    parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")

    args = parser.parse_args()

    conf = get_config(False)
    conf.use_mobilfacenet = True

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', False, True)
    else:
        learner.load_state(conf, '0.91145.pth', False, True)
    learner.model.eval()
    print('learner loaded')

    if args.update:
        print("begin update!")
        st = time.time()
        # for path in conf.facebank_path.iterdir():
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
        print('facebank updated', "cost time is :", time.time() - st)
    else:
        print("begin load")
        targets, names = load_facebank(conf)
        print('facebank loaded')
    
    print("beginning get val dataset")
    val_data, val_label = get_val_dataset(args.valimg_dir)
    print("got val dataset", val_data.shape, val_label.shape)
    
    
    print("begin evaluate on val dataset!")
    results, score = learner.infer(conf, faces, targets, tta=True)