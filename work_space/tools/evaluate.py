import argparse
from configs.config_arc import get_config
from mtcnn import MTCNN
from apis.learner import face_learner
from utils.utils_arc import load_facebank, prepare_facebank
from data.prepare_valdata import get_val_dataset
# from utils.verifacation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face evaluation')
    parser.add_argument("-p", "--img_dir", help="imgs dir for evaluation ./XX/val/", default='./data/faces_emore/val/', type=str)
    parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")

    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', False, True)
    else:
        learner.load_state(conf, 'final.pth', False, True)
    learner.model.eval()
    print('learner loaded')

    if args.update:
        # for path in conf.facebank_path.iterdir():
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    val_data, val_label = get_val_dataset(args.p)
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, val_data, val_label, nrof_folds=5, tta=False)
    print("accuracy:{}, best_threshold:{}".format(accuracy, best_threshold))
    # tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, actual_issame, nrof_folds=10, pca=0)

