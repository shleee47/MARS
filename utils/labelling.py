import os
import glob
import sys


def labeling(dataset_dir):
    ### Define label dir: '../dataset/Test_labels'
    if dataset_dir[-1] == '/':
        dataset_name = dataset_dir.split('/')[-2]
        label_dir_path = dataset_dir[:-1]+"_labels"
    else:
        dataset_name = dataset_dir.split('/')[-1]
        label_dir_path = dataset_dir+"_labels"
    if not os.path.exists(label_dir_path):
        os.makedirs(label_dir_path)

    for case in ["train", "val"]:
        ### Prepare classes of videos: ['fightT', 'fightF']
        case_dir = os.path.join(dataset_dir, case) # '../dataset/Test_labels/train'
        class_names = sorted([f for f in os.listdir(case_dir)])
        print("Case {}: {}".format(case, class_names))
        if '.DS_Store' in class_names:
            class_names.remove('.DS_Store')

        ### Define fout '../dataset/Test_labels/Test_train_labels.txt'
        fname = dataset_name+"_"+case+"_labels.txt"
        fout = os.path.join(label_dir_path, fname)
        with open(fout, 'w') as f:
            ### Iter class dir in case dir
            for cls in class_names:
                cls_dir = os.path.join(case_dir, cls)
                ### Iter video dir in class dir / Write
                for video_dir in os.listdir(os.path.join(cls_dir)):
                    if '.DS_Store' in video_dir:
                        continue
                    video_dir_full_path = os.path.join(cls_dir, video_dir)
                    print(case, cls, video_dir_full_path)
                    f_count = len(glob.glob1(video_dir_full_path, '0*.jpg')) # Extract only len(RGB frames)
                    cls_idx = class_names.index(cls)

                    f.write("{}/{}/{} {} {}\n".format(case, cls, video_dir, cls_idx, f_count))


if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    labeling(dataset_dir)
