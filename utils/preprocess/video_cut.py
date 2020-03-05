import sys, os
import json
from ast import literal_eval
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def countRatio(JsonOrTxt, rootdir):
    _, ext = os.path.splitext(JsonOrTxt)

    if ext == '.txt':
        with open(JsonOrTxt, 'r') as f:
            lines = f.readlines()

            ### count the ratio of train / test / val
            numb_train, numb_test, numb_val = 0, 0, 0
            for line in lines:
                dict = literal_eval(line)['database']
                videos = dict.keys()
                for video in videos:
                    if dict[video]['subset'] == "training":
                        numb_train += 1
                    elif dict[video]['subset'] == 'testing':
                        numb_test += 1
                    elif dict[video]['subset'] == 'validation':
                        numb_val += 1

    return numb_train, numb_test, numb_val


def videoCut(rootdir, video_ext='.mpeg', include_negative='true'):
    label_file = os.path.join(rootdir, 'label.txt')

    if rootdir[-1] == '/':
        class_name = rootdir.split('/')[-2]
    else:
        class_name = rootdir.split('/')[-1]

    targetdir = os.path.abspath(os.path.join(rootdir, class_name+'T'))
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    with open(label_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            dict = literal_eval(line)['database']
            for video in dict.keys():
                video_path = os.path.join(rootdir, 'origin', video+video_ext) # dataset/videos/Test/origin/fight_0001.mpeg'
                false_annotations, offset = [], 0
                ### get all segs for seg_duration >= 10 -> .../fightT/NUMB4-M3_START6_END6
                cnt = 1
                for seg in dict[video]['annotations']: # 'annotations': [{'segment': [start, end]}, {'segment': [start, end]}]
                    start_duration = int(seg['segment'][0])
                    end_duration = int(seg['segment'][1])

                    false_annotations.append({'segment': [offset, start_duration]})
                    offset = end_duration+1

                    while end_duration - start_duration >= 10:
                        seg_file = '{}-{:03d}_{:06d}_{:06d}'.format(video.split("_")[1], cnt, start_duration, start_duration+10)
#                        seg_dir_path = os.path.join(targetdir, seg_file) #
#                        if not os.path.exists(seg_dir_path):
#                            os.makedirs(seg_dir_path)
#                        else:
#                            raise
#                        ffmpeg_extract_subclip(video_path, start_duration, start_duration+10, targetname=os.path.join(seg_dir_path, seg_file+video_ext))
                        ffmpeg_extract_subclip(video_path, start_duration, start_duration+10, targetname=os.path.join(targetdir, seg_file+video_ext))
                        start_duration += 10
                        cnt += 1

                ### get all rests of video for rest_duration >= 10 -> .../fightF/NUMB4-M3_START6_END6
                if include_negative=='true':
                    negative_dir = os.path.join(rootdir, class_name+'F')
                    if not os.path.exists(negative_dir):
                        os.makedirs(negative_dir)

                    cnt = 1
                    for false_seg in false_annotations:
                        start_duration = int(false_seg['segment'][0])
                        end_duration = int(false_seg['segment'][1])
                        while end_duration - start_duration >= 10:
                            seg_file = '{}-{:03d}_{:06d}_{:06d}'.format(video.split("_")[1], cnt, start_duration, start_duration+10)
#                            seg_dir_path = os.path.join(negative_dir, seg_file)
#                            if not os.path.exists(seg_dir_path):
#                                os.makedirs(seg_dir_path)
#                            else:
#                                raise
#                            ffmpeg_extract_subclip(video_path, start_duration, start_duration+10, targetname=os.path.join(seg_dir_path, seg_file+video_ext))
                            ffmpeg_extract_subclip(video_path, start_duration, start_duration+10, targetname=os.path.join(negative_dir, seg_file+video_ext))
                            start_duration += 10
                            cnt += 1


if __name__ == '__main__':
    rootdir = sys.argv[1]
    video_ext = sys.argv[2]
    include_negative = sys.argv[3]
    videoCut(rootdir, video_ext, include_negative)

