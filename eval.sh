## For RGB stream:
#python3 test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
#--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
#--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
#--frame_dir "dataset/HMDB51" \
#--annotation_path "dataset/HMDB51_labels" \
#--result_path "results/"
#
## For Flow stream:
#python3 test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
#--log 0 --dataset HMDB51 --modality Flow --sample_duration 16 --split 1  \
#--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
#--frame_dir "dataset/HMDB51" \
#--annotation_path "dataset/HMDB51_labels" \
#--result_path "results/"

# For single stream MARS: 
#CUDA_VISIBLE_DEVICES=0 python3 test_single_stream.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 \
#--log 1 --dataset Kinetics --modality RGB --sample_duration 64 --split 1 --only_RGB  \
#--resume_path1 "results/pths/Kinetics/MARS_Kinetics_64f.pth" \
#--frame_dir "dataset/Customs" \
#--annotation_path "dataset/Kinetics_labels" \
#--result_path "results/CCTV" \
#--n_workers 4

#CUDA_VISIBLE_DEVICES=0 python3 test_single_stream.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 \
#--log 1 --dataset Kinetics --modality RGB --sample_duration 16 --split 1 --only_RGB  \
#--resume_path1 "results/pths/Kinetics/MARS_Kinetics_64f.pth" \
#--frame_dir "dataset/Kinetics" \
#--annotation_path "dataset/Kinetics_labels" \
#--result_path "results/Customs" \
#--n_workers 4
##--annotation_path "dataset/Fight_labels" \

CUDA_VISIBLE_DEVICES=0 python3 test_single_stream.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 \
--log 1 --dataset Fight --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "results/pths/Kinetics/MARS_Kinetics_16f.pth" \
--frame_dir "dataset/Fight" \
--annotation_path "dataset/Fight_labels" \
--result_path "results/Fight" \
--n_workers 4

## For two streams RGB+MARS:
#python3 test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
#--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
#--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
#--resume_path2 "trained_models/HMDB51/MARS_HMDB51_16f.pth" \
#--frame_dir "dataset/HMDB51" \
#--annotation_path "dataset/HMDB51_labels" \
#--result_path "results/"
#
## For two streams RGB+Flow:
#python3 test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
#--log 0 --dataset HMDB51 --modality RGB_Flow --sample_duration 16 --split 1 \
#--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
#--resume_path2 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
#--frame_dir "dataset/HMDB51/HMDB51_frames/" \
#--annotation_path "dataset/HMDB51_labels" \
#--result_path "results/"
