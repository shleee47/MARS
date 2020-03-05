### MARS
## origin
#python MARS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
#--n_classes 400 --n_finetune_classes 51 \
#--batch_size 16 --log 1 --sample_duration 16 \
#--model resnext --model_depth 101 --ft_begin_index 4 \
#--output_layers 'avgpool' --MARS_alpha 50 \
#--frame_dir "dataset/HMDB51" \
#--annotation_path "dataset/HMDB51_labels" \
#--pretrain_path "trained_models/Kinetics/MARS_Kinetics_16f.pth" \
#--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
#--result_path "results/" --checkpoint 1

## From Pretrained Kinetics400 Using MARS
#python3 MARS_train.py --dataset Fight --modality RGB_Flow --split 1  \
#--n_classes 400 --n_finetune_classes 2 \
#--batch_size 1 --log 1 --sample_duration 16 \
#--model resnext --model_depth 101 --ft_begin_index 5 \
#--output_layers 'avgpool' --MARS_alpha 50 \
#--frame_dir "dataset/Fight" \
#--annotation_path "dataset/Fight_labels" \
#--pretrain_path "results/pths/Kinetics/MARS_Kinetics_16f.pth" \
#--resume_path1 "results/pths/Kinetics/Flow_Kinetics_16f.pth" \
#--result_path "results/" --checkpoint 1

## From Pretrained Kinetics400 Using MARS
#CUDA_VISIBLE_DEVICES=0 python3 MARS_train.py --dataset Fight --modality RGB_Flow --split 1  \
#--n_classes 400 --n_finetune_classes 2 \
#--batch_size 1 --log 1 --sample_duration 16 \
#--model resnext --model_depth 101 --ft_begin_index 4 \
#--output_layers 'avgpool' --MARS_alpha 50 \
#--frame_dir "dataset/Fight" \
#--annotation_path "dataset/Fight_labels" \
#--pretrain_path "results/pths/Kinetics/MARS_Kinetics_16f.pth" \
#--resume_path1 "results/pths/Kinetics/Flow_Kinetics_16f.pth" \
#--result_path "results/" --checkpoint 1

# From Pretrained Kinetics400 Using MARS
#CUDA_VISIBLE_DEVICES=0 python3 MARS_train.py --dataset Fight --modality RGB_Flow --split 1  \
#--n_classes 400 --n_finetune_classes 2 \
#--batch_size 1 --log 1 --sample_duration 16 \
#--model resnext --model_depth 101 --ft_begin_index 4 \
#--output_layers 'avgpool' --MARS_alpha 50 \
#--frame_dir "dataset/Fight" \
#--annotation_path "dataset/Fight_labels" \
#--pretrain_path "results/pths/Kinetics/MARS_Kinetics_16f.pth" \
#--result_path "results/" --checkpoint 1

# From Pretrained Kinetics400 Using MARS-Speaker
#CUDA_VISIBLE_DEVICES=1 python3 MARS_train.py --dataset MOBIO --modality RGB_Flow --split 1  \
#--n_classes 400 --n_finetune_classes 2 \
#--batch_size 1 --log 1 --sample_duration 16 \
#--model resnext --model_depth 101 --ft_begin_index 1 \
#--output_layers 'avgpool' --MARS_alpha 50 \
#--frame_dir "dataset/MOBIO" \
#--annotation_path "dataset/MOBIO_labels" \
#--pretrain_path "results/pths/Kinetics/MARS_Kinetics_16f.pth" \
#--result_path "results/" --checkpoint 1

CUDA_VISIBLE_DEVICES=1 python3 MARS_train.py --dataset LIP --modality RGB_Flow --split 1  \
--n_classes 400 --n_finetune_classes 2 \
--batch_size 1 --log 1 --sample_duration 64 \
--model resnext --model_depth 101 --ft_begin_index 1 \
--output_layers 'avgpool' --MARS_alpha 50 \
--frame_dir "dataset/LIP" \
--annotation_path "dataset/LIP_labels" \
--pretrain_path "results/pths/Kinetics/MARS_Kinetics_64f.pth" \
--result_path "results/" --checkpoint 1







