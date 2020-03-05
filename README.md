# MARS: Motion-Augmented RGB Stream for Action Recognition
  
## Table of Contents
0. [Reference](#reference)
1. [Requirements](#requirements)
2. [Public Datasets](#public-datasets)
3. [Custom Datasets](#custom-datasets)
4. [Trained Models](#trained-models)
5. [Testing](#testing)
6. [Training](#training)
7. [Transfer Learning on custom data](#transfer-learning-on-custom-data)


## Reference
* [MARS Github 원본](https://github.com/craston/MARS)
* [논문(CVPR 2019 paper)](https://hal.inria.fr/hal-02140558/document)  / [website](https://europe.naverlabs.com/Research/Computer-Vision/Video-Analysis/MARS)
   * 저자: By Nieves Crasto, Philippe Weinzaepfel, Karteek Alahari and Cordelia Schmid
   * 요약: 
MARS is a strategy to learn a stream that takes only RGB frames as input
but leverages both appearance and motion information from them. This is
achieved by training a network to minimize the loss between its features and
the Flow stream, along with the cross entropy loss for recognition.
We release the testing code along trained models. 
For more details, please refer to our and our 


## Requirements
* Python3  
* [Pytorch 1.0](https://pytorch.org/get-started/locally/)  
   * e.g) `conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`
* ffmpeg version 3.2.4
* OpenCV with GPU support (will not be providing support in compiling this part)
   * [Installation guides depends on env](https://github.com/spmallick/learnopencv)
   * OpenCV with CUDA and FFMpeg on Ubuntu 16.04  
      1. Prepare Prerequisite
      ```  
      step1) Update / Upgrade pre-installed packages:  
      $sudo apt-get update  
      $sudo apt-get upgrade  
        
      step2) Install developer tools used to compile OpenCV  
      $sudo apt-get install build-essential cmake pkg-config  
        
      step3) Install libraries and packages used to read various image formats from disk
      $sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
        
      step4) Install a few libraries used to read video formats from disk  
      $sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
      $sudo apt-get install libxvidcore-dev libx264-dev
        
      step5) Install GTK so we can use OpenCV’s GUI features
      $sudo apt-get install libgtk-3-dev
        
      step6) Install packages that are used to optimize various functions inside OpenCV, such as matrix operations
      $sudo apt-get install libatlas-base-dev gfortran
        
      step7) Clone opencv / opencv_contrib from github
      $git clone https://github.com/opencv/opencv.git
      $git clone https://github.com/opencv/opencv_contrib.git
        
      step8) Step 9: Now set up the build
      $cd opencv
      $mkdir build
      $cd build
      ```
      2. Build depend on personal develop env 
      ```
      step9) Run cmake command with appropriate options   
      (following case is on Ubuntu16.04 / cudo-10.0.0 / Titan RTX: Turing Arch)

      $cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CUDA_ARCH_PTX=7.5 \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON \
      -D BUILD_opencv_python2=OFF \
      -D WITH_FFMPEG=1 \
      -D WITH_CUDA=ON \
      -D CUDA_GENERATION=Turing \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python \
      -D PYTHON3_INCLUDE_DIR=/usr/include/python3.5m \
      -D PYTHON3_LIBRARY=/usr/lib/libpython3.5m.so \
      -D PYTHON3_PACKAGES_PATH=/usr/lib/python3.5 \
      -D WITH_LAPACK=OFF \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3.5/site-packages/numpy/core/include ..
        
      setp10) Make the build now  
      $make -j4  
        
      step11) Install OpenCV  
      $sudo make install
      ```


* Directory tree
   ```
   dataset/
       HMDB51/ 
           ../(dirs of class names)
               ../(dirs of video names)
       HMDB51_labels/
   results/
       test.txt
   trained_models/
       HMDB51/
           ../(.pth files)
   ```


## Public Datasets
### (1/2) Prepare Data
* The datsets and splits can be downloaded from 

   *   [Kinetics400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
   
   *   [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
   
   *   [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
   
   *   [SomethingSomethingv1](https://20bn.com/datasets/something-something/v1)

### (2/2) Extract
* To extract only frames from videos  
   1. change **extract_frames.sh** options
      ```
      utils/extract_frames.py PATH_TO_VIDEO_FILES PATH_TO_EXTRACTED_FRAMES START_CLASS END_CLASS
      ```  
   2. execute  
      ```
      sh extract_frames.sh  
      ```

* To extract optical flows + frames from videos 
   1. Build
      ```
      g++ -std=c++11 tvl1_videoframes.cpp -o tvl1_videoframes -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudaarithm
      ```
   2. change **extract_frames_flows.sh** options
      ```  
      utils/extract_frames_flows.py path_to_video_files path_to_extracted_flows_frames start_class end_class gpu_id
       ```
   3. execute
      ```
      sh extract_frames_flows.sh  
      ```
      

## Custom Datasets
### (1/3) Prepare Video Format Data
1. 원본 동영상 준비
   1. 파일구조
      ```
      |--dataset
      |  |--video
      |  |  |--VIDEOCLASSNAME
      |  |  |  |--origin
      |  |  |  |  |--fight_0001.mpeg
      |  |  |  |  |--fight_0002.mpeg
      |  |  |  |--label.txt
      ```

   2. 라벨
      ```
      {
        "version": "1.0",
        "database": {
           "fight_0001": {
              "duration": 66.0666,
              "subset": "validation"
              "nb_frames": 2400,
              "frame_rate": 30.0,
              "source": "CCTV"
              "annotations": [
                {
                  "segment": [
                    12.233,
                    36.4
                  ],
                  "label": "Fight"
                }
                {
                  "segment": [
                    41.2,
                    53.5
                  ],
                  "label": "Fight"
                }
              ]
           }
           "fight_0002": {
             ...
           }
        }
      }
      ```

2. 10초 간격 자르기
   1. change **utils/prerocess/video_cut.sh** options
      ```
      video_cut.py PATH_TO_VIDEONAME VIDEOFORMAT INCLUDE_NEGATIVE
      ```

   2. execute
      ```
      sh video_cut.sh
      ```

3. 10초 간격 동영상 파일구조
   ```
   |--dataset
   |  |--video
   |  |  |--VIDEOCLASSNAME
   |  |  |  |--CLASSNAMET
   |  |  |  |  |--0001-001-000020_000030.mpeg
   |  |  |  |  |--0001-002-000047_000057.mpeg
   |  |  |  |  |--0002-001-000020_000030.mpeg
   |  |  |  |--CLASSNAMEF
   |  |  |  |  |--0001-001-000010_000020.mpeg
   ```

### (2/3) Extract Continuous Clips(appox. 10s) from Video
1. 10초 간격 자른 동영상 준비
   ```
   |--dataset
   |  |--video
   |  |  |--VIDEOCLASSNAME
   |  |  |  |--CLASSNAMET
   |  |  |  |  |--0001-001-000020_000030.mpeg
   |  |  |  |  |--0001-002-000047_000057.mpeg
   |  |  |  |  |--0002-001-000020_000030.mpeg
   |  |  |  |--CLASSNAMEF
   |  |  |  |  |--0001-001-000010_000020.mpeg
   ```

2. 프레임 추출
   * To extract only frames from videos  
      1. change **extract_frames.sh** options
         ```
         utils/extract_frames.py PATH_TO_VIDEO_FILES PATH_TO_EXTRACTED_FRAMES START_CLASS END_CLASS
         ```  
      2. execute  
         ```
         sh extract_frames.sh  
         ```
   
   * To extract optical flows + frames from videos 
      1. Build
         ```
         g++ -std=c++11 tvl1_videoframes.cpp -o tvl1_videoframes -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudaarithm
         ```
      2. change **extract_frames_flows.sh** options
         ```  
         utils/extract_frames_flows.py path_to_video_files path_to_extracted_flows_frames start_class end_class gpu_id
          ```
      3. execute
         ```
         sh extract_frames_flows.sh  
         ```

3. 10초 간격 프레임 파일구조
   ```
   |--dataset
   |  |--CUSTOMDATASET
   |  |  |--VIDEOCLASSNAME
   |  |  |  |--0001-001-000020_000030
   |  |  |  |  |--00001.jpg
   |  |  |  |  |--00XXX.jpg
   |  |  |  |  |--done
   |  |  |  |--0002-001-000020_000030
   ```

4. 프레임 분류 (수작업)

   ```
   |--dataset
   |  |--CUSTOMDATASET
   |  |  |--train
   |  |  |  |--VIDEOCLASSNAME
   |  |  |  |  |--0001-001-000020_000030
   |  |  |  |  |  |--00001.jpg
   |  |  |  |  |  |--00XXX.jpg
   |  |  |  |  |  |--done
   |  |  |  |  |--0002-001-000020_000030
   |  |  |--val
   |  |  |  |--VIDEOCLASSNAME
   |  |  |  |  |--0001-002-000047_000057
   |  |  |  |  |  |--00001.jpg
   |  |  |  |  |  |--00XXX.jpg
   |  |  |  |  |  |--done
   ```

### (3/3) Prepare Label
1. 분류된 프레임 파일구조 준비
   ```
   |--dataset
   |  |--CUSTOMDATASET
   |  |  |--train
   |  |  |  |--VIDEOCLASSNAME
   |  |  |  |  |--0001-001-000020_000030
   |  |  |  |  |  |--00001.jpg
   |  |  |  |  |  |--00XXX.jpg
   |  |  |  |  |  |--done
   |  |  |  |  |--0002-001-000020_000030
   |  |  |--val
   |  |  |  |--VIDEOCLASSNAME
   |  |  |  |  |--0001-002-000047_000057
   |  |  |  |  |  |--00001.jpg
   |  |  |  |  |  |--00XXX.jpg
   |  |  |  |  |  |--done
   ```

2. labelling
   1. change **labelling.sh** options
      ```
      labelling.py PATH_TO_EXTRACTED_DATASET
      ```

   2. execute
      ```
      sh labelling.sh
      ```

## Trained Models

Trained models can be found [here](https://drive.google.com/drive/folders/1OVhBnZ_FmqMSj6gw9yyrxJJR8yRINb_G?usp=sharing). (ex.
`stream_dataset_frames.pth` or `RGB_Kinetics_16f.pth (indicates --modality RGB --dataset Kinetics --sample_duration 16)` and etc)

For HMDB51 and UCF101, we have only provided trained models for the first split.


## Testing
1. edit **eval.sh** options
   * For RGB stream:
      ```
      python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
      --log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
      --resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --result_path "results/"
      ```
   
   * For Flow stream:
      ```
      python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
      --log 0 --dataset HMDB51 --modality Flow --sample_duration 16 --split 1  \
      --resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --result_path "results/"
      ```
   
   * For single stream MARS: 
      ```
      python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
      --log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
      --resume_path1 "trained_models/HMDB51/MARS_HMDB51_16f.pth" \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --result_path "results/"
      ```
   
   * For two streams RGB+MARS:
      ```
      python test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
      --log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
      --resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
      --resume_path2 "trained_models/HMDB51/MARS_HMDB51_16f.pth" \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --result_path "results/"
      ```
   
   * For two streams RGB+Flow:
      ```
      python test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
      --log 0 --dataset HMDB51 --modality RGB_Flow --sample_duration 16 --split 1 \
      --resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
      --resume_path2 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
      --frame_dir "dataset/HMDB51/HMDB51_frames/" \
      --annotation_path "dataset/HMDB51_labels" \
      --result_path "results/"
      ```

2. execute
   ```
   sh eval.sh
   ```


## Training
1. edit **train.sh** options
   #### For RGB stream: 
   * From scratch:
      ```
       python train.py --dataset Kinetics --modality RGB --only_RGB \
      --n_classes 400 \
      --batch_size 32 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101  \
      --frame_dir "dataset/Kinetics" \
      --annotation_path "dataset/Kinetics_labels" \
      --result_path "results/"
      ```
   
   * From pretrained Kinetics400:
      ```
       python train.py --dataset HMDB51 --modality RGB --split 1 --only_RGB \
      --n_classes 400 --n_finetune_classes 51 \
      --batch_size 32 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 --ft_begin_index 4 \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --pretrain_path "trained_models/Kinetics/RGB_Kinetics_16f.pth" \
      --result_path "results/"
      ```
   
   * From checkpoint:
      ```
       python train.py --dataset HMDB51 --modality RGB --split 1 --only_RGB \
      --n_classes 400 --n_finetune_classes 51 \
      --batch_size 32 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 --ft_begin_index 4 \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --pretrain_path "trained_models/Kinetics/RGB_Kinetics_16f.pth" \
      --resume_path1 "results/HMDB51/PreKin_HMDB51_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx4_varLR2.pth" \
      --result_path "results/"
      ```
   
   #### For Flow stream 
   * From scratch:
      ```
       python train.py --dataset Kinetics --modality Flow \
      --n_classes 400 \
      --batch_size 32 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101  \
      --frame_dir "dataset/Kinetics" \
      --annotation_path "dataset/Kinetics_labels" \
      --result_path "results/"
      ```
   
   * From pretrained Kinetics400:
      ```
       python train.py --dataset HMDB51 --modality Flow --split 1 \
      --n_classes 400 --n_finetune_classes 51 \
      --batch_size 32 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 --ft_begin_index 4 \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --pretrain_path "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
      --result_path "results/"
      ```
   
   * From checkpoint:
      ```
       python train.py --dataset HMDB51 --modality Flow --split 1 \
      --n_classes 400 --n_finetune_classes 51 \
      --batch_size 32 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 --ft_begin_index 4 \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --pretrain_path "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
      --resume_path1 "results/HMDB51/PreKin_HMDB51_1_Flow_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx4_varLR2.pth" \
      --result_path "results/"
      ```
   
   #### For MARS:
   * From scratch:  
      ```
      python MARS_train.py --dataset Kinetics --modality RGB_Flow \
      --n_classes 400 \
      --batch_size 16 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 \
      --output_layers 'avgpool' --MARS_alpha 50 \
      --frame_dir "dataset/Kinetics" \
      --annotation_path "dataset/Kinetics_labels" \
      --resume_path1 "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
      --result_path "results/" --checkpoint 1
      ```
   
   * From pretrained Kinetics400:
      ```
      python MARS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
      --n_classes 400 --n_finetune_classes 51 \
      --batch_size 16 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 --ft_begin_index 4 \
      --output_layers 'avgpool' --MARS_alpha 50 \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --pretrain_path "trained_models/Kinetics/MARS_Kinetics_16f.pth" \
      --resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
      --result_path "results/" --checkpoint 1
      ```
   * From checkpoint:
      ```
      python MARS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
      --n_classes 400 --n_finetune_classes 51 \
      --batch_size 16 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 --ft_begin_index 4 \
      --output_layers 'avgpool' --MARS_alpha 50 \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --pretrain_path "trained_models/Kinetics/MARS_Kinetics_16f.pth" \
      --resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
      --MARS_resume_path "results/HMDB51/MARS_HMDB51_1_train_batch16_sample112_clip16_lr0.001_nesterovFalse_manualseed1_modelresnext101_ftbeginidx4_layeravgpool_alpha50.0_1.pth" \
      --result_path "results/" --checkpoint 1
      ```
   
   #### For MERS:
   * From scratch:  
      ```
      python MERS_train.py --dataset Kinetics --modality RGB_Flow \
      --n_classes 400 \
      --batch_size 16 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 \
      --output_layers 'avgpool' --MARS_alpha 50 \
      --frame_dir "dataset/Kinetics" \
      --annotation_path "dataset/Kinetics_labels" \
      --resume_path1 "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
      --result_path "results/" --checkpoint 1
      ```
   
   * From pretrained Kinetics400:
      ```
      python MERS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
      --n_classes 400 --n_finetune_classes 51 \
      --batch_size 16 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 --ft_begin_index 4 \
      --output_layers 'avgpool' --MARS_alpha 50 \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --pretrain_path "trained_models/Kinetics/MERS_Kinetics_16f.pth" \
      --resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
      --result_path "results/" --checkpoint 1
      ```
   * From checkpoint:
      ```
      python MERS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
      --n_classes 400 --n_finetune_classes 51 \
      --batch_size 16 --log 1 --sample_duration 16 \
      --model resnext --model_depth 101 --ft_begin_index 4 \
      --output_layers 'avgpool' --MARS_alpha 50 \
      --frame_dir "dataset/HMDB51" \
      --annotation_path "dataset/HMDB51_labels" \
      --pretrain_path "trained_models/Kinetics/MARS_Kinetics_16f.pth" \
      --resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
      --MARS_resume_path "results/HMDB51/MERS_HMDB51_1_train_batch16_sample112_clip16_lr0.001_nesterovFalse_manualseed1_modelresnext101_ftbeginidx4_layeravgpool_alpha50.0_1.pth" \
      --result_path "results/" --checkpoint 1
      ```

2. execute
   ```
   sh train.sh
   ```

## Transfer Learning on custom data
### Manual  
   1. Prepare and preprocess custom datasets
   1. Change **train.sh**
      * Choose and prepare which pre-trained model to user (16f.pth vs 64f.pth)
      * Decide which layer to start transfer-learning
   1. Execute
     
### Example Usage  
   1. **CCTV Fight Detection True/False - (민석 / 혜진)**  
      <!-- R3257 / 1C2DFCD8 -->
      1. Ask for Authority for Downloading CCTV Fight datasets [[Source](http://rose1.ntu.edu.sg/Datasets/cctvFights.asp)]
      1. Pre-process datasets according to **[Custom Datasets Part](#custom-datasets)** of this document
      1. <strike>Need to fix MARS_train.py(Line#117~126) and Make proper 'python class' in dataset/dataset.py (Already Prepared by Sangbuem Seo)</strike>
      1. Change train.sh and Execute  
         * Specially adjust   
            * --n_finetune_classes: numbers of final classes to classify   
            * --ft_begin_index options: which layers to freeze or update weight (check **models/resnext.py** ```def get_fine_tuning_parameters(model, ft_begin_index)```)   
               * ex) set '4' to train 'layer4' and Fully Connected Layer
               * ex) set '5' to only train Fully Connected Layer
   1. **Lip Activation Detection True/False - (상훈 / 세영)**     
      1. Prepare Lip Activation Datasets
      1. Pre-process datasets according to **[Custom Datasets Part](#custom-datasets)** of this document
      1. Need to fix MARS_train.py(Line#117~126) and Make proper 'python class' in dataset/dataset.py (As ```class Fight_test(Dataset)``` (Line#358)
      1. Change train.sh and Execute
         * Specially adjust   
            * --n_finetune_classes: numbers of final classes to classify   
            * --ft_begin_index options: which layers to freeze or update weight (check **models/resnext.py** ```def get_fine_tuning_parameters(model, ft_begin_index)```)   
               * ex) set '4' to train 'layer4' and Fully Connected Layer
               * ex) set '5' to only train Fully Connected Layer

### Farther Trial
   1. Apply **LDA(Latent Dirichlet Allocation)** Analysis among Classes to sort out **most related** classes to out target class 


<hr>

<!-- #### Performance on Kinetics400

|Method 												| Stream   | Pretrain | Acc   |
|-------------------------------------------------------|:--------:|:--------:|:-----:|
|[I3D](https://arxiv.org/pdf/1705.07750.pdf)			| RGB      |ImageNet  | 71.1  |
|[ResNext101](https://arxiv.org/pdf/1711.09577.pdf)	    | RGB      |none      | 65.1  |
|[R(2+1)D](https://arxiv.org/pdf/1711.11248.pdf)	    | RGB      |Sport-1M  | 74.3  |
|[S3D-G](https://arxiv.org/pdf/1712.04851.pdf)		    | RGB      |ImageNet  | 74.7  |
|[NL-I3D](https://arxiv.org/pdf/1711.07971.pdf)		    | RGB      |ImageNet  |**77.7**|
|**MARS**                   							| RGB      | none     |  72.7 |
|**MARS+RGB**               							| RGB      | none     |  74.8 |
|[I3D](https://arxiv.org/pdf/1705.07750.pdf)			| RGB+Flow | ImageNet |  74.2 |
|[R(2+1)D](https://arxiv.org/pdf/1711.11248.pdf)		| RGB+Flow | Sports-1M| 75.4  |
|[S3D-G](https://arxiv.org/pdf/1712.04851.pdf)		    | RGB+Flow | ImageNet | 77.2  |
|**MARS+RGB+Flow**		          					    | RGB+Flow |  none    | 74.9  | -->

<!-- #### Performance om HMDB51, UCF101 and SomethingSomethingv1

|Method 				| Streams | Pretrain | UCF101 | HMDB51 | Something Somethingv1|
|-----------------------|:-------:|:--------:|:------:|:------:|:--------------------:|
[TRN](https://arxiv.org/pdf/1711.08496.pdf)                 |RGB      |none      | ---    | ---    | 34.4                 |
[MFNet](https://arxiv.org/pdf/1807.10037.pdf)   			|RGB      |none      | ---    | ---    | 43.9       |
[I3D](https://arxiv.org/pdf/1705.07750.pdf)					|RGB      |ImNet+Kin | 95.6   | 74.8   | ---        |
[ResNext101](https://arxiv.org/pdf/1711.09577.pdf)			|RGB      |Kinetics  | 94.5   | 70.1   | ---    	|
[S3D-G](https://arxiv.org/pdf/1712.04851.pdf)         		|RGB      |ImNet+Kin | 96.8   | 75.9   | 48.2		|
[R(2+1)D](https://arxiv.org/pdf/1711.11248.pdf)       		|RGB      |Kinetics  | 96.8   | 74.5   | ---        |
**MARS**                      								|RGB      |Kinetics  | 97.4   | 79.3   | 48.7		|
**MARS+RGB**                  								|RGB      |Kinetics  |**97.6**|**79.5**|**51.7**	|
[2-stream](https://arxiv.org/pdf/1406.2199.pdf)				|RGB+Flow |ImageNet  | 88.0   | 59.4   | --- 		|
[TRN](https://arxiv.org/pdf/1711.08496.pdf)					|RGB+Flow |none      | ---    | ---    | 42.0		|
[I3D](https://arxiv.org/pdf/1705.07750.pdf)             	|RGB+Flow |ImNet+Kin |**98.0**|**80.7**| ---		|
[R(2+1)D](https://arxiv.org/pdf/1711.11248.pdf)				|RGB+Flow |Kinetics  |  97.3  | 78.7   | --- 		|
[OFF](https://arxiv.org/pdf/1711.11152.pdf)					|RGB+Flow |none      | 96.0   | 74.2   | --- 		|
**MARS+RGB+Flow**            								|RGB+Flow |Kinetics  |**98.1**|**80.9**|**53.0**	| -->
