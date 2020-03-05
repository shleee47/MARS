### For Local Test
#python3 utils/extract_frames.py /Users/sangbuemseo/drive_google/project_06/cctv_fight/MARS/dataset/videos/Customs/val/fightT/ /Users/sangbuemseo/drive_google/project_06/cctv_fight/MARS/dataset/Customs/val/fightT/ 0 400
#python3 utils/extract_frames.py /Users/sangbuemseo/drive_google/project_06/cctv_fight/MARS/dataset/videos/Customs/val/fightF/ /Users/sangbuemseo/drive_google/project_06/cctv_fight/MARS/dataset/Customs/val/fightF/ 0 400
cd utils
python3 extract_frames.py ../dataset/video/MOBIO/ ../dataset/MOBIO/ 0 2

### For Server
#python3 utils/extract_frames.py /home/nas/DB/DB_video-nonlocal-light/400_val/ /home/sangbuem/MARS/dataset/Kinetics/ 0 399
#python3 utils/extract_frames.py /home/nas/DB/DB_CCTV/CCTV_fight_light/ /home/sangbuem/MARS/dataset/CCTV_fight_light/ 0 2
#python3 utils/extract_frames.py /home/nas/DB/Kinetics/my_test/ /home/sangbuem/MARS/dataset/Kinetics/mytest/ 0 400
#python3 utils/extract_frames.py /Users/sangbuemseo/drive_google/project_06/cctv_fight/MARS/dataset/videos/Customs/fight_test1/ /Users/sangbuemseo/drive_google/project_06/cctv_fight/MARS/dataset/ntu_cctv/val/fight_test1/ 0 400
