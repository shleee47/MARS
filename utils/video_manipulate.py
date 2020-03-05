from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mpy
from moviepy.video.fx.all import crop


def trimming(fin, start_time, duration, fout):
    ffmpeg_extract_subclip(fin, start_time, start_time+duration, targetname=fout)
    print("Done cutting {} to {}".format(fin, fout))


def cropping(fin, left_up, right_down, fout):
    clip = mpy.VideoFileClip(fin)
    cropped = crop(clip, x1=left_up[0], y1=left_up[1], x2=right_down[0], y2=right_down[1])
    cropped.write_videofile(fout)



if __name__ == '__main__':
#    trimming("/home/nas/DB/DB_CCTV/CCTV_light/original/CCTV_fight_light/v9zKA1-N14w_000022_000032.mp4", 22, 10, "/home/nas/DB/DB_CCTV/CCTV_light/val/CCTV_fight_light/v9zKA1-N14w_000022_000032.mp4")
#    trimming("/home/nas/DB/DB_CCTV/CCTV_light/original/CCTV_fight_light/ZJ58OrWchLc_000055_000066.mp4", 22, 10, "/home/nas/DB/DB_CCTV/CCTV_light/val/CCTV_fight_light/ZJ58OrWchLc_000055_000066.mp4")

    cropping("/home/nas/DB/Kinetics/my_test/punching-person/1qO2kWVNV60_000154_000164.mp4", (300,1), (550,250), "/home/nas/DB/Kinetics/my_test/punching-person/cropped.mp4")
