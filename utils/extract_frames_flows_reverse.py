'''
Code extracts frames from video at a rate of 25fps and 
'''
  
import sys, os, pdb, cv2, subprocess, struct
import numpy as np
from tqdm import tqdm
TAG_FLOAT = 202021.25
MIN_WIDTH = 1
MAX_WIDTH = 99999
MIN_HEIGHT = 1
MAX_HEIGHT = 99999


def readFlowFile(filename):
    """
    flow_utils.readFlowFile(<FILENAME>) reads a flow file <FILENAME> into a 2-band np.array.

    if <FILENAME> does not exist, an IOError is raised.
    if <FILENAME> does not finish by '.flo' or the tag, the width, the height or the file's size is illegal, an Expcetion is raised.

    ---- PARAMETERS ----
        filename: string containg the name of the file to read a flow

    ---- OUTPUTS ----
        a np.array of dimension (height x width x 2) containing the flow of type 'float32'
    """
        
    # check filename
    if not filename.endswith(".flo"):
        raise Exception("readFlowFile({:s}): filename must finish with '.flo'".format(filename))
    
    # open the file and read it
    with open(filename,'rb') as f:
        # check tag
        tag = struct.unpack('f',f.read(4))[0]
        if tag != TAG_FLOAT:
            raise Exception("flow_utils.readFlowFile({:s}): wrong tag".format(filename))
        # read dimension
        w,h = struct.unpack('ii',f.read(8))
        if w < MIN_WIDTH or w > MAX_WIDTH:
            raise Exception("flow_utils.readFlowFile({:s}: illegal width {:d}".format(filename,w))
        if h < MIN_HEIGHT or h > MAX_HEIGHT:
            raise Exception("flow_utils.readFlowFile({:s}: illegal height {:d}".format(filename,h))
        flow = np.fromfile(f,'float32')
        if not flow.shape == (h*w*2,):
            raise Exception("flow_utils.readFlowFile({:s}: illegal size of the file".format(filename))
        flow.shape = (h,w,2)
        return flow       

def extract_flows(vid_dir, frame_dir, start, end, gpu, reverse, redo):
    print('start = ', start, ' end = ', end)
    class_list = sorted(os.listdir(vid_dir))[start:end] # Files in vid_dir
    if '.DS_Store' in class_list:
        class_list.remove('.DS_Store')

    print(class_list)
    for ic, cls in enumerate(class_list):
        print(ic, cls) 
        vlist = sorted(os.listdir(vid_dir + cls)) # list of videos of cur_clO
#        vlist=vlist[::-1]
        print(ic+1, len(class_list), cls, len(vlist)) # index, classes in vid_dir, Aur_cls, numb_of_videos
        for v in tqdm(vlist):
            # Outdir name by Extension
            if len(v.split('.')[-1]) == 3:
                outdir = os.path.join(frame_dir, cls, v[:-4])
            elif len(v.split('.')[-1]) == 4:
                outdir = os.path.join(frame_dir, cls, v[:-5])
            else:
                raise

            # Checking if frames already extracted
#            frame_list=sorted(os.listdir(outdir))
 
#            tvl1_num=0
#            rgb_num=0
#            for name in frame_list:
#                if name[0]=="T":
#                    tvl1_num+=1
#                elif name.split('.')[-1]=="jpg":
#                    rgb_num+=1
           # print("tv1 vs rgb : {} vs {}".format(tvl1_num, rgb_num))
#            rgb_real_num= 2*rgb_num-2
           # print(os.path.isfile(os.path.join(outdir, 'done')), redo, (tvl1_num == rgb_real_num))
            if os.path.isfile(os.path.join(outdir, 'done')) and redo:
                print("already done:{}".format(v))
                continue
            try:  
                # make dir
                os.system('mkdir -p "%s"'%(outdir))

                # check if horizontal or vertical scaling factor
                o = subprocess.check_output('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"'%(os.path.join(vid_dir, cls, v)), shell=True).decode('utf-8')
                lines = o.splitlines()
                width = int(lines[0].split('=')[1])
                height = int(lines[1].split('=')[1])
                resize_str = '-1:256' if width>height else '256:-1'

                # extract frames
                os.system('ffmpeg -i "%s" -r 30 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1'%( os.path.join(vid_dir, cls, v), resize_str, os.path.join(outdir, '%05d.jpg')))
                nframes = len([fname for fname in os.listdir(outdir) if fname.endswith('.jpg') and len(fname)==9])
                if nframes==0: raise Exception # pdb.set_trace()

                # extract flows 
#                os.system('export CUDA_VISIBLE_DEVICES=%d; /home/ncrasto/code/workspace/action-recog-release/utils1/tvl1_videoframes "%s" %d "%s" '%(gpu, os.path.join(outdir, '%05d.jpg'), nframes, os.path.join(outdir,'%05d_flow256crop.flo') ))
                os.system('export CUDA_VISIBLE_DEVICES=%d; /home/sangbuem/MARS/utils/tvl1_videoframes "%s" %d "%s" '%(gpu, os.path.join(outdir, '%05d.jpg'), nframes, os.path.join(outdir,'%05d_flow256crop.flo') ))

                """
                """
                ######
                tvl1_dir = './tvl1_videoframes'
                if not os.path.exists(tvl1_dir):
                    os.makedirs(tvl1_dir)
                    print("made******************************************************")
                print('export CUDA_VISIBLE_DEVICES=%d; "%s" "%s" %d "%s" '%(gpu, tvl1_dir, os.path.join(outdir, '%05d.jpg'), nframes, os.path.join(outdir,'%05d_flow256crop.flo')))
                os.system('export CUDA_VISIBLE_DEVICES=%d; ./tvl1_videoframes "%s" %d "%s" '%(gpu, os.path.join(outdir, '%05d.jpg'), nframes, os.path.join(outdir,'%05d_flow256crop.flo')))
                nflows = len([fname for fname in os.listdir(outdir) if fname.endswith('.flo')])
                """
                """

#                return #####

                if nflows != nframes-1: raise Exception 

                # read flow and save jpg
                for i in range(1, nframes):
                    flow = readFlowFile(os.path.join(outdir,'%05d_flow256crop.flo'%(i)))
                    tflow = np.minimum(20, np.maximum(-20, flow))
                    iflow = ((tflow+20)/40.0*255.0).astype(np.uint8)
                    cv2.imwrite(os.path.join(outdir, 'TVL1jpg_x_%05d.jpg'%(i)), iflow[:,:,0])
                    cv2.imwrite(os.path.join(outdir, 'TVL1jpg_y_%05d.jpg'%(i)), iflow[:,:,1])

                # mark as done
                for i in range(1,nframes):
                    os.system('rm "%s"'%(os.path.join(outdir,'%05d_flow256crop.flo'%(i))))
                os.system('touch "%s"'%(os.path.join(outdir, 'done')))
            except:
                print("ERROR", cls, v)


if __name__ == '__main__':
    vid_dir   = sys.argv[1]
    frame_dir = sys.argv[2]
    start     = int(sys.argv[3])
    end       = int(sys.argv[4])
    gpu       = int(sys.argv[5])
    reverse   = sys.argv[6]
    redo      = sys.argv[7]
    extract_flows(vid_dir, frame_dir, start, end, gpu, reverse, redo)
