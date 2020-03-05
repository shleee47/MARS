
from vidstab.VidStab import VidStab

stabilizer = VidStab()
vidcap = cv2.VideoCapture('front_m108_12_f07_i0_0.mpg')

while True:
     grabbed_frame, frame = vidcap.read()

     if frame is not None:
        # Perform any pre-processing of frame before stabilization here
        pass

     # Pass frame to stabilizer even if frame is None
     # stabilized_frame will be an all black frame until iteration 30
     stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,
                                                   smoothing_window=30)
     if stabilized_frame is None:
         # There are no more frames available to stabilize
         break

     # Perform any post-processing of stabilized frame here
     pass
