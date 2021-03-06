import cv2
import numpy as np
from tools.ParticleFilter import ParticleFilter


def track(input, w_cx, w_cy, w_w, w_h, N, sigma, alpha=None, scale_bins=None, output=None, display=True, frames_to_save=None):
    """
    apply particle filter to track something in a video
    :param input: the path string of the input video
    :param w_cx: position of the window center (x)
    :param w_cy: position of the window center (y)
    :param w_w: window width
    :param w_h: window height
    :param N: the amount of particles
    :param sigma: the smoothing factor
    :param alpha: the alpha factor used in appearance model update
    :param output: the basename of the output video, it will be extended with params values
    :param display: boolean indicating whenever to display the frame as they are beeing tracked
    :param frames_to_save: a list of frames to save, those will be saved as jpg an returned as result
    :return: the dict of the saved frames
    """
    out = None
    output_frames = {}
    if frames_to_save is None:
        frames_to_save = []
    cap = cv2.VideoCapture(input)
    i = 0
    if cap.isOpened():
        ret, frame = cap.read()
        i += 1
        gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if output is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output/' + output +
                                  '_w-' + str(w_w) + '_h-' + str(w_h) + 'N_' + str(N) +
                                  '_sigma-' + str(sigma) + '_alpha-' + str(alpha) +
                                  '.avi',
                                  fourcc, 20.0, (gr.shape[1], gr.shape[0]))
        pf = ParticleFilter(w_cx, w_cy, frame[w_cx - w_w:w_cx + w_w, w_cy - w_h:w_cy + w_h], N, sigma, alpha, scale_bins)
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            pf.update(frame)
            frame = pf.draw_tracking_window(pf.draw_particles(frame))
            if display:
                cv2.imshow('frame', frame)
            if frames_to_save.__contains__(i):
                cv2.imwrite('output/' + output +
                            '_w-' + str(w_w) + '_h-' + str(w_h) + 'N_' + str(N) +
                            '_sigma-' + str(sigma) + '_alpha-' + str(alpha) +
                            '_frame-' + str(i) +
                            '.jpg',
                            frame)
                output_frames[i] = frame
                output_frames[10000+i] = pf.window
            if out is not None:
                out.write(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            i += 1
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    return output_frames


if __name__ == '__main__':
    # track('inputs/pres_debate.avi', 239, 371, 64, 51, N=5000, sigma=10, alpha=None, display=True, output='Q1', frames_to_save=[28, 84, 144])
    # track('inputs/pres_debate.avi', 433, 570, 60, 60, N=5000, sigma=2000, alpha=0.5, display=True, output='Q2',
    #       frames_to_save=[15, 50, 140])
    track('inputs/pedestrians.avi', 140, 240, 50, 50, N=1000, sigma=1000, alpha=0.9, scale_bins=np.arange(1.0, 0.6, -0.05), display=True, output='Q33',
          frames_to_save=[40, 100, 240])
