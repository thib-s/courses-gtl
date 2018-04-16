import cv2
import numpy as np
from tools.ParticleFilter import ParticleFilter

cap = cv2.VideoCapture('inputs/pres_debate.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

if cap.isOpened():
    ret, frame = cap.read()
    gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # pf = ParticleFilter(243, 371, gr[150:300, 325:425], 1000, 10)
    # pf = ParticleFilter(243, 371, gr[235:287, 349:407], 1000, 10)
    # pf = ParticleFilter(450, 575, gr[380:498, 529:609], 10000, 10, 0.999)
    pf = ParticleFilter(0.5*350+0.5*540, 0.5*280+0.5*400, gr[350:540, 280:400], 10000, 10, 0.99)
    while cap.isOpened():
        ret, frame = cap.read()
        pf.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame = pf.draw_tracking_window(pf.draw_particles(frame))
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()