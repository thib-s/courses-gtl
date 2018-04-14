import cv2
import numpy as np
from tools.ParticleFilter import ParticleFilter

cap = cv2.VideoCapture('inputs/pres_debate.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

if cap.isOpened():
    ret, frame = cap.read()
    gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # pf = ParticleFilter(np.mat([[371], [243]]), gr[150:300, 225:325], 500, 20)
    pf = ParticleFilter(np.mat([[371], [243]]), gr[349:407, 235:287], 200, 10)
    while cap.isOpened():
        ret, frame = cap.read()
        pf.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame = pf.draw_tracking_window(pf.draw_particles(frame))
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()