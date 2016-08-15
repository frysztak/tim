#!/usr/bin/env python

import sys, json, collections, os
from nanomsg import Socket, PAIR
import cv2
import numpy as np

if __name__ == '__main__':
    socket = Socket(PAIR)
    socket.connect('ipc:///tmp/tim.ipc')
    filePath = '/mnt/things/tim/json/act.json' #socket.recv()
    # open json file
    with open(filePath) as f:
        jsonData = json.load(f, object_pairs_hook=collections.OrderedDict)

    cap = cv2.VideoCapture(os.path.join('/mnt/things/tim/videos/', jsonData['video']))
    ret, frame = cap.read()
    mask = np.zeros(frame.shape, np.uint8)
    points = []
    for (x,y) in jsonData['roi']:
        points.append([int(x*frame.shape[1]), int(y*frame.shape[0])])
    movedPointIdx = -1
    
    def generateMask():
        global mask
        mask = np.zeros(frame.shape, np.uint8)
        poly = cv2.approxPolyDP(np.asarray(points), 1.0, True)
        cv2.fillConvexPoly(mask, poly.astype(np.int32), (255,255,255))

    def mouseCallback(event, x, y, flags, param):
        global movedPointIdx

        if event == cv2.EVENT_LBUTTONDBLCLK and len(points) < 4:
            points.append([x,y])
            if len(points) == 4:
                generateMask()
        elif event == cv2.EVENT_LBUTTONDOWN and len(points) == 4 and movedPointIdx == -1:
            for i, (x_,y_) in enumerate(points):
                if abs(x-x_) <= 8 and abs(y-y_) <= 8:
                    movedPointIdx = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and len(points) == 4 and movedPointIdx != -1:
            points[movedPointIdx] = [x, y]
            generateMask()
        elif event == cv2.EVENT_LBUTTONUP and len(points) == 4 and movedPointIdx != -1:
            movedPointIdx = -1

    cv2.namedWindow('OpenCV')
    cv2.setMouseCallback('OpenCV', mouseCallback)
    generateMask()

    while True:
        disp = frame.copy()
        for (x,y) in points:
            cv2.circle(disp, (x,y), 8, (255,0,0), -1)

        disp = cv2.add(disp, mask)
        cv2.imshow('OpenCV', disp)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            mask = np.zeros(frame.shape, np.uint8)
            points.clear()
        elif key == ord('s'):
            jsonData['roi'] = []
            for (x, y) in points:
                jsonData['roi'].append([x/frame.shape[1], y/frame.shape[0]])
            with open(filePath, 'w') as f:
                json.dump(jsonData, f, indent=4)

    cap.release()
    cv2.destroyAllWindows()
    socket.close()
    sys.exit(0)
