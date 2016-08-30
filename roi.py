#!/usr/bin/env python

import sys, json, collections, os
from nanomsg import Socket, PAIR
import cv2
import numpy as np
from enum import Enum

class Modes(Enum):
    NORMAL = 0
    POLYGON = 1
    LINE1 = 2
    LINE2 = 3

if __name__ == '__main__':
    socket = Socket(PAIR)
    socket.connect('ipc:///tmp/tim.ipc')
    filePath = '/mnt/things/tim/json/lausanne.json' #socket.recv()
    # open json file
    with open(filePath) as f:
        jsonData = json.load(f, object_pairs_hook=collections.OrderedDict)

    currentMode = Modes.NORMAL
    cap = cv2.VideoCapture(os.path.join('/mnt/things/tim/videos/', jsonData['video']))
    ret, frame = cap.read()
    mask = np.zeros(frame.shape, np.uint8)
    polygonPoints = []
    linesPoints = [[], []]
    for (x,y) in jsonData['roi']:
        polygonPoints.append([int(x*frame.shape[1]), int(y*frame.shape[0])])
    for (x,y) in jsonData['lines'][0:2]:
        linesPoints[0].append([int(x*frame.shape[1]), int(y*frame.shape[0])])
    for (x,y) in jsonData['lines'][2:4]:
        linesPoints[1].append([int(x*frame.shape[1]), int(y*frame.shape[0])])
    movedPointIdx = -1
    movedLineIdx = -1
    
    def generateMask():
        global mask, polygonPoints, movedPointIdx
        mask = np.zeros(frame.shape, np.uint8)
        # sorting may change index of currently selected point
        if movedPointIdx != -1:
            savedPoint = polygonPoints[movedPointIdx]

        # use convex hull to sort polygonPoints
        sortedPoints = cv2.convexHull(np.asarray(polygonPoints)).tolist()
        # convex hull, besides sorting, for some reason returns polygonPoints wrapped in a list twice.
        # for example: when polygonPoints = [[710, 359], [602, 361], [545, 719], [913, 720]],
        #             sortedPoints = [[[913, 720]], [[545, 719]], [[602, 361]], [[710, 359]]]
        # let's fix that.
        polygonPoints.clear()
        for [p] in sortedPoints:
            polygonPoints.append([p[0], p[1]])
        poly = cv2.approxPolyDP(np.asarray(polygonPoints), 1.0, True)
        cv2.fillConvexPoly(mask, poly.astype(np.int32), (255,255,255))

        # update idx
        for i, (x_,y_) in enumerate(polygonPoints):
            if movedPointIdx != -1 and savedPoint == [x_, y_]:
                movedPointIdx = i
                break

    def mouseCallback(event, x, y, flags, param):
        global movedPointIdx, movedLineIdx

        if event == cv2.EVENT_LBUTTONDBLCLK:
            if currentMode is Modes.POLYGON:
                polygonPoints.append([x,y])
                generateMask()
            elif currentMode is not Modes.NORMAL:
                lineIdx = 0 if currentMode is Modes.LINE1 else 1
                if len(linesPoints[lineIdx]) < 2:
                    linesPoints[lineIdx].append([x,y])

        elif event == cv2.EVENT_LBUTTONDOWN and movedPointIdx == -1:
            # first check polygon
            for i, (x_,y_) in enumerate(polygonPoints):
                if abs(x-x_) <= 12 and abs(y-y_) <= 12:
                    movedPointIdx = i
                    return

            # now check lines
            for lineIdx, line in enumerate(linesPoints):
                for i, (x_,y_) in enumerate(line):
                    if abs(x-x_) <= 12 and abs(y-y_) <= 12:
                        movedPointIdx = i
                        movedLineIdx = lineIdx
                        break

        elif event == cv2.EVENT_MOUSEMOVE and movedPointIdx != -1:
            if movedLineIdx == -1:
                polygonPoints[movedPointIdx] = [x, y]
                generateMask()
            else:
                linesPoints[movedLineIdx][movedPointIdx] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP and movedPointIdx != -1:
            movedPointIdx = -1
            movedLineIdx = -1

    def updateWindowTitle():
        name = 'ROI editor'
        if currentMode is Modes.POLYGON:
            name += ' - polygon'
        elif currentMode is Modes.LINE1:
            name += ' - 1st line'
        elif currentMode is Modes.LINE2:
            name += ' - 2nd line'
        cv2.setWindowTitle('OpenCV', name)

    cv2.namedWindow('OpenCV')
    cv2.setMouseCallback('OpenCV', mouseCallback)
    generateMask()
    updateWindowTitle()

    while True:
        disp = frame.copy()
        for (x,y) in polygonPoints:
            cv2.circle(disp, (x,y), 8, (255,0,0), -1)
        disp = cv2.add(disp, mask)

        for i, line in enumerate(linesPoints):
            for (x,y) in line:
                cv2.circle(disp, (x,y), 8, (0,255,0), -1)
            if len(line) == 2:
                cv2.line(disp, tuple(line[0]), tuple(line[1]), (0,255,0), 2)
                cv2.putText(disp, 'line ' + str(i), tuple(line[0]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255))

        cv2.imshow('OpenCV', disp)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

        elif key == 27: # ESC key
            currentMode = Modes.NORMAL
            movedPointIdx = -1

        elif key == ord('p') and currentMode is Modes.NORMAL:
            currentMode = Modes.POLYGON

        elif key == ord('l'):
            if currentMode is Modes.NORMAL:
                currentMode = Modes.LINE1
            elif currentMode is Modes.LINE1:
                currentMode = Modes.LINE2

        elif key == ord('c'):
            if currentMode is Modes.POLYGON:
                mask = np.zeros(frame.shape, np.uint8)
                polygonPoints.clear()
            elif currentMode is Modes.LINE1:
                linesPoints[0].clear()
            elif currentMode is Modes.LINE2:
                linesPoints[1].clear()

        elif key == ord('s'):
            jsonData['roi'] = []
            jsonData['lines'] = []

            for (x, y) in polygonPoints:
                jsonData['roi'].append([x/frame.shape[1], y/frame.shape[0]])

            if len(linesPoints[0]) + len(linesPoints[1]) == 4:
                for line in linesPoints:
                    for (x, y) in line:
                        jsonData['lines'].append([x/frame.shape[1], y/frame.shape[0]])

            with open(filePath, 'w') as f:
                json.dump(jsonData, f, indent=4)

        updateWindowTitle()

    cap.release()
    cv2.destroyAllWindows()
    socket.close()
    sys.exit(0)
