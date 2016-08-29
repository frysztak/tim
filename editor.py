#!/usr/bin/env python

import sys, json, collections
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QWidget, QFormLayout, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QLabel)
from nanomsg import Socket, PAIR 

socket = Socket(PAIR)

class Dialog(QDialog):
    def __init__(self, jsonPath, jsonData):
        super(Dialog, self).__init__()
        self.jsonPath = jsonPath
        self.jsonData = jsonData
        blacklist = ["initialVariance", "initialWeight", "learningRate", "medianFilterSize", "roi"]
        
        mainLayout = QVBoxLayout()
        for key, value in [x for x in jsonData.items() if not isinstance(x[1], str) and x[0] not in blacklist]:
            mainLayout.addWidget(self.createWidget(key, value))
        self.setLayout(mainLayout)
   
    def createWidget(self, key, value):
        widget = widget2 = QWidget()
        layout = QFormLayout()
        
        if isinstance(value, bool):
            widget2 = QCheckBox()
            widget2.setChecked(value)
            widget2.toggled.connect(self.valueChangedHandler)
        else:
            widget2 = QDoubleSpinBox() if isinstance(value, float) else QSpinBox()
            if isinstance(value, float):
                widget2.setSingleStep(0.01)
            widget2.setRange(0, 300)
            widget2.setValue(value)
            widget2.valueChanged.connect(self.valueChangedHandler)
            
        widget2.setProperty('key', key)
        layout.addRow(QLabel(key), widget2)
        widget.setLayout(layout)
        return widget

    def valueChangedHandler(self):
        sender = self.sender()
        key = sender.property('key')
        self.jsonData[key] = sender.isChecked() if isinstance(sender, QCheckBox) else sender.value()
        socket.send(json.dumps(jsonData))

        with open(self.jsonPath, 'w') as f:
            json.dump(jsonData, f, indent=4)

if __name__ == '__main__':
    # open json file
    msgShown = False
    while True:
        try:
            with open('/tmp/tim.path') as path:
                filePath = path.readline()
                if msgShown:
                    print('oh, there it is')
                break
        except:
            if not msgShown:
                print('waiting for magical file to be materialised...')
                msgShown = True

    with open(filePath) as f:
        jsonData = json.load(f, object_pairs_hook=collections.OrderedDict)

    socket.bind('ipc:///tmp/tim.ipc')

    # Create main app
    myApp = QApplication(sys.argv)
    myApp.aboutToQuit.connect(lambda: socket.close())
    dialog = Dialog(filePath, jsonData)
    dialog.show()
    sys.exit(myApp.exec_())

