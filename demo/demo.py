import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPlainTextEdit, QStyleFactory, QTableWidget,
                            QAbstractItemView, QTableWidgetItem, QGridLayout, QPushButton, QCheckBox, QComboBox, QHeaderView, QGridLayout,
                            QSpacerItem, QSizePolicy)
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, QWaitCondition, QMutex, Qt
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QFont
import torch
import pandas as pd
import numpy as np
import time

import model
# import qtmodern.styles
# import random


class MyThread(QThread):
    # 시그널 선언
    change_value = pyqtSignal(object)

    def __init__(self):
        QThread.__init__(self)
        self.cond = QWaitCondition()
        self.mutex = QMutex()
        self._status = False
        self._attack = False
        self._read_speed = 1000
        self.consume = dict()
        self.consume['stop'] = 'Stop!!'

    def __del__(self):
        self.wait()

    # 추론 및 기록 시작
    def run(self):
        net = model.OneNet(self.packet_num)
        # net.load_state_dict(torch.load('model_weight_%d.pth' % self.packet_num))
        net.load_state_dict(torch.load('99.pth', map_location='cpu'))
        net.to(self.device)
        net.eval()

        packet_state = torch.zeros(1, model.STATE_DIM).to(self.device)
        inference_count = 0
        accuracy = 0.0
        normal_idx = 0
        abnormal_idx = 0

        te_no_load = np.load('./fuzzy_tensor_normal_numpy.npy')
        te_ab_load = np.load('./fuzzy_tensor_abnormal_numpy.npy')
        no_load = np.load('./fuzzy_normal_numpy.npy')
        ab_load = np.load('./fuzzy_abnormal_numpy.npy')

        while True:
            self.mutex.lock()

            if not self._status:
                self.consume['type'] = 'end'
                self.change_value.emit(self.consume)
                self.cond.wait(self.mutex)
            
            if not self._attack:
                inputs = torch.from_numpy(te_no_load[normal_idx]).float()
                labels = 1
            else:
                inputs = torch.from_numpy(te_ab_load[abnormal_idx]).float()
                labels = 0
            
            inputs = inputs.to(self.device)

            with torch.no_grad():
                time_temp = time.time()
                outputs, packet_state = net(inputs, packet_state)
                time_temp = time.time() - time_temp
                packet_state = torch.autograd.Variable(packet_state, requires_grad=False)

                _, preds = torch.max(outputs, 1)

                inference_count += 1
                # print(preds.item(), labels)
                if preds.item() == labels:
                    self.consume['check'] = 'ok'
                    accuracy += 1.0
                else:
                    self.consume['check'] = 'no'
                    accuracy += 0.0
            
            self.consume['type'] = 'start'
            self.consume['acc'] = accuracy / inference_count * 100.0
            self.consume['time'] = round(time_temp, 6)

            # 반복
            if not self._attack:
                self.consume['packet'] = no_load[normal_idx]
                normal_idx += 1
                if normal_idx == len(no_load):
                    normal_idx = 0
            else:
                self.consume['packet'] = ab_load[abnormal_idx]
                abnormal_idx += 1
                if abnormal_idx == len(ab_load):
                    abnormal_idx = 0
        
            self.change_value.emit(self.consume)
            self.msleep(self._read_speed)  # QThread에서 제공하는 sleep     

            self.mutex.unlock()

    def toggle_status(self):
        self._status = not self._status
        if self._status:
            self.cond.wakeAll()

    def toggle_attack(self):
        self._attack = not self._attack

    def parameter(self, packet_num, device):
        self.packet_num = packet_num
        self.device = device

    def set_speed(self, value):
        self._read_speed = int(value)

    @property
    def status(self):
        return self._status


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.prev_packet_num = 0
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Detection")
        self.resize(740, 400)

        # 메인 수평 레이아웃
        self.main_horizontalLayout = QtWidgets.QHBoxLayout()

        # 왼쪽 수직 레이아웃
        self.left_verticalLayout = QtWidgets.QVBoxLayout()

        # 패킷 보여줄 곳
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        
        # self.packet_area = QPlainTextEdit()
        # self.scrollArea.setWidget(self.packet_area)

        # 테이블 시작
        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)

        # row, column 갯수 설정해야만 tablewidget 사용할수있다.
        self.table.setColumnCount(10)
        self.table.setRowCount(0)

        # column header
        self.table.setHorizontalHeaderLabels(["ID"])
        self.table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignCenter)  # header 정렬 방식
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # edit 금지 모드

        self.table.setShowGrid(False)  # grid line 숨기기
        self.table.verticalHeader().setVisible(False)  # row header 숨기기
        # 테이블 끝

        self.scrollArea.setWidget(self.table)
        
        self.left_verticalLayout.addWidget(self.scrollArea)
        #
        
        # 정확도 보여줄 곳
        self.accuracy_horizontalLayout = QtWidgets.QHBoxLayout()

        self.accuracy_groupBox = QtWidgets.QGroupBox()
        self.accuracy_groupBox.setTitle("Log")

        self.accuracy_formLayout = QtWidgets.QGridLayout()
        # self.accuracy_formLayout.setRowStretch(0, 1)
        # self.accuracy_formLayout.setRowStretch(2, 1)
        # self.accuracy_formLayout.setRowStretch(4, 1)

        self.now_accuracy = QLabel("?")
        self.accuracy_formLayout.addWidget(QLabel("Accuracy:"), 0, 0)
        self.accuracy_formLayout.addWidget(self.now_accuracy, 0, 1)

        self.now_inference_time = QLabel("?")
        self.accuracy_formLayout.addWidget(QLabel("Inference Time:"), 1, 0)
        self.accuracy_formLayout.addWidget(self.now_inference_time, 1, 1)
        self.accuracy_formLayout.setAlignment(Qt.AlignLeft)

        self.accuracy_groupBox.setLayout(self.accuracy_formLayout)

        self.accuracy_horizontalLayout.addWidget(self.accuracy_groupBox)

        self.left_verticalLayout.addLayout(self.accuracy_horizontalLayout)
        self.left_verticalLayout.setStretchFactor(self.scrollArea, 3)
        self.left_verticalLayout.setStretchFactor(self.accuracy_horizontalLayout, 1)
        #

        # 왼쪽 끝
        self.main_horizontalLayout.addLayout(self.left_verticalLayout)

        # 오른쪽 시작
        # 오른쪽 수직 레이아웃
        self.right_verticalLayout = QtWidgets.QVBoxLayout()

        # 읽을 패킷 숫자
        self.parameter_groupBox = QtWidgets.QGroupBox()
        self.parameter_groupBox.setTitle("Parameter")

        # group 박스 안에 grid
        self.parameter_formLayout = QtWidgets.QGridLayout()

        self.packet_num_line = QLineEdit()
        self.parameter_formLayout.addWidget(QLabel("Packet num:"), 0, 0)
        self.parameter_formLayout.addWidget(self.packet_num_line, 0, 1)

        self.parameter_formLayout.addWidget(QLabel("(1 ~ 1)"), 1, 0)
        self.parameter_formLayout.addWidget(QLabel(""), 2, 0) # grid spacing ...?

        # csv 읽는 속도 선택용
        self.time_combo = QComboBox()
        self.time_combo.addItems(["0.25s", "0.5s", "1.0s", "0.1s"])
        self.parameter_formLayout.addWidget(QLabel("Packet read speed:"), 3, 0)
        self.parameter_formLayout.addWidget(self.time_combo, 3, 1)

        # 버튼
        self.start_pushButton = QtWidgets.QPushButton("Start")
        self.start_pushButton.setCheckable(True)
        self.start_pushButton.toggled.connect(self.start_toggle)

        self.attack_pushButton = QtWidgets.QPushButton("Attack")
        self.attack_pushButton.setCheckable(True)
        self.attack_pushButton.toggled.connect(self.attack_toggle)

        self.parameter_formLayout.addWidget(QLabel(""), 4, 0) # grid spacing ...?
        self.parameter_formLayout.addWidget(QLabel(""), 5, 0) # grid spacing ...?
        # self.parameter_formLayout.setRowStretch(4, 1)
        
        # self.parameter_formLayout.setRowStretch(2, 1)
        # vspacer = QtGui.QSpacerItem(
        #     QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        # layout.addItem(vspacer, last_row, 0, 1, -1)

        # hspacer = QtGui.QSpacerItem(
        #     QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        # layout.addItem(hspacer, 0, last_column, -1, 1)

        self.parameter_formLayout.addWidget(self.start_pushButton, 6, 0)
        self.parameter_formLayout.addWidget(self.attack_pushButton, 6, 1)

        self.parameter_formLayout.setRowStretch(7, 1)
        # self.parameter_formLayout.setVerticalSpacing(50)
        # self.parameter_formLayout.setContentsMargins(5, 5, 5, 5) # left, top, right, bottom

        self.parameter_groupBox.setLayout(self.parameter_formLayout)

        self.right_verticalLayout.addWidget(self.parameter_groupBox)

        self.main_horizontalLayout.addLayout(self.right_verticalLayout)
        self.main_horizontalLayout.setStretchFactor(self.left_verticalLayout, 2)
        self.main_horizontalLayout.setStretchFactor(self.right_verticalLayout, 1)
        # 오른쪽 끝

        self.setLayout(self.main_horizontalLayout)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.show()

    def start_demo(self):
        # 입력 확인
        packet_num = self.packet_num_line.text()
        if packet_num == '':
            print("Empty Value Not Allowed")
            self.packet_num_line.setFocus()
            return
        
        packet_num = int(packet_num)
        if packet_num < 1 or packet_num > 1:
            print("too many packet")
            self.packet_num_line.setFocus()
            return
        else:
            self.packet_num_line.clearFocus()

        # 초기화
        self.add_spanRow_text('Start!! Please wait')
        
        # 읽는 패킷이 달라졌음, 새로 시작
        if self.prev_packet_num != packet_num:
            self.prev_packet_num = packet_num
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.predict_thread = MyThread()
            self.predict_thread.parameter(packet_num, device)
            self.predict_thread.change_value.connect(self.update_line_edit)
            self.predict_thread.start()

        csv_read_speed = float(self.time_combo.currentText()[:-1])
        self.predict_thread.set_speed(csv_read_speed * 1000)


    def update_line_edit(self, consume):
        if consume['type'] == 'start':
            self.now_accuracy.setText(str(consume['acc']))
            self.now_inference_time.setText(str(consume['time']))

            if consume['check'] == 'ok': # 맞춤
                color = QtGui.QColor(150, 255, 150) # Red, Green, Blue, Alpha
            else:
                color = QtGui.QColor(255, 150, 150)

            next_row = self.table.rowCount()
            self.table.insertRow(next_row)   # row 추가
            col_idx = 0
            for consume_packet in consume['packet']:
                self.table.setItem(next_row, col_idx, QTableWidgetItem(str(consume_packet)))
                self.table.item(next_row, col_idx).setBackground(color)
                col_idx += 1
            self.table.scrollToBottom()
        else:
            self.add_spanRow_text(consume['stop'])

    def add_row_text(self, text):
        next_row = self.table.rowCount()
        self.table.insertRow(next_row)   # row 추가
        self.table.setItem(next_row, 0, QTableWidgetItem(text))
        self.table.scrollToBottom()

    def add_spanRow_text(self, text):
        next_row = self.table.rowCount()
        self.table.insertRow(next_row)   # row 추가
        self.table.setSpan(next_row, 0, 1, 10)  # 1 x 10 크기의 span 생성
        self.table.setItem(next_row, 0, QTableWidgetItem(text))
        self.table.scrollToBottom()

    @pyqtSlot(bool)
    def start_toggle(self, state):
        # self.start_pushButton.setStyleSheet("background-color: %s" % ({True: "green", False: "red"}[state]))
        self.start_pushButton.setText({True: "Stop", False: "Start"}[state])
        self.packet_num_line.setEnabled({True: False, False: True}[state])
        if state:
            self.start_demo()
        else:
            # self.packet_area.appendPlainText('Trying to stop..')
            self.add_spanRow_text('Trying to stop..')
        self.predict_thread.toggle_status()

    @pyqtSlot(bool)
    def attack_toggle(self, state):
        # self.attack_pushButton.setStyleSheet("background-color: %s" % ({True: "green", False: "red"}[state]))
        self.attack_pushButton.setText({True: "Stop", False: "Attack"}[state])
        self.predict_thread.toggle_attack()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    # qtmodern.styles.light(app)
    # app.setStyle(QStyleFactory.create('Fusion'))
    ex = MyApp()
    sys.exit(app.exec_())
    
