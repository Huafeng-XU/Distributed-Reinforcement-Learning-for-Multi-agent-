import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from mainMultiple import *
import rospy
from std_msgs.msg import Float64, UInt8, Bool
from geometry_msgs.msg import Twist
from PyQt5.QtCore import *


class Mythread(QThread):
    linSignal = pyqtSignal(str)
    andSignal = pyqtSignal(str)
    proSignal = pyqtSignal(int)
    lin_speed='0.011'
    ang_speed='0.001'
    lane_action=1

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        print('enter Mythread run')
        print(self.lin_speed)
        print(self.ang_speed)
        print(self.lane_action)

        self.linSignal.emit(str(self.lin_speed))
        self.andSignal.emit(str(self.ang_speed))
        self.proSignal.emit(self.lane_action)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        self.thread1=Mythread()
        self.thread2=Mythread()
        self.thread3=Mythread()
        self.thread4=Mythread()
        # self.thread1.breakSignal.connect(self.setTxt)
        self.progressBars1=[self.progressBar_6, self.progressBar_7, self.progressBar_5, self.progressBar_8]
        self.progressBars2=[self.progressBar_10, self.progressBar_11, self.progressBar_9, self.progressBar_12]
        self.progressBars3=[self.progressBar_14, self.progressBar_15, self.progressBar_13, self.progressBar_16]
        self.progressBars4=[self.progressBar_18, self.progressBar_19, self.progressBar_17, self.progressBar_20]
        self.last_action1=1
        self.last_action2=1
        self.last_action3=1
        self.last_action4=1

    def setLinear1(self,lin_speed):
        self.textEdit_3.setText(lin_speed)

    def setAngular1(self, ang_speed):
        self.textEdit_4.setText(ang_speed)

    def setProgress1(self, lane_action):
        self.progressBars1[self.last_action1].setProperty("value", 0)
        self.progressBars1[lane_action].setProperty("value", 99)
        self.last_action1=lane_action

    def setLinear2(self,lin_speed):
        self.textEdit_5.setText(lin_speed)

    def setAngular2(self, ang_speed):
        self.textEdit_6.setText(ang_speed)

    def setProgress2(self, lane_action):
        self.progressBars2[self.last_action2].setProperty("value", 0)
        self.progressBars2[lane_action].setProperty("value", 99)
        self.last_action2 = lane_action

    def setLinear3(self,lin_speed):
        self.textEdit_7.setText(lin_speed)

    def setAngular3(self, ang_speed):
        self.textEdit_8.setText(ang_speed)

    def setProgress3(self, lane_action):
        self.progressBars3[self.last_action3].setProperty("value", 0)
        self.progressBars3[lane_action].setProperty("value", 99)
        self.last_action3 = lane_action

    def setLinear4(self,lin_speed):
        self.textEdit_9.setText(lin_speed)

    def setAngular4(self, ang_speed):
        self.textEdit_10.setText(ang_speed)

    def setProgress4(self, lane_action):
        self.progressBars4[self.last_action4].setProperty("value", 0)
        self.progressBars4[lane_action].setProperty("value", 99)
        self.last_action4 = lane_action

    def main(self):
        print('enter main')
        self.thread1.linSignal.connect(self.setLinear1)
        self.thread1.andSignal.connect(self.setAngular1)
        self.thread1.proSignal.connect(self.setProgress1)

        self.thread2.linSignal.connect(self.setLinear2)
        self.thread2.andSignal.connect(self.setAngular2)
        self.thread2.proSignal.connect(self.setProgress2)

        self.thread3.linSignal.connect(self.setLinear3)
        self.thread3.andSignal.connect(self.setAngular3)
        self.thread3.proSignal.connect(self.setProgress3)

        self.thread4.linSignal.connect(self.setLinear4)
        self.thread4.andSignal.connect(self.setAngular4)
        self.thread4.proSignal.connect(self.setProgress4)


app = QApplication(sys.argv)
myWin = MyWindow()

def speedCallback1(twist):
    print('enter speed callback')
    myWin.thread1.lin_speed="%.4f" % twist.linear.x
    myWin.thread1.ang_speed="%.4f" % twist.angular.z
    myWin.thread1.run()

def lanCallback1(lane_action_msg):
    print('enter laneCallback: ', lane_action_msg)
    myWin.thread1.lane_action=lane_action_msg.data

def speedCallback2(twist):
    print('enter speed callback')
    myWin.thread2.lin_speed="%.4f" % twist.linear.x
    myWin.thread2.ang_speed="%.4f" % twist.angular.z
    myWin.thread2.run()

def lanCallback2(lane_action_msg):
    print('enter laneCallback: ', lane_action_msg)
    myWin.thread2.lane_action=lane_action_msg.data


def speedCallback3(twist):
    print('enter speed callback')
    myWin.thread3.lin_speed="%.4f" % twist.linear.x
    myWin.thread3.ang_speed="%.4f" % twist.angular.z
    myWin.thread3.run()

def lanCallback3(lane_action_msg):
    print('enter laneCallback: ', lane_action_msg)
    myWin.thread3.lane_action=lane_action_msg.data

def speedCallback4(twist):
    print('enter speed callback')
    myWin.thread4.lin_speed="%.4f" % twist.linear.x
    myWin.thread4.ang_speed="%.4f" % twist.angular.z
    myWin.thread4.run()

def lanCallback4(lane_action_msg):
    print('enter laneCallback: ', lane_action_msg)
    myWin.thread4.lane_action=lane_action_msg.data

if __name__ == '__main__':
    rospy.init_node('car-gui-multiple', anonymous=True)
    myWin.show()
    myWin.main()
    sub_speed1 = rospy.Subscriber('/robot0/cmd_vel', Twist, speedCallback1, queue_size=1)
    sub_lane_action1 = rospy.Subscriber('/robot0/lane_action', UInt8, lanCallback1, queue_size=1)
    sub_speed2 = rospy.Subscriber('/robot2/cmd_vel', Twist, speedCallback2, queue_size=1)
    sub_lane_action2 = rospy.Subscriber('/robot2/lane_action', UInt8, lanCallback2, queue_size=1)
    sub_speed3 = rospy.Subscriber('/robot3/cmd_vel', Twist, speedCallback3, queue_size=1)
    sub_lane_action3 = rospy.Subscriber('/robot3/lane_action', UInt8, lanCallback3, queue_size=1)
    sub_speed4 = rospy.Subscriber('/robot4/cmd_vel', Twist, speedCallback4, queue_size=1)
    sub_lane_action4 = rospy.Subscriber('/robot4/lane_action', UInt8, lanCallback4, queue_size=1)
    #myWin.textEdit_7.setText('55')
    sys.exit(app.exec_())
