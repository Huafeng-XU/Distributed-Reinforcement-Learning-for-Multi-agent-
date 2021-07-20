import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from mainMultiple import *
import rospy
from std_msgs.msg import Float64, UInt8, Bool
from geometry_msgs.msg import Twist
from PyQt5.QtCore import *


class Mythread(QThread):
    # 定义信号,定义参数为str类型
    breakSignal = pyqtSignal(str)

    def __init__(self, txt, parent=None):
        super().__init__(parent)
        self.txt=txt

    def run(self):
        # 要定义的行为，比如开始一个活动什么的
        print('enter Mythread run')
        self.breakSignal.emit('42342')



class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        # self.sub_speed=rospy.Subscriber('/robot3/cmd_vel',Twist,self.speedCallback3,queue_size=1)
        # self.sub_lane_action3 = rospy.Subscriber('/robot3/lane_action', UInt8, self.lanCallback3,queue_size=1)
        self.setupUi(self)
        # self.setSpeed('1111')

        # self.progressBar.setProperty("value", 99)
        # self.lin_texts=[self.textEdit_3,self.textEdit_5,self.textEdit_7,self.textEdit_9]
        # self.ang_texts=[self.textEdit_4,self.textEdit_6,self.textEdit_8,self.textEdit_10

    # def speedCallback3(self,msg):
    #     print('enter speed callback')
    #     #self.textEdit_7.setText('fdfsdf')
    #     #setTxt('12121212')
    #     mythred=Mythread('11234')
    #     mythred.breakSignal.connect(setTxt)
    #     mythred.start()
        #self.setTxt('111')
        # self.textEdit_7.setText(twist.linear.x)
        #QApplication.processEvents()
        #self.textEdit_8.setText(twist.angular.z)

app = QApplication(sys.argv)
myWin = MyWindow()


def speedCallback3(msg):
    print('enter speed callback')
    #myWin.textEdit_7.setText('11111')
    # self.textEdit_7.setText('fdfsdf')
    # setTxt('12121212')
    mythred = Mythread('11234')
    mythred.breakSignal.connect(setTxt)
    mythred.start()
    mythred.sleep(2)

def setTxt(txt):
    myWin.textEdit_7.setText('1111')

if __name__ == '__main__':
    rospy.init_node('car-gui-multiple', anonymous=True)
    myWin.show()
    sub_speed = rospy.Subscriber('/robot3/cmd_vel', Twist, speedCallback3, queue_size=1)
    #myWin.textEdit_7.setText('55')
    sys.exit(app.exec_())