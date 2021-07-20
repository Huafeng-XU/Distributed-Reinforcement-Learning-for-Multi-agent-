import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from mainWindow import *
import rospy
from std_msgs.msg import Float64, UInt8, Bool

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.pub_lane_behavior = rospy.Publisher('/robot0/lane_behavior', UInt8, self.lanCallback,queue_size=1)
        self.setupUi(self)
        self.textEdit.setText('1111')
        self.progressBar.setProperty("value", 99)

    def lanCallback(self,msg):
        lane_behavior=msg.data



if __name__ == '__main__':
    rospy.init_node('car-gui', anonymous=True)
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())