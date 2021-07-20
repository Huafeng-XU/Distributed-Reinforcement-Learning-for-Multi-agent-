from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import time

class Mythread(QThread):
    # 定义信号,定义参数为str类型
    _signal = pyqtSignal(str)

    def __init__(self):
        super(Mythread, self).__init__()

    def run(self):
        for i in range(2000000):
            # 发出信号
            self._signal.emit('当前循环值为:%s' % i)
            # 让程序休眠
            time.sleep(0.5)


if __name__ == '__main__':
    app = QApplication([])
    dlg = QDialog()
    dlg.resize(400, 300)
    dlg.setWindowTitle("自定义按钮测试")
    dlgLayout = QVBoxLayout()
    dlgLayout.setContentsMargins(40, 40, 40, 40)
    btn = QPushButton('测试按钮')
    dlgLayout.addWidget(btn)
    dlgLayout.addStretch(40)
    dlg.setLayout(dlgLayout)
    dlg.show()


    def chuli(s):
        dlg.setWindowTitle(s)
        btn.setText(s)

    # 创建线程
    thread = Mythread()
    # 注册信号处理函数
    thread._signal.connect(chuli)
    # 启动线程
    thread.start()
    dlg.exec_()
    app.exit()