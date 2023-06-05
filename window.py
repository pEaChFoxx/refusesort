# -*- coding: UTF-8 –*-
import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import json
import urllib.request
from PIL import Image
import numpy as np
import shutil


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('垃圾智能分类系统')
        self.model = tf.keras.models.load_model("models/mobilenet_245_epoch30.h5")  # todo 修改为自己的模型路径
        self.to_predict_name = "images/cc.jpeg"
        self.class_names = ['其他垃圾_PE塑料袋', '其他垃圾_U型回形针', '其他垃圾_一次性杯子', '其他垃圾_一次性棉签', '其他垃圾_串串竹签', '其他垃圾_便利贴', '其他垃圾_创可贴',
                            '其他垃圾_卫生纸', '其他垃圾_厨房手套', '其他垃圾_厨房抹布', '其他垃圾_口罩', '其他垃圾_唱片', '其他垃圾_图钉', '其他垃圾_大龙虾头',
                            '其他垃圾_奶茶杯', '其他垃圾_干燥剂', '其他垃圾_彩票', '其他垃圾_打泡网', '其他垃圾_打火机', '其他垃圾_搓澡巾', '其他垃圾_果壳', '其他垃圾_毛巾',
                            '其他垃圾_涂改带', '其他垃圾_湿纸巾', '其他垃圾_烟蒂', '其他垃圾_牙刷', '其他垃圾_电影票', '其他垃圾_电蚊香', '其他垃圾_百洁布', '其他垃圾_眼镜',
                            '其他垃圾_眼镜布', '其他垃圾_空调滤芯', '其他垃圾_笔', '其他垃圾_胶带', '其他垃圾_胶水废包装', '其他垃圾_苍蝇拍', '其他垃圾_茶壶碎片',
                            '其他垃圾_草帽', '其他垃圾_菜板', '其他垃圾_车票', '其他垃圾_酒精棉', '其他垃圾_防霉防蛀片', '其他垃圾_除湿袋', '其他垃圾_餐巾纸',
                            '其他垃圾_餐盒', '其他垃圾_验孕棒', '其他垃圾_鸡毛掸', '厨余垃圾_八宝粥', '厨余垃圾_冰激凌', '厨余垃圾_冰糖葫芦', '厨余垃圾_咖啡',
                            '厨余垃圾_圣女果', '厨余垃圾_地瓜', '厨余垃圾_坚果', '厨余垃圾_壳', '厨余垃圾_巧克力', '厨余垃圾_果冻', '厨余垃圾_果皮', '厨余垃圾_核桃',
                            '厨余垃圾_梨', '厨余垃圾_橙子', '厨余垃圾_残渣剩饭', '厨余垃圾_水果', '厨余垃圾_泡菜', '厨余垃圾_火腿', '厨余垃圾_火龙果', '厨余垃圾_烤鸡',
                            '厨余垃圾_瓜子', '厨余垃圾_甘蔗', '厨余垃圾_番茄', '厨余垃圾_秸秆杯', '厨余垃圾_秸秆碗', '厨余垃圾_粉条', '厨余垃圾_肉类', '厨余垃圾_肠',
                            '厨余垃圾_苹果', '厨余垃圾_茶叶', '厨余垃圾_草莓', '厨余垃圾_菠萝', '厨余垃圾_菠萝蜜', '厨余垃圾_萝卜', '厨余垃圾_蒜', '厨余垃圾_蔬菜',
                            '厨余垃圾_薯条', '厨余垃圾_薯片', '厨余垃圾_蘑菇', '厨余垃圾_蛋', '厨余垃圾_蛋挞', '厨余垃圾_蛋糕', '厨余垃圾_豆', '厨余垃圾_豆腐',
                            '厨余垃圾_辣椒', '厨余垃圾_面包', '厨余垃圾_饼干', '厨余垃圾_鸡翅', '可回收物_不锈钢制品', '可回收物_乒乓球拍', '可回收物_书', '可回收物_体重秤',
                            '可回收物_保温杯', '可回收物_保鲜膜内芯', '可回收物_信封', '可回收物_充电头', '可回收物_充电宝', '可回收物_充电牙刷', '可回收物_充电线',
                            '可回收物_凳子', '可回收物_刀', '可回收物_包', '可回收物_单车', '可回收物_卡', '可回收物_台灯', '可回收物_吊牌', '可回收物_吹风机',
                            '可回收物_呼啦圈', '可回收物_地球仪', '可回收物_地铁票', '可回收物_垫子', '可回收物_塑料制品', '可回收物_太阳能热水器', '可回收物_奶粉桶',
                            '可回收物_尺子', '可回收物_尼龙绳', '可回收物_布制品', '可回收物_帽子', '可回收物_手机', '可回收物_手电筒', '可回收物_手表', '可回收物_手链',
                            '可回收物_打包绳', '可回收物_打印机', '可回收物_打气筒', '可回收物_扫地机器人', '可回收物_护肤品空瓶', '可回收物_拉杆箱', '可回收物_拖鞋',
                            '可回收物_插线板', '可回收物_搓衣板', '可回收物_收音机', '可回收物_放大镜', '可回收物_日历', '可回收物_暖宝宝', '可回收物_望远镜',
                            '可回收物_木制切菜板', '可回收物_木桶', '可回收物_木棍', '可回收物_木质梳子', '可回收物_木质锅铲', '可回收物_木雕', '可回收物_枕头',
                            '可回收物_果冻杯', '可回收物_桌子', '可回收物_棋子', '可回收物_模具', '可回收物_毯子', '可回收物_水壶', '可回收物_水杯', '可回收物_沙发',
                            '可回收物_泡沫板', '可回收物_灭火器', '可回收物_灯罩', '可回收物_烟灰缸', '可回收物_热水瓶', '可回收物_燃气灶', '可回收物_燃气瓶',
                            '可回收物_玩具', '可回收物_玻璃制品', '可回收物_玻璃器皿', '可回收物_玻璃壶', '可回收物_玻璃球', '可回收物_瑜伽球', '可回收物_电动剃须刀',
                            '可回收物_电动卷发棒', '可回收物_电子秤', '可回收物_电熨斗', '可回收物_电磁炉', '可回收物_电脑屏幕', '可回收物_电视机', '可回收物_电话',
                            '可回收物_电路板', '可回收物_电风扇', '可回收物_电饭煲', '可回收物_登机牌', '可回收物_盒子', '可回收物_盖子', '可回收物_盘子', '可回收物_碗',
                            '可回收物_磁铁', '可回收物_空气净化器', '可回收物_空气加湿器', '可回收物_笼子', '可回收物_箱子', '可回收物_纸制品', '可回收物_纸牌',
                            '可回收物_罐子', '可回收物_网卡', '可回收物_耳套', '可回收物_耳机', '可回收物_衣架', '可回收物_袋子', '可回收物_袜子', '可回收物_裙子',
                            '可回收物_裤子', '可回收物_计算器', '可回收物_订书机', '可回收物_话筒', '可回收物_豆浆机', '可回收物_路由器', '可回收物_轮胎', '可回收物_过滤网',
                            '可回收物_遥控器', '可回收物_量杯', '可回收物_金属制品', '可回收物_钉子', '可回收物_钥匙', '可回收物_铁丝球', '可回收物_铅球',
                            '可回收物_铝制用品', '可回收物_锅', '可回收物_锅盖', '可回收物_键盘', '可回收物_镊子', '可回收物_闹铃', '可回收物_雨伞', '可回收物_鞋',
                            '可回收物_音响', '可回收物_餐具', '可回收物_餐垫', '可回收物_饰品', '可回收物_鱼缸', '可回收物_鼠标', '有害垃圾_指甲油', '有害垃圾_杀虫剂',
                            '有害垃圾_温度计', '有害垃圾_灯', '有害垃圾_电池', '有害垃圾_电池板', '有害垃圾_纽扣电池', '有害垃圾_胶水', '有害垃圾_药品包装', '有害垃圾_药片',
                            '有害垃圾_药瓶', '有害垃圾_药膏', '有害垃圾_蓄电池', '有害垃圾_血压计']
        self.resize(900, 700)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        # left_layout.setAlignment(Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        label_result = QLabel(' 垃圾种类 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))

        label_result_f = QLabel(' 物品名称 ')
        self.result_f = QLabel("等待识别")

        label_result_f.setFont(QFont('楷体', 16))
        self.result_f.setFont(QFont('楷体', 24))

        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(label_result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '主页')
        self.setTabIcon(0, QIcon('images/主页面.png'))

# 拍摄图片
    def change_img(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        flag = cap.isOpened()
        self.index = 0
        while (flag):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):  # 按下s键，进入下面的保存图片操作
                cv2.imwrite("images/show.png", frame)
                target_image_name = "images/show.png"
                self.to_predict_name = target_image_name
                img_init = cv2.imread(self.to_predict_name)
                h, w, c = img_init.shape
                scale = 400 / h
                img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
                cv2.imwrite("images/show.png", img_show)
                img_init = cv2.resize(img_init, (224, 224))
                cv2.imwrite('images/target.png', img_init)
                self.img_label.setPixmap(QPixmap("images/show.png"))
                print("save" + ".jpg successfuly!")
                print("-------------------------")
                self.index += 1
            elif self.index == 1:  # 按下q键，程序退出
                self.index = 0
                break
        cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 释放并销毁窗口

# 预测图片
    def predict_img(self):
        img = Image.open('images/target.png')
        img = np.asarray(img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]
        names = result.split("_")
        # print(result)
        self.result.setText(names[0])
        self.result_f.setText(names[1])


if __name__ == "__main__":
    app = QApplication(sys.argv)   # 每一个 PyQt5 程序都需要有一个 QApplication 对象。sys.argv 是一个命令行参数列表。
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
