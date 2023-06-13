from cmath import sqrt
# from types import NoneType
from PyQt6 import uic
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
import sys,os
import cv2
import numpy as np
import time
import math
import shutil
import json
import pickle
from segment_anything import sam_model_registry, SamPredictor

from utils import *
from dialogs import *
from background import *
from data_define import *


# Todo:
# (Done)完成导出功能
# 增加扩充掩码选项框，如果选中（并输入一个整数x），则对掩码dilate，程度视x而定
# 为扩充掩码选项框和反转掩码选项框增加保存提示
# 增加一个选项框，可选择在segment图像框显示整体分割图，还是只显示当前对象的分割图
class MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super().__init__(parent)
        uic.loadUi("src/mainwindow.ui",self)

        self.path_workspace = "workspace"
        self.path_finished = "workspace_finished"
        self.dir_images = lambda workspace: ensureDir(os.path.join(workspace, "images"))
        self.dir_data = lambda workspace: ensureDir(os.path.join(workspace, "data"))
        self.dir_segments = lambda workspace: ensureDir(os.path.join(workspace, "segments"))
        self.dir_masks = lambda workspace: ensureDir(os.path.join(workspace, "masks"))
        self.dir_masks_instance = lambda workspace: ensureDir(os.path.join(workspace, "masks_instance"))

        self.dir_cache = lambda: ensureDir("cache")
        self.to_cache_path = lambda path: os.path.join(self.dir_cache(), path)
        shutil.rmtree(self.dir_cache(), ignore_errors=True)

        # 左上角的按钮
        self.btnPrevious:QPushButton = self.findChild(QPushButton, "btnPrevious")
        self.btnNext:QPushButton = self.findChild(QPushButton,"btnNext")

        self.btnNext.clicked.connect(self.onBtnNext)
        self.btnPrevious.clicked.connect(self.onBtnPrevious)

        # 文件列表及三个图片显示槽位
        self.listWidget:QListWidget = self.findChild(QListWidget,"listWidget")
        self.lblImage:QLabel = self.findChild(QLabel,"lblImage")
        self.lblMixed:QLabel = self.findChild(QLabel,"lblMixed")
        self.lblSegment:QLabel = self.findChild(QLabel,"lblSegment")

        # 右下角的按钮
        self.wgtOperation:QWidget = self.findChild(QWidget, "wgtOperation")
        self.btnAddObject:QPushButton = self.findChild(QPushButton, "btnAddObject")
        self.btnRemoveObject:QPushButton = self.findChild(QPushButton,"btnRemoveObject")
        self.cbbObject:QComboBox = self.findChild(QComboBox,"cbbObject")
        self.ckbInvertMask:QCheckBox = self.findChild(QCheckBox,"ckbInvertMask")
        self.btnReset:QPushButton = self.findChild(QPushButton, "btnReset")
        self.btnSave:QPushButton = self.findChild(QPushButton,"btnSave")
        self.btnWithdraw:QPushButton = self.findChild(QPushButton, "btnWithdraw")
        self.btnExport:QPushButton = self.findChild(QPushButton,"btnExport")

        self.btnAddObject.clicked.connect(self.onBtnAddObject)
        self.btnRemoveObject.clicked.connect(self.onBtnRemoveObject)
        self.cbbObject.currentIndexChanged.connect(self.onCbbObjectChanged)
        self.ckbInvertMask.stateChanged.connect(self.onCkbInvertMask)
        self.btnReset.clicked.connect(self.onBtnReset)
        self.btnSave.clicked.connect(self.onBtnSave)
        self.btnWithdraw.clicked.connect(self.onBtnWithdraw)
        self.btnExport.clicked.connect(self.onBtnExport)



        
        #导出设置相关的控件
        self.lblExportPath:QLabel = self.findChild(QLabel,"lblExportPath")
        self.ckbExportExpand:QCheckBox = self.findChild(QCheckBox,"ckbExportExpand")
        self.ckbExportImageSize:QCheckBox = self.findChild(QCheckBox,"ckbExportImageSize")
        self.lblExportImageSize:QLabel = self.findChild(QLabel,"lblExportImageSize")
        self.ckbExportSegmentSize:QCheckBox = self.findChild(QCheckBox,"ckbExportSegmentSize")
        self.lblExportSegmentSize:QLabel = self.findChild(QLabel,"lblExportSegmentSize")
        self.ckbExportMaskSize:QCheckBox = self.findChild(QCheckBox,"ckbExportMaskSize")
        self.lblExportMaskSize:QLabel = self.findChild(QLabel,"lblExportMaskSize")

        self.ckbExportImageSize.stateChanged.connect(self.onCkbExportImageSize)
        self.ckbExportSegmentSize.stateChanged.connect(self.onCkbExportSegmentSize)
        self.ckbExportMaskSize.stateChanged.connect(self.onCkbExportMaskSize)
        self.lblExportPath.setText(f"./{self.path_finished}")
        
        self.resetData()
        

        self.installEventFilter(self)


        for obj in self.findChildren((QLabel, QSlider, QListWidget,QScrollBar)):
            obj.installEventFilter(self)
            obj.setMouseTracking(True)

        


        self.listWidget.itemSelectionChanged.connect(self.onItemSelectionChanged)
        self.updateListWidget()


        self.background = Background(self)
        self.background.sig_func.connect(self.onSigFunc)
        self.background.sig_predict_finished.connect(self.onPredictFinished)


        self.background.loadSAM()

        self.resizeUI()


    def onSigFunc(self, ls):
        fn, args = ls[0], ls[1:]
        fn(*args)

    def onPredictFinished(self, name, masks, scores, logits, temp_flag):
        if self.idx_item < 0 or name != self.listWidget.item(self.idx_item).text():
            return
        if temp_flag:
            self.temp_masks = masks
            self.drawTempMixedImage()
        else:
            data = self.cbbObject.currentData()
            data.masks = masks
            data.scores = scores
            data.logits = logits
            self.drawImages()

    def onBtnNext(self):
        count = self.listWidget.count()
        idx = min(self.listWidget.currentRow() + 1, count - 1)
        self.listWidget.setCurrentRow(idx)

    def onBtnPrevious(self):
        idx = max(self.listWidget.currentRow() - 1, 0)
        self.listWidget.setCurrentRow(idx)

    def onBtnAddObject(self):
        self.backupData()
        idx = self.ls_data.item(self.ls_data.rowCount()-1).text()
        idx = int(idx)+1
        item = QStandardItem(f"{idx}")
        item.setData(SAMData(), role=Qt.ItemDataRole.UserRole)
        self.ls_data.appendRow(item)
        self.cbbObject.setCurrentIndex(self.cbbObject.count()-1)

    
    def onBtnRemoveObject(self):
        if self.cbbObject.count() <= 1:
            if self.cbbObject.count() == 1:
                InformationDialog.showInfoWithButtons(self, self.tr("提示"), [], [self.tr("不能删除唯一的对象")])
            return
        self.backupData()
        idx = self.cbbObject.currentIndex()
        self.ls_data.removeRow(idx)

    def onCbbObjectChanged(self, i):
        if self.cbbObject.currentIndex() < 0:
            return
        data = self.cbbObject.currentData()
        self.ckbInvertMask.setChecked(data.invertMask)
        self.drawImages()

    def onCkbInvertMask(self):
        if self.cbbObject.currentIndex() < 0:
            return
        data = self.cbbObject.currentData()
        data.invertMask = self.ckbInvertMask.isChecked()
        self.drawImages()

    def onBtnReset(self):
        if self.idx_item < 0:
            return
        self.backupData()
        self.cbbObject.setItemData(self.cbbObject.currentIndex(), SAMData())
        self.drawImages()

    def onBtnSave(self):
        self.saveData()

    def onBtnWithdraw(self):
        self.restoreData()

    def onBtnExport(self):
        self.exportWorkspace(self.path_finished)

    def processCkbInputSize(parent, ckb, lbl):
        if ckb.isChecked():
            width, height = InputDialog.getInputWidthHeight(parent)
            if width is not None and height is not None:
                lbl.setText(f"{width} x {height}")
            else:
                lbl.setText(parent.tr("未生效 (无效输入) "))
            return
        lbl.setText("")
    def getCkbInputSize(self,ckb, lbl):
        if ckb.isChecked():
            try:
                temp = lbl.text().split("x")
                width = int(temp[0])
                height = int(temp[1])
                return width, height
            except:
                return None, None
        return None, None
    def onCkbExportImageSize(self):
        MainWindow.processCkbInputSize(self, self.ckbExportImageSize, self.lblExportImageSize)
    def onCkbExportSegmentSize(self):
        MainWindow.processCkbInputSize(self, self.ckbExportSegmentSize, self.lblExportSegmentSize)
    def onCkbExportMaskSize(self):
        MainWindow.processCkbInputSize(self, self.ckbExportMaskSize, self.lblExportMaskSize)


    def resizeEvent(self, event: QResizeEvent) -> None:
        self.resetImageView()
        self.drawImages()
    
    def closeEvent(self, event):
        if self.is_data_modified:
            # 提示尚未保存
            ans = InformationDialog.confirmInfo(self, title=self.tr("未保存"), ls_info=[self.tr("是否保存更改？")],\
                            txtBtnYes=self.tr("保存"), txtBtnNo=self.tr("放弃"))
            if ans == BasicDialog.Accepted:
                self.saveData()
            elif ans == BasicDialog.Rejected:
                pass
            else:
                return event.ignore()
        self.background.shutdown()
        
    def eventFilter(self, obj, event):
        
        # 监听图像的点击和移动事件
        ls_lbl = [ self.lblMixed, self.lblSegment, self.lblImage]
        if obj in ls_lbl:
            if self.idx_item < 0:
                return super().eventFilter(obj, event)
            if event.type() == QEvent.Type.Enter:
                self.box_start_point=None
                self.hovering=True
            elif event.type() == QEvent.Type.Leave:
                self.hovering=False
                self.drawTempMixedImage()

        if type(event) == QMouseEvent:
            if self.idx_item < 0:
                return super().eventFilter(obj, event)
            data = self.cbbObject.currentData()
            if data == None:
                return super().eventFilter(obj, event)
            mouse_event:QMouseEvent = event
            # Ctrl + 鼠标左键画框，这里是松开左键的逻辑，松开无需Ctrl
            if event.type() == QMouseEvent.Type.MouseButtonRelease:
                
                if mouse_event.button() != Qt.MouseButton.LeftButton:
                    return super().eventFilter(obj, event)
                if self.box_start_point != None and self.box_end_point != None:
                    self.backupData()
                    pts = np.array([self.box_start_point[0], self.box_start_point[1], self.box_end_point[0], self.box_end_point[1]])
                    data.input_box = self.mapCoords(pts, data2view=False)
                    ensurePointsInBox(data.input_points, data.input_labels, data.input_box)
                    self.background.predict(name=self.listWidget.item(self.idx_item).text(), 
                        input_points=data.input_points, 
                        input_labels=data.input_labels, 
                        input_box=data.input_box, 
                        input_mask=data.logits)
                self.box_start_point = None

            if obj not in ls_lbl:
                return super().eventFilter(obj, event)

            
            coords = np.array([[event.pos().x(), event.pos().y()]])
            coords = self.mapCoords(coords, data2view=False)
            
            lbl:QLabel = obj
            if event.type() == QMouseEvent.Type.MouseButtonPress:
                # Ctrl + 鼠标左键画框，这里是按下左键的逻辑
                if event.modifiers()==Qt.KeyboardModifier.ControlModifier:
                    self.box_start_point = [int(event.pos().x()), int(event.pos().y())]
                # 鼠标左键画点，鼠标右键画负面点
                else:
                    self.backupData()
                    label = None
                    
                    if mouse_event.buttons() == Qt.MouseButton.LeftButton:
                        label = 1
                    elif mouse_event.buttons() == Qt.MouseButton.RightButton:
                        label = 0
                    else:
                        return super().eventFilter(obj, event)
                    data.input_points = np.append(data.input_points, coords, axis=0)
                    data.input_labels = np.append(data.input_labels, label)
                    ensurePointsInBox(data.input_points, data.input_labels, data.input_box)
                    self.background.predict(name=self.listWidget.item(self.idx_item).text(), 
                        input_points=data.input_points, 
                        input_labels=data.input_labels, 
                        input_box=data.input_box, 
                        input_mask=data.logits)
                    
            # 鼠标放在图片上时，显示预览效果
            elif event.type() == QMouseEvent.Type.MouseMove:
                self.box_end_point = [int(event.pos().x()), int(event.pos().y())]
                temp_input_points = data.input_points.copy()
                temp_input_labels = data.input_labels.copy()
                temp_input_box = data.input_box.copy() if data.input_box is not None else None
                if event.modifiers()==Qt.KeyboardModifier.ControlModifier:
                    if mouse_event.buttons() == Qt.MouseButton.LeftButton and self.box_start_point != None:
                        pts = np.array([self.box_start_point[0], self.box_start_point[1], self.box_end_point[0], self.box_end_point[1]])
                        temp_input_box = self.mapCoords(pts, data2view=False)
                    else:
                        return super().eventFilter(obj, event)
                else:
                    temp_input_points = np.append(temp_input_points, coords, axis=0)
                    temp_input_labels = np.append(temp_input_labels, 1)
                self.accept_temp_masks=True
                self.background.predict(name=self.listWidget.item(self.idx_item).text(), 
                    input_points=temp_input_points, 
                    input_labels=temp_input_labels, 
                    input_box=temp_input_box, 
                    input_mask=data.logits,
                    temp_flag=True)


        # 监听键盘快捷键
        if event.type() == QEvent.Type.KeyPress:
            event:QKeyEvent = event
            # 方向键上和下切换图片
            if event.key() == Qt.Key.Key_Up:
                self.onBtnPrevious()
            elif event.key() == Qt.Key.Key_Down:
                self.onBtnNext()
            # Ctrl + Z 撤回
            elif event.key() == Qt.Key.Key_Z and event.modifiers()==Qt.KeyboardModifier.ControlModifier:
                self.onBtnWithdraw()
            # Ctrl + S 保存
            elif event.key() == Qt.Key.Key_S and event.modifiers()==Qt.KeyboardModifier.ControlModifier:
                self.onBtnSave()
            else:
                return super().eventFilter(obj, event)
            return True
        # 监听鼠标滚轮
        if event.type() == QEvent.Type.Wheel:
            if obj != self:
                return True
            event:QWheelEvent = event
            if event.angleDelta().y() > 0:
                self.onBtnPrevious()
            elif event.angleDelta().y() < 0:
                self.onBtnNext()
            return True

        return super().eventFilter(obj, event)

    def resetData(self):
        self.image=None
        self.image_scaled=None
        self.temp_masks=None
        self.accept_temp_masks=False
        self.ls_data = QStandardItemModel()
        self.cbbObject.setModel(self.ls_data)
        self.clearBackupData()
        self.is_data_modified = False
        self.idx_item = -1
        self.box_start_point = None
        self.box_end_point = None
    
    def  saveData(self):
        if self.countBackupData() <= 0:
            return;
        self.is_data_modified = False
        dir = self.dir_data(self.path_workspace)
        name = replacePostfix(self.listWidget.item(self.idx_item).text(), "pkl")
        path = os.path.join(dir, name)
        ls = initListByModel(self.ls_data)
        with open(path, "wb") as f:
            pickle.dump([ls, self.cbbObject.currentIndex()], f)
    def backupData(self):
        if self.idx_item < 0:
            return
        self.is_data_modified = True
        count = self.countBackupData()
        path = self.getBackupDataPath(count)
        ls = initListByModel(self.ls_data)
        with open(path, "wb") as f:
            pickle.dump([ls, self.cbbObject.currentIndex()], f)
    def restoreData(self):
        count = self.countBackupData()
        if count <= 0:
            return
        path = self.getBackupDataPath(count-1)
        with open(path,"rb") as f:
            ls_data, idx_data = pickle.load(f)
        initModelByList(self.ls_data,ls_data)
        self.cbbObject.setCurrentIndex(idx_data)
        os.remove(path)
        self.drawImages()
    def countBackupData(self):
        if self.idx_item == -1:
            return -1
        count = 0;
        while(True):
            path = self.getBackupDataPath(count)
            if os.path.exists(path):
                count += 1
            else:
                break;
        return count
    def getBackupDataPath(self, count):
        name = replacePostfix(self.listWidget.item(self.idx_item).text(), f".backup{count}.pkl", need_dot=False)
        path = self.to_cache_path(name)
        return path
    def clearBackupData(self):
        pattern = "^.*\.backup[0-9]+\.pkl$"
        removeMatchedFiles(self.dir_cache(), pattern)

    # 根据workspace文件夹内容，更新ListWidget
    def updateListWidget(self):
        ls_file_name = list(os.listdir(self.dir_images(self.path_workspace)))
        ls_file_name.sort()

        idx_origin = self.listWidget.currentRow()
        self.listWidget.setCurrentRow(-1)
        self.listWidget.clear()
        self.listWidget.addItems(ls_file_name)
        idx_origin = min(idx_origin, self.listWidget.count()-1)
        self.listWidget.setCurrentRow(idx_origin)
        
    def onItemSelectionChanged(self):
        if self.listWidget.currentRow() == -1:
            self.resetData()
            self.resetImageView()
        if self.idx_item == self.listWidget.currentRow():
            # 如果实际没有切换图片，则不需要更新GUI
            return
        if self.is_data_modified:
            # 提示尚未保存
            ans = InformationDialog.confirmInfo(self, title=self.tr("未保存"), ls_info=[self.tr("是否保存更改？")],\
                            txtBtnYes=self.tr("保存"), txtBtnNo=self.tr("放弃"))

            if ans == BasicDialog.Accepted:
                self.saveData()
            elif ans == BasicDialog.Rejected:
                pass
            else:
                # 未做选择，回到上一个Item
                self.listWidget.setCurrentRow(self.idx_item)
                return

        self.idx_item = self.listWidget.currentRow()
        item = self.listWidget.currentItem()
        name_current = item.text()

        # 读取图片
        path = os.path.join(self.dir_images(self.path_workspace), name_current)
        try:
            self.image = imRead(path, cv2.IMREAD_COLOR)
        except:
            self.image = None
            self.updateListWidget()
            return
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.background.createEmbeddings(self.image, name_current)

        self.is_data_modified=False
        self.clearBackupData()
        self.resetImageView()

        # 初始化数据，如果已有数据则读取
        path = os.path.join(self.dir_data(self.path_workspace), replacePostfix(name_current, "pkl"))

        if os.path.exists(path):
            with open(path, "rb") as f:
                ls_data, idx_data = pickle.load(f)
            initModelByList(self.ls_data, ls_data)
            self.cbbObject.setCurrentIndex(idx_data)
        else:
            initModelByList(self.ls_data, [SAMData()])
            self.cbbObject.setCurrentIndex(0)


    def resetImageView(self):
        # 清空ImageView(Label)
        self.lblImage.setPixmap(QPixmap())
        self.lblSegment.setPixmap(QPixmap())
        self.lblMixed.setPixmap(QPixmap())
        # 显示图片
        if np.array_equal(self.image,None):
            return
        
        # 设置ImageView的中间对齐和最小大小
        for lbl in [self.lblImage, self.lblSegment, self.lblMixed]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_scaled = imResizeKeepRatio(self.image, self.lblImage.parent().width(),  self.lblImage.parent().size().height())


    def drawTempMixedImage(self):
        if self.cbbObject.currentIndex() < 0:
            return
        if not self.accept_temp_masks or np.array_equal(self.temp_masks, None):
            return
        data = self.cbbObject.currentData()
        height, width = self.image_scaled.shape[:2]
        input_points = self.mapCoords(data.input_points, data2view=True)
        input_box = self.mapCoords(data.input_box, data2view=True)

        if not self.hovering:
            mixed = self.mixed_scaled.astype(np.uint8)
            drawPrompts(mixed,  input_points, data.input_labels, input_box)
            mixed = QImage(mixed, width, height,width*3,QImage.Format.Format_RGB888)
            self.lblMixed.setPixmap(QPixmap.fromImage(mixed))
            return

        if self.box_start_point != None and self.box_end_point != None:
            # 框的预览
            input_box = np.array([self.box_start_point[0], self.box_start_point[1], self.box_end_point[0], self.box_end_point[1]])
        temp_mask_scaled = cv2.resize(self.temp_masks[0].astype("uint8"), (width, height), interpolation=cv2.INTER_NEAREST)
        if hasattr(self, "mask_scaled") and not np.array_equal(self.mask_scaled,None):
            temp_mask_scaled = np.logical_and(temp_mask_scaled>0, self.mask_scaled==0).astype("uint8")

        dilated = cv2.dilate(temp_mask_scaled, np.ones((5,5),np.uint8), iterations=1)
        border = dilated - temp_mask_scaled
        
        temp_mask_scaled = np.stack([temp_mask_scaled, temp_mask_scaled, temp_mask_scaled], axis=-1)
        image_masked = self.image_scaled * temp_mask_scaled
        mixed = self.mixed_scaled.copy()
        mixed += image_masked * self.mix_factor
        mixed[border>0] = [0, 255, 0]
        mixed = mixed.astype(np.uint8)
        drawPrompts(mixed,  input_points, data.input_labels, input_box)
        mixed = QImage(mixed, width, height,width*3,QImage.Format.Format_RGB888)
        self.lblMixed.setPixmap(QPixmap.fromImage(mixed))

    def drawImages(self):
        if self.cbbObject.currentIndex() < 0:
            return
        self.accept_temp_masks=False
        data:SAMData = self.cbbObject.currentData()

        height, width = self.image_scaled.shape[:2]
        image = self.image_scaled.copy()
        input_points = self.mapCoords(data.input_points, data2view=True)
        input_box = self.mapCoords(data.input_box, data2view=True)


        drawPrompts(image, input_points, data.input_labels, input_box)
        image = QImage(image, width, height,width*3,QImage.Format.Format_RGB888)
        self.lblImage.setPixmap(QPixmap.fromImage(image))

        # create mixed image
        self.mix_factor = 0.65
        mixed = self.image_scaled.copy() * (1-self.mix_factor)
        

        mask = np.zeros((height, width), dtype=np.uint8)
        if not np.array_equal(data.masks, None):
            mask = cv2.resize(data.masks[0].astype("uint8"), (width, height), interpolation=cv2.INTER_NEAREST)
        dilated = cv2.dilate(mask, np.ones((5,5),np.uint8), iterations=1)
        self.mask_scaled = cv2.dilate(dilated, np.ones((5,5),np.uint8), iterations=1)
        border = dilated - mask

        
        mask = np.stack([mask, mask, mask], axis=-1)

        image_masked = self.image_scaled * mask
        
        
        mixed += image_masked * self.mix_factor
        mixed[border>0] = [0, 0, 255]

        self.mixed_scaled = mixed.copy()
        
        mixed = mixed.astype(np.uint8)
        drawPrompts(mixed,  input_points, data.input_labels, input_box)
        mixed = QImage(mixed, width, height,width*3,QImage.Format.Format_RGB888)
        self.lblMixed.setPixmap(QPixmap.fromImage(mixed))

        # create segment result
        
        
        ls_mask, ls_invert = [],[]
        for i in range(self.ls_data.rowCount()):
            d:SAMData = self.ls_data.item(i).data(role=Qt.ItemDataRole.UserRole)
            if np.array_equal(d.masks, None):
                continue
            ls_mask.append(d.masks[0])
            ls_invert.append(d.invertMask)
        if len(ls_mask) == 0:
            mask_final = np.zeros((height, width), dtype=np.uint8)
        else:
            mask_final = mergeMasks(ls_mask, ls_invert)
            mask_final = cv2.resize(mask_final.astype("uint8"), (width, height), interpolation=cv2.INTER_NEAREST)


        image_masked_final = self.image_scaled * np.stack([mask_final, mask_final, mask_final], axis=-1)
        segment = np.zeros((height, width, 4), dtype=np.uint8)
        segment[mask_final>0,3] = 255
        segment[:,:,0:3] = image_masked_final


  
        segment = QImage(segment, width, height,width*4,QImage.Format.Format_RGBA8888)
        self.lblSegment.setPixmap(QPixmap.fromImage(segment))
        
    def exportWorkspace(self, path_export):
        imageSize = self.getCkbInputSize(self.ckbExportImageSize, self.lblExportImageSize)
        segmentSize = self.getCkbInputSize(self.ckbExportSegmentSize, self.lblExportSegmentSize)
        maskSize = self.getCkbInputSize(self.ckbExportMaskSize,self.lblExportMaskSize)

        ls_name_finished = []
        for i in range(self.listWidget.count()):
            name = self.listWidget.item(i).text()
            path = os.path.join(self.dir_data(self.path_workspace), replacePostfix(name, "pkl"))
            if os.path.exists(path):
                ls_name_finished.append(name)
        ls_info = [
            self.tr("请确认导出信息："),
            self.tr("已处理图片数量: ") + str(len(ls_name_finished)),
            self.tr("导出路径：") + path_export,
            self.ckbExportExpand.text() + ": " + (self.tr("是") if self.ckbExportExpand.isChecked() else self.tr("否")),
            self.ckbExportImageSize.text() + self.lblExportImageSize.text(),
            self.ckbExportSegmentSize.text() + self.lblExportSegmentSize.text(),
            self.ckbExportMaskSize.text() + self.lblExportMaskSize.text(),
        ]
        ans = InformationDialog.confirmInfo(self, title=self.tr("导出设置"), ls_info=ls_info)
        if ans != BasicDialog.Accepted:
            return
        for name in ls_name_finished:  
            path_image = os.path.join(self.dir_images(self.path_workspace), name)
            
            # image
            image = imRead(path_image, cv2.IMREAD_COLOR)
            image_out = imResizeUndeformed(image, imageSize[0], imageSize[1]) if None not in imageSize else image
            path_out = os.path.join(self.dir_images(path_export), replacePostfix(name, "png"))
            imWrite(path_out, image_out)
            # mask
            path_data = os.path.join(self.dir_data(self.path_workspace), replacePostfix(name, "pkl"))
            with open(path_data, "rb") as f:
                ls_samdata, idx_data = pickle.load(f)
            ls_mask, ls_invert = [],[]
            for idx, samdata in enumerate(ls_samdata):
                ls_mask.append(samdata.masks[0])
                ls_invert.append(samdata.invertMask)
                # mask_instance
                mask = samdata.masks[0]
                mask_out = imResizeUndeformed(mask, maskSize[0], maskSize[1], cv2.INTER_NEAREST) if None not in maskSize else mask
                path_instance = os.path.join(self.dir_masks_instance(path_export), replacePostfix(name, f"_{idx}.png", False))
                imWrite(path_instance, mask_out*255)

            mask_final = mergeMasks(ls_mask, ls_invert).astype("uint8")
            mask_final_out =  imResizeUndeformed(mask_final, maskSize[0], maskSize[1], cv2.INTER_NEAREST) if None not in maskSize else mask_final
            path_out = os.path.join(self.dir_masks(path_export), replacePostfix(name, "png"))
            imWrite(path_out, mask_final_out*255)

            

            # segment
            if self.ckbExportExpand.isChecked():
                step = 5
                thresh = 1
                for top in range(0, mask_final.shape[0]):
                    flags = [(np.sum(mask_final[top+i])>=thresh) for i in range(step)]
                    if (np.array(flags) == True).all():
                        break
                for bottom in range(mask_final.shape[0]-1, -1, -1):
                    flags = [(np.sum(mask_final[bottom-i])>=thresh) for i in range(step)]
                    if (np.array(flags) == True).all():
                        break
                for left in range(0, mask_final.shape[1]):
                    flags = [(np.sum(mask_final[:,left+i])>=thresh) for i in range(step)]
                    if (np.array(flags) == True).all():
                        break
                for right in range(mask_final.shape[1]-1, -1, -1):
                    flags = [(np.sum(mask_final[:,right-i])>=thresh) for i in range(step)]
                    if (np.array(flags) == True).all():
                        break
                if top < bottom and left < right:
                    mask_valid = mask_final[top:bottom+1, left:right+1]
                    image_valid = image[top:bottom+1, left:right+1]
                else:
                    mask_valid = mask_final
                    image_valid = image
            else:
                mask_valid = mask_final
                image_valid = image
            if None not in segmentSize:
                mask_valid = imResizeUndeformed(mask_valid, segmentSize[0], segmentSize[1], cv2.INTER_NEAREST)
                image_valid = imResizeUndeformed(image_valid, segmentSize[0], segmentSize[1])

            
            image_masked = image_valid * np.stack([mask_valid, mask_valid, mask_valid], axis=-1)
            segment = image_masked
            # segment = np.zeros((image_masked.shape[0], image_masked.shape[1], 3), dtype=np.uint8)
            # segment[mask_valid>0,3] = 255
            # segment[:,:,0:3] = image_masked
            segment[mask_valid<=0,:] = 255

            path_out = os.path.join(self.dir_segments(path_export), replacePostfix(name, "png"))
            imWrite(path_out, segment)
        InformationDialog.showInfoWithButtons(self, title=self.tr("导出完成"), \
            ls_info=[self.tr(f"已将{len(ls_name_finished)}张图片的处理结果导出到{path_export}")], buttons=[self.tr("确定")])


    def mapCoords(self, coords:np.array, data2view:bool):
        if np.array_equal(coords, None):
            return None
        assert not np.array_equal(self.image_scaled, None) 
        assert not np.array_equal(self.image, None) 
        if data2view:
            scale_factor = self.image_scaled.shape[0] / self.image.shape[0]
        else:
            scale_factor = self.image.shape[0] / self.image_scaled.shape[0]

        return coords * scale_factor

    def resizeUI(self):
        screen = QGuiApplication.primaryScreen().geometry()  # 获取屏幕类并调用geometry()方法获取屏幕大小
        width = screen.width()  # 获取屏幕的宽
        height = screen.height()  # 获取屏幕的高
        # desktop = QApplication.desktop()

        minimumBtnSize = self.btnAddObject.minimumSize()
        minimumWndSize = self.minimumSize()
        if width >= 1920 and height >= 1080:
            minimumBtnSize = QSize(150,60)
            minimumWndSize = (1400,900)
        elif width >= 3000 and height >= 2160:
            minimumBtnSize = QSize(250, 80)
            minimumWndSize = QSize(2500, 2000)
        
        self.setMinimumSize(minimumWndSize)

        font = QFont()
        font.setFamily("Tahoma") 
        font.setPointSize(12)  
        def setFont(widget):
            if isinstance(widget, QWidget):
                widget.setFont(font)
            if isinstance(widget, QPushButton):
                widget.setMinimumSize(minimumBtnSize)
        traverseChildren(self.wgtOperation, setFont)
        font.setPointSize(10)
        self.listWidget.setFont(font)
        self.statusBar().setFont(font)


    
        pass
if __name__ == "__main__":
    app = QApplication(sys.argv)    # 创建QApplication对象，作为GUI主程序入口
    window = MainWindow()
    window.show()               # 显示主窗体
    sys.exit(app.exec())   # 循环中等待退出程序