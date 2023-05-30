from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import torch
import numpy as np
import time

from mainwindow import MainWindow
from segment_anything import sam_model_registry, SamPredictor

class BackgroundThread(QThread):
    def __init__(self, parent):
        super().__init__()
        self.setParent(parent)
        self.queue_task = []
        self.mtx_queue = QMutex()
        self.semaphore = QSemaphore()
        self.alive = True
        self.curr_task = None

    def run(self):
        while(self.alive):
            self.semaphore.acquire(1)
            while(True):
                with QMutexLocker(self.mtx_queue):
                    if len(self.queue_task)>0:
                        self.curr_task = self.queue_task[0]
                        self.queue_task.pop(0)
                if self.curr_task == None:
                    break;
                fn,  args = self.curr_task[0], self.curr_task[2:]
                fn(*args)
                self.curr_task = None

class Background(QObject):
    sig_func = pyqtSignal(list)
    sig_loading_finished = pyqtSignal()
    sig_embeddings_finished = pyqtSignal(str)
    sig_predict_finished = pyqtSignal(str,np.ndarray, np.ndarray, np.ndarray, bool)
    def __init__(self, mainwindow):
        super().__init__()
        self.mainwindow:MainWindow = mainwindow

        self.thread = BackgroundThread(mainwindow)
        self.thread.start()

        self.is_sam_loaded = False
        self.embedded_image = None
        self.mtx_embeddings = QMutex()

    def shutdown(self):
        self.thread.alive = False
        self.thread.semaphore.release(1)
        self.thread.wait()

    def enqueue(self, fn, cancelable=False, *args):
        with QMutexLocker(self.thread.mtx_queue):
            self.thread.queue_task.append([fn, cancelable, *args])
        self.thread.semaphore.release(1)

    def cancelTasks(self, fn, cancel_current=False):
        with QMutexLocker(self.thread.mtx_queue):
            new_queue = []
            for task in self.thread.queue_task:
                cancelable = task[1]
                if fn != task[0] or not cancelable:
                    new_queue.append(task)
            self.thread.queue_task = new_queue
        assert cancel_current == False, "cancel current task may result in crash!"
        if cancel_current and (self.thread.curr_task != None):
            if self.thread.curr_task[0] == fn:
                self.thread.terminate()
                self.thread.wait()
                self.thread = BackgroundThread(self.mainwindow)
                self.thread.start()
                
    def loadSAM(self, model_type, path):
        self.enqueue(self.__loadSAM__, False, model_type, path)
    def __loadSAM__(self, model_type, path):
        start = time.time()
        self.sig_func.emit([self.setStatusBar,"Loading SAM......"])
        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.is_sam_loaded = True
        end = time.time()
        self.sig_func.emit([self.setStatusBar,"SAM loaded, "\
                    "took {:.3f} seconds.".format(end-start)])
        self.sig_loading_finished.emit()

    def createEmbeddings(self, image:np.ndarray, name:str):
        for fn in [self.__createEmbeddings__, self.__predict__]:
            self.cancelTasks(fn, cancel_current=False)
        self.enqueue(self.__createEmbeddings__, True, image.copy(),name)
    def __createEmbeddings__(self,image, name:str):
        with QMutexLocker(self.mtx_embeddings):
            start = time.time()
            self.sig_func.emit([self.setStatusBar,"Compute embeddings......"])
            self.predictor.set_image(image)
            self.embedded_image = name
            end = time.time()
            self.sig_func.emit([self.setStatusBar,"Computing embeddings finished, "\
                        "took {:.3f} seconds.".format(end-start)])
            self.sig_embeddings_finished.emit(name)
        

    def predict(self, name, input_points, input_labels, input_box, input_mask, temp_flag=False):
        self.cancelTasks(self.__predict__, cancel_current=False)
        self.enqueue(self.__predict__,  temp_flag, name, input_points, input_labels, input_box, input_mask, temp_flag)
    def predictImmediate(self, name, input_labels, input_box, input_mask, temp_flag=False):
        assert False, "predictImmediate is deprecated, use predict instead"
        res = (None, None, None, None, None)
        if(self.mtx_embeddings.tryLock()):
            if self.embedded_image != None and self.embedded_image == name:
               res = self.__predict__(name, input_points, 
                    input_labels, input_box, input_mask, temp_flag)
            self.mtx_embeddings.unlock()
        return res
    def __predict__(self, name, input_points, input_labels, input_box, input_mask, temp_flag):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points.copy() if input_points.shape[0] != 0 else None,
            point_labels=input_labels.copy() if input_labels.shape[0] != 0 else None,
            box=input_box.copy() if (not np.array_equal(input_box, None)) else None,
            mask_input=input_mask.copy() if not np.array_equal(input_mask, None) else None,
            multimask_output=False,
            return_logits=False
        )

        self.sig_predict_finished.emit(name, masks, scores, logits, temp_flag)
        return name, masks, scores, logits, temp_flag

    def setStatusBar(self, s):
        self.mainwindow.statusBar().showMessage(s)
    


# class SAMData:
#     def __init__(self):
#         self.input_points:np.array = np.ndarray((0,2))
#         self.input_labels:np.array = np.ndarray(0)
#         self.input_box:np.arrry = np.array(None)
#         self.masks:np.array=np.array(None)
#         self.scores:np.array=np.array(None)
#         self.logits:np.array=np.array(None)