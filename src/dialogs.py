from enum import Enum
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



# def askYesOrNo(parent, title, text, yesText, noText):
#     messageBox = QMessageBox(parent)
#     messageBox.setModal(True)
#     messageBox.setWindowTitle(title)
#     messageBox.setText(text)
#     btnYes = messageBox.addButton(yesText, QMessageBox.ButtonRole.YesRole)
#     btnNo = messageBox.addButton(noText, QMessageBox.ButtonRole.NoRole)

#     messageBox.exec()
#     print(messageBox.accepted(), messageBox.rejected())
#     if messageBox.clickedButton() == btnYes:
#         return UserAnswer.Yes
#     elif messageBox.clickedButton() == btnNo:
#         return UserAnswer.No
#     else:
#         return UserAnswer.Unknown

class BasicDialog(QDialog):
    Accepted = 1
    Rejected = 2
    Unknown = 3
    
    def __init__(self, parent, title, buttons):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle(title)
        
        self.setMinimumWidth(400)
        font = QFont()
        font.setFamily("Tahoma") 
        font.setPointSize(10)  
        self.setFont(font)
        self.clicked_button = None
        

        h_layout = QHBoxLayout()
        h_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
        for name in buttons:
            btn = QPushButton(name)
            btn.clicked.connect(self.onBtn)
            h_layout.addWidget(btn)
            h_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
            
            
        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.setContentsMargins(20,20,20,20)
        self.setLayout(v_layout)

    def addLayout(self, layout):
        self.layout().insertSpacerItem(self.layout().count()-1, QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.layout().insertLayout(self.layout().count()-1, layout)
        self.layout().insertSpacerItem(self.layout().count()-1, QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))

    def onBtn(self):
        self.clicked_button = self.sender().text()
        self.accept()

    
class InputDialog(BasicDialog):
    def __init__(self, parent, title, buttons, ls_attrs):
        super().__init__(parent,title, buttons)
        
        f_layout = QFormLayout()
        self.dict_inputs = {}
        for attr in ls_attrs:
            self.dict_inputs[attr] = QLineEdit()
            f_layout.addRow(QLabel(attr), self.dict_inputs[attr])

        self.addLayout(f_layout)

    def getInputText(self, attr):
        return self.dict_inputs[attr].text()

    def getInputWidthHeight(parent):
        buttons = [parent.tr("确定"), parent.tr("取消")]
        ls_attr = [parent.tr("宽度"), parent.tr("高度")]
        dialog = InputDialog(parent, parent.tr("输入宽高"), buttons, ls_attr)
        dialog.exec()
        if dialog.clicked_button == parent.tr("确定"):
            try:
                width = int(dialog.getInputText(parent.tr("宽度")))
                height = int(dialog.getInputText(parent.tr("高度")))
                if width <= 0 or height <= 0:
                    return None, None
                return width, height
            except Exception:
                return None, None
        else:
            return None, None

class InformationDialog(BasicDialog):
    def __init__(self, parent, title,buttons, ls_info):
        super().__init__(parent, title, buttons)

        v_layout = QVBoxLayout()
        for info in ls_info:
            v_layout.addWidget(QLabel(info))
        self.addLayout(v_layout)
    
    def confirmInfo(parent, title, ls_info, txtBtnYes=None, txtBtnNo=None):
        buttons = [parent.tr("确定"), parent.tr("取消")]
        if txtBtnYes is not None:
            buttons[0] = txtBtnYes
        if txtBtnNo is not None:
            buttons[1] = txtBtnNo
        
        clicked_button = InformationDialog.showInfoWithButtons(parent, title, buttons, ls_info)
        if clicked_button == buttons[0]:
            return InformationDialog.Accepted
        elif clicked_button == buttons[1]:
            return InformationDialog.Rejected
        else:
            return InformationDialog.Unknown

    def showInfoWithButtons(parent, title, buttons, ls_info):
        dialog = InformationDialog(parent, title, buttons, ls_info)
        dialog.exec()
        return dialog.clicked_button
    