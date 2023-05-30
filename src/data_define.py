import numpy as np

    

class SAMData:
    def __init__(self):
        self.input_points:np.array = np.ndarray((0,2))
        self.input_labels:np.array = np.ndarray(0)
        self.input_box:np.arrry = np.array(None)
        self.masks:np.array=np.array(None)
        self.scores:np.array=np.array(None)
        self.logits:np.array=np.array(None)
        self.invertMask=False

#     def copy(self):
#         new_data = SAMData()
#         new_data.input_points = self.input_points.copy() if self.input_points is not None else None
#         new_data.input_labels = self.input_labels.copy() if self.input_labels is not None else None
#         new_data.input_box = self.input_box.copy() if self.input_box is not None else None
#         new_data.masks = self.masks.copy() if self.masks is not None else None
#         new_data.scores = self.scores.copy() if self.scores is not None else None
#         new_data.logits = self.logits.copy() if self.logits is not None else None

#         return new_data

    # def save(self, path):
    #     np.savez(path, self.input_points, self.input_labels, self.input_box, self.masks, self.scores, self.logits)
    # def load(path):
    #     new_obj = SAMData()
    #     data = np.load(path, allow_pickle=True)
    #     new_obj.input_points = data["arr_0"]
    #     new_obj.input_labels = data["arr_1"]
    #     new_obj.input_box = data["arr_2"]
    #     new_obj.masks = data["arr_3"]
    #     new_obj.scores = data["arr_4"]
    #     new_obj.logits = data["arr_5"]
    #     return new_obj

    # def toDict(self):
    #     return {
    #         "input_points":self.input_points.tolist(),
    #         "input_labels":self.input_labels.tolist(),
    #         "input_box":self.input_box.tolist(),
    #         "masks":self.masks.tolist() if self.masks is not None else None,
    #         "scores":self.scores.tolist() if self.scores is not None else None,
    #         "logits":self.logits.tolist() if self.logits is not None else None,
    #     }
    # def fromDict(d):
    #     new_data = SAMData()
    #     new_data.input_points = np.array(d["input_points"])
    #     new_data.input_labels = np.array(d["input_labels"])
    #     new_data.input_box = np.array(d["input_box"])
    #     new_data.masks = np.array(d["masks"]) if d["masks"] is not None else None
    #     new_data.scores = np.array(d["scores"]) if d["scores"] is not None else None
    #     new_data.logits = np.array(d["logits"]) if d["logits"] is not None else None
    #     return new_data


    # def dataEqual(self, another):
    #     if another==None:
    #         return False
    #     return np.array_equal(self.input_points, another.input_points) and \
    #         np.array_equal(self.input_labels, another.input_labels) and \
    #         np.array_equal(self.input_box, another.input_box) and \
    #         np.array_equal(self.masks, another.masks) and \
    #         np.array_equal(self.scores, another.scores) and \
    #         np.array_equal(self.logits, another.logits)

     

        