import os 
from import_utils import *

class  SegmentationMap:
    """
        Initiate a SegmentationMap object from an image ID of the BSDS300
    """
    def __init__(self,iid):
        self.iid = iid #image
        self.jpg_path = os.path.abspath(os.path.join(JPG_PATH,iid,".jpg"))
        self.seg_path = os.path.abspath(os.path.join(SEG_PATH,iid,".seg"))
        self.img = import_jpg(self.jpg_path)