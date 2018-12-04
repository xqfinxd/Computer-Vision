import argparse
import os
import cv2
import numpy as np
import aligment
import blend
import warp
import config

class PanoramaModule:
	def __init__(self, conf):
		self.images = []
		self.conf = conf

	def loadImages(self):
		dirpath = self.conf["path"]
		if not dirpath:
			return
		files = sorted(os.listdir(diroath))
		files = [
			f for in files
			if f.endswith('.jpg') or f.endswith('png')
		]
		self.images = [cv2.imread(os.path.join(dirpath, i)) for i in files]
		print('loaded {0} images from {1}'.format(len(self.images), dirpath))

	def compute(self):





