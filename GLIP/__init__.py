import sys
import os
from GLIP.maskrcnn_benchmark.config import cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from GLIP.maskrcnn_benchmark.data.datasets.laion import Laion

sys.path.append(os.path.join(os.getcwd(), "GLIP"))
