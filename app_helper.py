from turtle import distance
from fastai.vision.all import *
from pathlib import Path
import pathlib

if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

Model_path = Path("Model/model.pkl")
Neighbor_path = Path("Model/neighbor.pkl")

if not Model_path.exists():
    os.system("dvc pull Model/model.pkl.dvc")
    os.system("dvc pull Model/neighbor.pkl.dvc")
    

fnames = pickle.load(open("Model/fnames.pkl", "rb"))
neighbor = pickle.load(open("Model/neighbor.pkl", "rb"))

class RImgModel(Module):
    def __init__(self, body,head):
        self.body = body
        self.head = head
    
    def forward(self, x):
        x = self.body(x)
        return self.head(x)


def model_splitter(model):
    return L(model.body, model.head).map(params)

def get_feature_vector(learn, X = None, dl = None, batch = False):
  with hook_output(learn.model.head[16]) as h:
    if batch:
      y = learn.get_preds(dl=dl)
    else:
      y = learn.predict(X)
  return h.stored


def get_result(x):
    learn = load_learner(Model_path)
    feature_vector = get_feature_vector(learn, tensor(x)).numpy()
    distances, idx = neighbor.kneighbors(feature_vector)
    s_images = [Image.open(fnames[i]) for i in idx.flatten()]
    return s_images