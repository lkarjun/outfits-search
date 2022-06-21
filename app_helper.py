from fastai.vision.all import *
from pathlib import Path
import pathlib
from jinja2 import Environment, BaseLoader

template = Environment(loader=BaseLoader)
df = pd.read_csv("Dataset/data(Updated).csv")


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
    return idx



def result_html(idx):
  html = """
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-0evHe/X+R7YkIZDRvuzKMRqM+OrBnVFBL6DOitfPri4tjfHxaWutUpFmBp4vmVor" crossorigin="anonymous">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/js/bootstrap.bundle.min.js" integrity="sha384-pprn3073KE6tl6bjs2QrFaJGz5/SUsLqktiwsUTF55Jfv3qYSDhgCecCxMW52nD2" crossorigin="anonymous"></script>

        <div class="row p-3 text-center">
        {% for i in df.iterrows() %}

          <div class="col-sm-4 pb-5">
            <div class="card">
              <img class="card-img-top" src="{{ i[1]['product_img_url'] }}">
              <div class="card-body">
                <h6 class="card-title text-capitalize">{{ i[1]['product_name'].lower() }}</h6>
                <p class="card-text text-capitalize">{{ i[1]['product_category'] }} <b>

                    {% if i[1]['product_price'] == 'None' %}
                      Not Available
                    {% else %}
                      {{ i[1]['product_price'] }}
                    {% endif %}
                    </b>
                
                </p>
                <a href="https://www.myntra.com/{{ i[1]['product_source'] }}" class="btn btn-sm" style="background-color: #5FD068;" target="_blank"><b>Buy Now</b></a>
              </div>
            </div>
          </div>

        {% endfor %}
        </div>

        <hr>
  """

  rslt = df.iloc[idx[0]]

  return template.from_string(html).render(df=rslt)