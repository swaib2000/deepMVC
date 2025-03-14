from models.DeepMVC import DeepMVC
from utils.data_preprocessing import load_caltech101_data
from utils.evaluation import evaluate

def run_caltech_experiment():
    data = load_caltech101_data()
    model = DeepMVC()
    embeddings = model.fit(data)
    evaluate(embeddings)