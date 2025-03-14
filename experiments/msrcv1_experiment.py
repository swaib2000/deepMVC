from models.DeepMVC import DeepMVC
from utils.data_preprocessing import load_msrc_data
from utils.evaluation import evaluate

def run_msrc_experiment():
    data = load_msrc_data()
    model = DeepMVC()
    embeddings = model.fit(data)
    evaluate(embeddings)