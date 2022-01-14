import torch
from data import load_lost_and_found_with_distance
from models import LadderDenseNet
from evaluations import OODDistEvaluation
from prettytable import PrettyTable

DATA_DIR = './Lost_and_found'
def create_output(results):
    x = PrettyTable()
    x.add_column("Range (m)", ["AP", "FPR at TPR 95%", "AUROC"])
    for res in results:
        stats, range = res[:-1], res[-1]
        x.add_column(f"{range[0]}-{range[1]}", [str(round(s * 100., 2)) for s in stats])
    return x

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = load_lost_and_found_with_distance(DATA_DIR)

    # initialize the model
    state = './params/model.pth'
    model = LadderDenseNet(num_classes=19, checkpointing=False).to(device)
    model.load_state_dict(torch.load(state, map_location=device), strict=True)
    model.eval()
    print("> Loaded model.")


    # define the anomaly detector
    # higher values for anomalous pixels
    def compute_ood_probs(img):
        img = img.float()/255.
        with torch.no_grad():
            logit = model(img.to(device))
        prob = torch.softmax(logit, dim=1).max(1)[0]
        return 1 - prob

    evaluator = OODDistEvaluation(loader, compute_ood_probs)
    print('> Calculating metrics might take a while..')
    results = evaluator.calculate_ood_scores()
    print(create_output(results))





if __name__ == '__main__':
    main()