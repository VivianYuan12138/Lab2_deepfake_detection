"""
Evaluation script for ASVspoof detection model.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data_utils import Dataset_ASVspoof2019_devNeval, genSpoof_list
from evaluation import calculate_tDCF_EER
from model import Model

def main(args: argparse.Namespace) -> None:
    # Load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    device = "cpu"
    # Define model
    model = Model(config["model_config"]).to(device)

    # Load pretrained model
    if args.eval_model_weights:
        model.load_state_dict(torch.load(args.eval_model_weights, map_location=device))
    else:
        print("Model weights path must be specified for evaluation.")
        sys.exit(1)
    
    print("Model loaded.")

    # Define evaluation data loader
    eval_loader = get_loader(config)

    # Evaluation
    print("Start evaluation...")
    eval_score_path = Path(args.output_dir) / config["eval_output"]
    eval_trial_path = Path(config["database_path"]) / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    produce_evaluation_file(eval_loader, model, device, eval_score_path, eval_trial_path)
    
    calculate_tDCF_EER(cm_scores_file=eval_score_path,
                       asv_score_file=Path(config["database_path"]) / config["asv_score_path"],
                       output_file=Path(args.output_dir) / "t-DCF_EER.txt")
    print("Evaluation done.")


def get_loader(config: dict) -> DataLoader:
    """
    Return DataLoader for evaluation.
    """
    database_path = Path(config["database_path"])
    eval_trial_path = database_path / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    
    file_eval = genSpoof_list(dir_meta=str(eval_trial_path), is_train=False, is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval, base_dir=str(database_path) + "/ASVspoof2019_LA_eval/",limit= config["eval_limit"])
    
    eval_loader = DataLoader(eval_set, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    return eval_loader


def produce_evaluation_file(data_loader: DataLoader, model, device: torch.device, save_path: Path, trial_path: Path) -> None:
    """
    Perform evaluation and save the scores to a file.
    """
    if not save_path.parent.exists():
        os.makedirs(save_path.parent)
    with open(trial_path, "r") as fh:
        trial_list = fh.readlines()
    model.eval()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _ , batch_out = model(batch_x)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_list):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write(f"{utt_id} {src} {key} {sco}\n")
    
    fh.close()
    print(f"Scores saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof evaluation script")
    parser.add_argument("--config", type=str,default="./RawNet2.conf", help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_result", help="Output directory for results")
    parser.add_argument("--eval_model_weights", type=str, help="Path to model weights for evaluation",default="./rawnet2_model.pth")
    args = parser.parse_args()
    main(args)
