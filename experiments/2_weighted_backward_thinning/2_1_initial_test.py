import argparse
from utils import build_joint_embedding, save_data_to_csv
from datasets import get_data
from subsampling_models import WeightedBackwardThinning
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Random Projection Tree")
    parser.add_argument('--problem_type', type=str, choices=['classification', 'regression'])
    parser.add_argument('--task_id', type=int)
    parser.add_argument('--path_to_save', type=str)
    parser.add_argument('--version', type=str, default='version_tree')
    parser.add_argument('--max_removals', type=int, default=2)
    args = parser.parse_args()

    X, y, dataset_name, task_id = get_data(args.problem_type, args.task_id)
    
    # Compute the joint embedding components
    phi_X, psi_y = build_joint_embedding(X, y, problem=args.problem_type)
    
    backward_thinning = WeightedBackwardThinning(feature_map_X=phi_X,
                                                 feature_map_y=psi_y,
                                                 depth=args.max_removals,
                                                 mmd_tolerance=0.7,
                                                 alpha=1.0
                                                 )  
    backward_thinning.fit(X, y)
    coreset, weights = backward_thinning.transform(X, y)
    print(f"Selected coreset of size {len(coreset)}/{len(X)} with weights: {min(weights):.4f} to {max(weights):.4f}")



if __name__ == "__main__":
    main()
