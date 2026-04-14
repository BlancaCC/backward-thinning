import os
import argparse
from datasets import get_data
from utils import build_joint_embedding, save_data_to_csv
from itertools import combinations
import numpy as np




def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Multiple Removal")
    parser.add_argument('--problem_type', type=str, choices=['classification', 'regression'], help='Type of problem: classification or regression')
    parser.add_argument('--task_id', type=int, help='Task ID for the experiment')
    parser.add_argument('--path_to_save', type=str, help='Path to save the results')
    parser.add_argument('--version', type=str, help='Version of the experiment', default='version_1')
    parser.add_argument('--max_removals', type=int, help='Maximum number of points to remove in combinations', default=2)
    args = parser.parse_args()

    # Load the dataset
    X,y,dataset_name, task_id = get_data(args.problem_type, args.task_id)
    max_l = 20

    X = X[:max_l]
    y = y[:max_l]

    # Compute the joint embedding
    phi_X, psi_y = build_joint_embedding(X, y, problem=args.problem_type)

    lifted_X = phi_X(X)
    lifted_y = psi_y(y)

    mu_p = (lifted_X.T @ lifted_y) / lifted_X.shape[0]

    for m in range(1, args.max_removals + 1):
        combinations_list = list(combinations(range(lifted_X.shape[0]), m))
        random_indices = range(len(combinations_list))#np.random.choice(len(combinations_list), size=min(100, len(combinations_list)), replace=False)
        for indices in [combinations_list[i] for i in random_indices]:
            error = np.linalg.norm(m* mu_p - (lifted_X[indices,:].T @ lifted_y[indices,:]))
            #print(f"Removed indices: {indices}, MMD Error: {error}")
            # Save the results
            save_path = os.path.join(args.path_to_save, args.version)
            save_data_to_csv(
                save_path,
                f"{args.problem_type[:3]}_{args.task_id}.csv",
                {   
                    "problem_type": args.problem_type,
                    "error": error,
                    "task_id": task_id,
                    "m": m,
                    "dataset": dataset_name,
                    "removed_indices": indices,
                }
            )

if __name__ == "__main__":
    main()
