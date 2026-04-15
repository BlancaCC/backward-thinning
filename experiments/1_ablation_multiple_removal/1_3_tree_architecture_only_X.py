import os
import argparse
import numpy as np
from itertools import combinations
from datasets import get_data
from utils import build_joint_embedding, save_data_to_csv
from collections import defaultdict



class Signatures:
    """
    Generates a bitstring signature for each sample based on T random projections.
    """
    def __init__(self, features, T):
        self.features = features
        n_samples = features.shape[0]
        index = np.random.randint(0, n_samples, size=T)
        self.projections = features[index, :].T  # Shape: (d, T)
        # crealo recursivamente 
        self.dict_index = defaultdict(list)


    def get_signatures(self, features):
        # Compute dot products and get signs (True for +, False for -)
        bool_matrix = (features @ self.projections) > 0

        for i, row in enumerate(bool_matrix):
            key = tuple(row)  
            self.dict_index[key].append(i)

        # Convert rows to tuples to use as dictionary keys
        return self.dict_index

def get_opposite_signature(sig):
    """Returns the bitwise NOT of a signature tuple."""
    return tuple(not b for b in sig)

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Random Projection Tree")
    parser.add_argument('--problem_type', type=str, choices=['classification', 'regression'])
    parser.add_argument('--task_id', type=int)
    parser.add_argument('--path_to_save', type=str)
    parser.add_argument('--version', type=str, default='version_tree')
    parser.add_argument('--max_removals', type=int, default=2)
    parser.add_argument('--tree_depth', type=int, default=8, help='Number of random projections (T)')
    args = parser.parse_args()

    X, y, dataset_name, task_id = get_data(args.problem_type, args.task_id)
    
    # Compute the joint embedding components
    phi_X, psi_y = build_joint_embedding(X, y, problem=args.problem_type)
    lifted_X = phi_X(X)
    #lifted_y = psi_y(y)
    n_samples = lifted_X.shape[0]

    # 1. Compute mu_p (The mean operator)
    #mu_p = np.tensordot(lifted_X.mean(axis=0), lifted_y.mean(axis=0), axes=0)
    mu_px = lifted_X.mean(axis=0)
    #mu_py = lifted_y.mean(axis=0)

    # 2. Construct the Joint Features Phi(x,y) = phi(x) \otimes psi(y)
    # We flatten them to treat them as rows in a matrix for the projections
    print("Constructing joint features...")

    V = mu_px[np.newaxis, :] - lifted_X
    sig = Signatures(V, args.tree_depth)
    dict_index = sig.get_signatures(V)

    combinations_to_evaluate = [(i,) for i in range(n_samples)]
    for dict_key, indices in dict_index.items():
        sig_opposite = get_opposite_signature(dict_key)
        combinations_to_evaluate.extend([(i,j) for i in indices for j in dict_index.get(sig_opposite, []) if i < j])

    
    # Process the selected heuristic combinations
    print(f"Found {len(combinations_to_evaluate)} opposite pairs vs {n_samples * (n_samples - 1) // 2}. Evaluating errors...")
   
    for indices in combinations_to_evaluate:
        # We use the same joint_features we already computed
        subset_sum = np.sum(V[list(indices)], axis=0)
        # Error = || m * mu_p - sum(Phi_selected) ||
        error = np.linalg.norm(len(indices) * mu_px - subset_sum)
   
        # Save the results
        save_path = os.path.join(args.path_to_save, args.version)
        save_data_to_csv(
            save_path,
            f"{args.problem_type[:3]}_{args.task_id}.csv",
            {   
                "problem_type": args.problem_type,
                "error": error,
                "task_id": task_id,
                "m": len(indices),
                "dataset": dataset_name,
                "removed_indices": indices,
                "norm_phi_index": str({'f{i}': np.linalg.norm(V[i]) for i in indices})
            }
        )
        
    print(f"Done. Processed {len(combinations_to_evaluate)} heuristic pairs.")

if __name__ == "__main__":
    main()