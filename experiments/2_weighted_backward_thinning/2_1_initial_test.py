import argparse

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils import build_joint_embedding, save_data_to_csv, compute_gamma_scale
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Compute the joint embedding components
    phi_X, psi_y = build_joint_embedding(X_train, y_train, problem=args.problem_type, n_components=30)
    
    backward_thinning = WeightedBackwardThinning(feature_map_X=phi_X,
                                                 feature_map_y=psi_y,
                                                 depth=args.max_removals,
                                                 mmd_tolerance=0.7,
                                                 alpha=1.0
                                                 )  
    backward_thinning.fit(X_train, y_train)
    coreset, weights, index_to_keep = backward_thinning.transform(X_train, y_train)
    print(f"Selected coreset of size {len(coreset)}/{len(X_train)} with weights: {min(weights):.4f} to {max(weights):.4f}")

    # Train a KSVM on the coreset and evaluate on the full dataset
    gamma = compute_gamma_scale(coreset)
    clf = SVC(kernel='rbf', gamma=gamma)
    clf.fit(coreset, y_train[index_to_keep])
    acc = clf.score(X_test, y_test )
    print(f"Accuracy on full dataset using coreset: {acc:.4f}")

    # clf.fit(coreset, y_train[index_to_keep], sample_weight=weights)
    # acc_weighted = clf.score(X_test, y_test)
    
    #print(f"Accuracy on full dataset using weighted coreset: {acc_weighted:.4f}")
    # full
    clf_full = SVC(kernel='rbf', gamma=gamma)
    clf_full.fit(X_train, y_train)
    acc_full = clf_full.score(X_test, y_test)
    print(f"Accuracy on full dataset using all data: {acc_full:.4f}")

    random_indices= np.random.choice(len(X_train), size=len(coreset), replace=False)
    clf_random = SVC(kernel='rbf', gamma=gamma)
    clf_random.fit(X_train[random_indices], y_train[random_indices])
    acc_random = clf_random.score(X_test, y_test)

    print(f"Accuracy on full dataset using random subset: {acc_random:.4f}")
    # nystrom approximation
    from sklearn.kernel_approximation import Nystroem
    nystroem = Nystroem(kernel='rbf', gamma=gamma, n_components=len(coreset), random_state=42)
    X_nystroem = nystroem.fit_transform(X_train)
    clf_nystroem = SVC(kernel='linear')
    clf_nystroem.fit(X_nystroem[random_indices], y_train[random_indices])
    X_test_nystroem = nystroem.transform(X_test)
    acc_nystroem = clf_nystroem.score(X_test_nystroem, y_test)

    print(f"Accuracy on full dataset using Nystroem random subset: {acc_nystroem:.4f}")
    # Save results
    results = {
        'dataset': dataset_name,
        'task_id': task_id,
        'coreset_size': len(coreset),
        'accuracy_coreset': acc,
        'accuracy_weighted_coreset': acc_weighted,
        'accuracy_full': acc_full,
        'accuracy_random': acc_random,
        'accuracy_nystroem': acc_nystroem

    }
    os.makedirs(args.path_to_save, exist_ok=True)
    save_path = os.path.join(args.path_to_save, f"{args.version}_results.csv")
    save_data_to_csv(directory_path=save_path, file_name=f"{args.version}_results.csv", data=results)
    print(f"Results saved to {save_path}")




if __name__ == "__main__":
    main()
