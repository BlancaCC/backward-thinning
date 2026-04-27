"""
kernel herding
backward thinning
kernel thinning x 10 

"""


import argparse
from time import perf_counter
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.kernel_approximation import Nystroem
from subsampling_models import FastBackwardThinning, KernelHerding, KernelThinning, BackwardThinning, KernelThinningRFF, make_gaussian_kernel, FlexibleKernelThinning

"""
from subsampling_models.KernelThinningRFF import make_gaussian_kernel, KernelThinningRFF
from subsampling_models.BackwardThinning_rm import BackwardThinning
from subsampling_models.FastBackwardThinning import FastBackwardThinning
from subsampling_models.KernelThinning import KernelThinning
"""
# Asumo que importas la nueva clase desde donde la hayas guardado
# from subsampling_models import IterativeWeightedBackwardThinning 
from utils import build_joint_embedding, save_data_to_csv, compute_gamma_scale
from datasets import get_data

def main():
    parser = argparse.ArgumentParser(description="Prueba de Coreset Iterativo")
    parser.add_argument('--problem_type', type=str, choices=['classification', 'regression'], default='classification')
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--path_to_save', type=str, default='./results')
    parser.add_argument('--version', type=str, default='iterative_v1')
    # Nuevo argumento para el tamaño del coreset
    parser.add_argument('--target_size', type=int, default=50, help="Número final de puntos deseados")
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha para la redistribución de pesos")
  
    args = parser.parse_args()

    # Selección de modelo según tipo de problema
    if args.problem_type == 'classification':
        ml_model = SVC
        model_params = {
            'kernel': 'rbf'
        }
    else:
        # Heurística para epsilon: 0.1 * std de y_train (se ajusta más abajo tras split)
        ml_model = SVR
        model_params = {
            'kernel': 'rbf'
        }
    metric = 'accuracy' if args.problem_type == 'classification' else 'r2'
    # 1. Carga de datos
    X, y, dataset_name, task_id = get_data(args.problem_type, args.task_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Ajuste de epsilon para regresión basado en los datos
    if args.problem_type == 'regression':
        epsilon = 0.1 * np.std(y_train)
        model_params['epsilon'] = epsilon
    
    # 2. Construcción del embedding (mapeo de características)
    # Nota: phi_X suele ser una función o un objeto que proyecta los datos
    phi_X, psi_y = build_joint_embedding(X_train, y_train, problem=args.problem_type, n_components=100)
    
    #psi_y = None
    # 3. Inicialización y ejecución del algoritmo iterativo
    # Pasamos phi_X como el feature_map que el algoritmo usará internamente
    
    model = BackwardThinning(
        feature_map_X=phi_X,
        feature_map_y=psi_y
    )
    print(f"Calculando coreset iterativo para {dataset_name}...")
    num_to_remove = len(X_train) - int(args.target_size*len(X_train) // 100)  # Convertimos porcentaje a número absoluto
    time_start = perf_counter() 
    coreset_X,coreset_y = model.thin(num_to_remove, X=X_train, Y=y_train)  # Convertimos porcentaje a número absoluto
    time_coreset_backward = perf_counter()  - time_start
    print(f"Coreset iterativo calculado en {time_coreset_backward:.2f} segundos.")

    # Obtenemos los índices originales para recuperar las etiquetas y correspondencias
     # 4. Evaluación: SVC con Coreset (Sin Pesos)
    gamma = compute_gamma_scale(coreset_X)
    clf = ml_model(gamma=gamma, **model_params)
    clf.fit(coreset_X, coreset_y.ravel())  # Aseguramos que coreset_y sea un vector 1D
    acc_backward_thinning = clf.score(X_test, y_test.ravel())
    print(f"{metric} backward thinning: {acc_backward_thinning:.4f} en {time_coreset_backward:.2f} segundos.")


    model = FastBackwardThinning(
        feature_map_X=phi_X,
        feature_map_y=psi_y
    )
    print(f"Calculando coreset iterativo para {dataset_name}...")
    num_to_remove = len(X_train) - int(args.target_size*len(X_train) // 100)  # Convertimos porcentaje a número absoluto
    time_start = perf_counter() 
    coreset_X,coreset_y = model.thin(num_to_remove, X=X_train, Y=y_train)  # Convertimos porcentaje a número absoluto
    time_coreset_fast_backward = perf_counter()  - time_start

    # Obtenemos los índices originales para recuperar las etiquetas y correspondencias

    # 4. Evaluación: SVC con Coreset (Sin Pesos)
    gamma = compute_gamma_scale(coreset_X)
    clf = ml_model(gamma=gamma, **model_params)
    clf.fit(coreset_X, coreset_y.ravel())  # Aseguramos que coreset_y sea un vector 1D
    acc_fast_backward = clf.score(X_test, y_test.ravel())
    print(f"{metric} fast backward thinning: {acc_fast_backward:.4f} en {time_coreset_fast_backward:.2f} segundos.")


    # 6. Evaluación: Full Dataset (Baseline)
    clf_full = ml_model(gamma=gamma, **model_params)
    clf_full.fit(X_train, y_train.ravel())
    acc_full = clf_full.score(X_test, y_test.ravel())
    print(f"{metric} (Full data): {acc_full:.4f}")

    # 7. Evaluación: Random Subset (Baseline)
    random_indices = np.random.choice(len(X_train), size=len(coreset_X), replace=False)
    clf_random = ml_model(gamma=gamma, **model_params)
    clf_random.fit(X_train[random_indices], y_train[random_indices].ravel())
    acc_random = clf_random.score(X_test, y_test.ravel())
    print(f"{metric} (Random subset): {acc_random:.4f}")

    # 8. Evaluación: Nystroem Approximation
    time_start = perf_counter() 
    nystroem = Nystroem(kernel='rbf', gamma=gamma, n_components=len(coreset_X), random_state=42)
    X_train_nys = nystroem.fit_transform(X_train)
    X_test_nys = nystroem.transform(X_test)
    time_nystroem = perf_counter()  - time_start
    
    if args.problem_type == 'classification':
        clf_nys = SVC(kernel='linear')
    else:
        clf_nys = SVR(kernel='linear', epsilon=model_params.get('epsilon', 0.1))
    clf_nys.fit(X_train_nys, y_train.ravel())
    acc_nystroem = clf_nys.score(X_test_nys, y_test.ravel())
    print(f"{metric} (Nystroem full): {acc_nystroem:.4f} en {time_nystroem:.2f} segundos.")

    # 9 Evaluación de Kernel Herding (Baseline)
    time_start = perf_counter() 
    bw = 1/np.sqrt(2*gamma)
    k_x = make_gaussian_kernel(bw)
    if args.problem_type == 'regression':
        bw_y = 1/np.sqrt(2*epsilon)
        k_y = make_gaussian_kernel(bw_y)
    else:
        k_y = lambda x, y: 1.0 if x == y else 0.0  # Kernel delta para clasificación == k_rt
    kernel_herding = KernelHerding(m=len(coreset_X), phi_x=phi_X, phi_y=psi_y)
    index = kernel_herding.get_coreset(X_train, y_train)
    coreset_X_herding = X_train[index]
    coreset_y_herding = y_train[index]
    time_herding = perf_counter()  - time_start
    clf_herding = ml_model(gamma=gamma, **model_params)
    clf_herding.fit(coreset_X_herding, coreset_y_herding.ravel())
    acc_herding = clf_herding.score(X_test, y_test.ravel())
    print(f"{metric} (Kernel Herding): {acc_herding:.4f} en {time_herding:.2f} segundos.")

    # 10. Evaluación de Kernel Thinning (Baseline)
    time_start = perf_counter() 
    kernel_thinning = KernelThinning(alpha=args.alpha)
    
    coreset_indices_thinning = kernel_thinning.thin(phi_X(X_train), psi_y(y_train) if psi_y is not None else None, target_size=len(coreset_X))
    coreset_X_thinning = X_train[coreset_indices_thinning]
    coreset_y_thinning = y_train[coreset_indices_thinning]  
    time_thinning = perf_counter()  - time_start

    clf_thinning = ml_model(gamma=gamma, **model_params)
    clf_thinning.fit(coreset_X_thinning, coreset_y_thinning.ravel())
    acc_thinning = clf_thinning.score(X_test, y_test.ravel())
    print(f"{metric} (Kernel Thinning): {acc_thinning:.4f} en {time_thinning:.2f} segundos.")

    # 11. Evaluación de Kernel Thinning con RFF

    time_start = perf_counter() 
    bw = 1/np.sqrt(2*gamma)
    k_rt = make_gaussian_kernel(bw/np.sqrt(2))
    k_star = make_gaussian_kernel(bw )
    if args.problem_type == 'regression':
        bw_y = 1/np.sqrt(2*epsilon)
        k_y = make_gaussian_kernel(bw_y)
    else:
        k_y = lambda x, y: 1.0 if x == y else 0.0  # Kernel delta para clasificación == k_rt

    kt_rff = KernelThinningRFF(
        k_rt=k_rt,
        phi_rt=phi_X,   
        k_star=k_star,
        k_y=k_y,
        phi_rt_y=psi_y   
    )

    m_halvings = int(np.log2(len(X_train) / args.target_size)) 
    coreset_indices_thinning = kt_rff.thin(X_train, y=y_train if psi_y else None, m=m_halvings)

    time_thinning_rff = perf_counter() - time_start

    clf_thinning_rff = ml_model(gamma=gamma, **model_params)
    clf_thinning_rff.fit(coreset_X_thinning, coreset_y_thinning.ravel())
    acc_thinning_rff = clf_thinning_rff.score(X_test, y_test.ravel())
    print(f"{metric} (Kernel Thinning con RFF): {acc_thinning_rff:.4f} en {time_thinning_rff:.2f} segundos.")




    # 12 Flexible Kernel Thinning (Baseline)
    time_start = perf_counter() 
    bw = 1/np.sqrt(2*gamma)
    k_rt = make_gaussian_kernel(bw/np.sqrt(2))
    k_star = make_gaussian_kernel(bw )
    if args.problem_type == 'regression':
        bw_y = 1/np.sqrt(2*epsilon)
        k_y = make_gaussian_kernel(bw_y)
    else:
        k_y = lambda x, y: 1.0 if x == y else 0.0  # Kernel delta para clasificación == k_rt

    kt_rff = FlexibleKernelThinning(
        k_rt=k_rt,
        phi_rt=phi_X,   
        k_star=k_star,
        k_y=k_y,
        phi_rt_y=psi_y   
    )

    coreset_flexible_indices_thinning = kt_rff.thin_fraction(X_train, y=y_train, p=args.target_size/100)
    time_flexible_thinning_rff = perf_counter() - time_start

    X_flexible_thinning_rff = X_train[coreset_flexible_indices_thinning]
    y_flexible_thinning_rff = y_train[coreset_flexible_indices_thinning] 

    clf_thinning_rff = ml_model(gamma=gamma, **model_params)
    clf_thinning_rff.fit(X_flexible_thinning_rff, y=y_flexible_thinning_rff.ravel())
    acc_flexible_thinning_rff = clf_thinning_rff.score(X_test, y_test.ravel())

    print(f"{metric} (FLEXIBLE Kernel Thinning con RFF): {acc_flexible_thinning_rff:.4f} en {time_flexible_thinning_rff:.2f} segundos.")
    print(f'Flexible Thinning RFF seleccionó {len(coreset_flexible_indices_thinning)} muestras, objetivo era {args.target_size}% ({int(args.target_size/100*len(X_train))} muestras)')


    # 9. Guardar Resultados
    results = {
        'dataset': dataset_name,
        'task_id': task_id,
        'coreset_size': len(coreset_X),
        'backward_thinning': acc_backward_thinning,
        'fast_backward': acc_fast_backward,
        'full': acc_full,
        'random': acc_random,
        'nystroem': acc_nystroem,
        'herding': acc_herding,
        'thinning': acc_thinning,
        'thinning_rff': acc_thinning_rff,
        'flexible_thinning_rff': acc_flexible_thinning_rff,
        'time_coreset': time_coreset_backward,
        'time_nystroem': time_nystroem,
        'time_herding': time_herding,
        'time_thinning': time_thinning,
        'time_thinning_rff': time_thinning_rff,
        'time_flexible_thinning_rff': time_flexible_thinning_rff
    }
    
    final_path = os.path.join(args.path_to_save, args.problem_type)
    os.makedirs(final_path, exist_ok=True)
    file_name = f"{args.task_id}_results.csv"
    
    # Asegúrate de que save_data_to_csv maneje la creación del archivo
    save_path = os.path.join(final_path)

    save_data_to_csv(directory_path=save_path, file_name=file_name, data=results)
    print(f"Resultados guardados en {save_path}/{file_name}")

if __name__ == "__main__":
    main()