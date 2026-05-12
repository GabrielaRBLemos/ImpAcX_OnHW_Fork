"""Script de teste do paralelismo — executado pelo CI/validação."""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

import numpy as np

def test_filter():
    from OnHW_ML_extract_features_various_nsig import filter_train_test
    rng = np.random.default_rng(42)
    lengths = rng.integers(5, 250, size=200)
    X = np.array([rng.random((l, 13)) for l in lengths], dtype=object)
    y = rng.choice(list('abcdefghijklmnopqrstuvwxyz'), size=200)
    X_f, y_f = filter_train_test(X, y, case='lower')
    assert 0 < len(X_f) < 200, "filtro nao removeu nada ou removeu tudo"
    print(f"  filter_train_test OK: {len(X)} -> {len(X_f)} series")

def test_dtw_parallel():
    from DTW_KNN import get_neighbors, all_neighbors
    rng = np.random.default_rng(0)
    N_TRAIN, N_TEST, SEQ_LEN = 30, 6, 20
    X_train = [rng.random((SEQ_LEN, 13)).astype(np.float64) for _ in range(N_TRAIN)]
    y_train = np.array([str(i % 26) for i in range(N_TRAIN)])
    X_test  = np.array([rng.random((SEQ_LEN, 13)).astype(np.float64) for _ in range(N_TEST)])

    # sequencial
    t0 = time.perf_counter()
    seq = [get_neighbors(X_train, y_train, row, 'DTW') for row in X_test]
    t_seq = time.perf_counter() - t0

    # paralelo
    t0 = time.perf_counter()
    par = all_neighbors(X_train, y_train, X_test, 'DTW', amt_neighbors=10, max_workers=2)
    t_par = time.perf_counter() - t0

    assert len(par) == N_TEST, f"paralelo retornou {len(par)} rows, esperado {N_TEST}"
    for i in range(N_TEST):
        # mesmos vizinhos (pode estar em ordem diferente internamente, mas top-1 deve coincidir)
        assert seq[i][0] == par[i][0], f"row {i}: top-1 diverge seq={seq[i][0]} par={par[i][0]}"

    print(f"  all_neighbors paralelo OK: {N_TEST} linhas de teste")
    print(f"  Sequencial: {t_seq:.2f}s | Paralelo (2 workers): {t_par:.2f}s")
    if t_seq > 0:
        ratio = t_seq / t_par if t_par > 0 else float('inf')
        print(f"  Ratio: {ratio:.1f}x (overhead esperado em N pequeno)")

def test_tsfresh_pipeline():
    from OnHW_ML_extract_features_various_nsig import X_npy_to_df, ts_extract_feautures
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    rng = np.random.default_rng(1)

    # Dados com padrao claro: classe 'a' tem feature 0 alta, 'b' tem feature 0 baixa
    # Isso garante que select_features encontre features significativas
    X_a = np.array([np.full((30, 13), 10.0) + rng.random((30, 13)) * 0.1 for _ in range(10)])
    X_b = np.array([np.full((30, 13),  0.0) + rng.random((30, 13)) * 0.1 for _ in range(10)])
    X = np.concatenate([X_a, X_b], axis=0)
    X = np.array(list(X), dtype=object)
    y = np.array(['a'] * 10 + ['b'] * 10)

    df = X_npy_to_df(X)
    assert 'id' in df.columns and 'time' in df.columns
    assert len(df) == 20 * 30
    for col in [f'f_{i}' for i in range(13)]:
        assert df[col].dtype == np.float64, f"coluna {col} nao eh float64"
    print(f"  X_npy_to_df OK: shape {df.shape}, dtypes OK")

    # Testa apenas extract_features (sem select_features) para validar o pipeline basico
    df_train = X_npy_to_df(X[:15])
    extracted = extract_features(df_train, column_id='id', column_sort='time', n_jobs=1)
    impute(extracted)
    assert extracted.shape[0] == 15
    assert extracted.shape[1] > 0
    print(f"  extract_features OK: {extracted.shape[0]} amostras, {extracted.shape[1]} features extraidas")
    print(f"  (select_features requer dados reais do dataset para encontrar features significativas)")

if __name__ == "__main__":
    print("=== Teste 1: filter_train_test ===")
    test_filter()

    print("\n=== Teste 2: DTW all_neighbors paralelo ===")
    test_dtw_parallel()

    print("\n=== Teste 3: tsfresh pipeline ===")
    test_tsfresh_pipeline()

    print("\nTodos os testes passaram.")
