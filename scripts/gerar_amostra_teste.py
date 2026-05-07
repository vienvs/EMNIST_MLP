"""
Gera um recorte estratificado do conjunto de teste do EMNIST Balanced
para ser usado pelo app.py. Pega N amostras por classe, aplica a mesma
correcao de orientacao do notebook (rotacao -90 graus + flip horizontal)
e salva em data/test_sample.npz.

Uso:
    python scripts/gerar_amostra_teste.py

Saida:
    data/test_sample.npz  (imagens, labels e classes)
"""
import os
import numpy as np
import torchvision

SAMPLES_POR_CLASSE = 30
SEED = 42
DESTINO = os.path.join("data", "test_sample.npz")


def corrigir_orientacao_array(img):
    """Mesma correcao usada no notebook: rot90 com k=-1 + fliplr.
    Opera sobre numpy uint8 (sem precisar de torch aqui)."""
    girada = np.rot90(img, k=-1)
    corrigida = np.fliplr(girada)
    return corrigida.copy()


def main():
    print("Carregando EMNIST Balanced (test split)...")
    ds = torchvision.datasets.EMNIST(
        root="./data", split="balanced", train=False, download=True,
    )

    imagens_cru = ds.data.numpy()
    labels = ds.targets.numpy()
    classes = np.array(ds.classes)

    print(f"  total de amostras : {len(imagens_cru):,}")
    print(f"  classes           : {len(classes)}")

    rng = np.random.default_rng(SEED)

    indices_finais = []
    for c in range(len(classes)):
        candidatos = np.where(labels == c)[0]
        n = min(SAMPLES_POR_CLASSE, len(candidatos))
        escolhidos = rng.choice(candidatos, size=n, replace=False)
        indices_finais.extend(int(i) for i in escolhidos)

    indices_finais = np.array(indices_finais, dtype=np.int64)
    rng.shuffle(indices_finais)

    imagens_selecionadas = imagens_cru[indices_finais]
    labels_selecionados = labels[indices_finais].astype(np.int64)

    imagens_corrigidas = np.empty_like(imagens_selecionadas)
    for i in range(len(imagens_selecionadas)):
        imagens_corrigidas[i] = corrigir_orientacao_array(imagens_selecionadas[i])

    os.makedirs(os.path.dirname(DESTINO), exist_ok=True)
    np.savez_compressed(
        DESTINO,
        images=imagens_corrigidas,
        labels=labels_selecionados,
        classes=classes,
    )

    tam_kb = os.path.getsize(DESTINO) / 1024
    print(f"\nArquivo salvo em: {DESTINO}")
    print(f"  amostras      : {len(labels_selecionados):,}")
    print(f"  tamanho final : {tam_kb:,.1f} KB")


if __name__ == "__main__":
    main()
