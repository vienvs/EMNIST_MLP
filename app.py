"""
Aplicacao Streamlit para o trabalho EMNIST MLP

Duas abas:

- "Desenhar": o usuario traca um caractere no canvas e ve a predição.
- "EMNIST":   sorteia amostras reais do conjunto de teste, mostra o
              rotulo real lado a lado com a predição do modelo.

Os modelos são carregados via load_state_dict a partir dos arquivos
.pth produzidos pelo notebook emnist_mlp.ipynb.
"""
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


class MLP_A(nn.Module):
    def __init__(self, n_classes=47):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.rede(x)


class MLP_B(nn.Module):
    def __init__(self, n_classes=47):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.rede(x)


# Media e desvio padrao calculados no proprio notebook (celula 2.2).
MEDIA = 0.1751
DESVIO = 0.3331

# Lista canonica das 47 classes do EMNIST Balanced.
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

ARQUIVOS_CKPT = {
    "MLP_A": "checkpoints/MLP_A_melhor.pth",
    "MLP_B": "checkpoints/MLP_B_melhor.pth",
}

ARQ_AMOSTRA = os.path.join("data", "test_sample.npz")


@st.cache_resource(show_spinner=False)
def carregar_modelo(nome):
    """Carrega um checkpoint. Devolve (modelo, info_dict) ou (None, None)."""
    caminho = ARQUIVOS_CKPT[nome]
    if not os.path.exists(caminho):
        return None, None
    ckpt = torch.load(caminho, map_location="cpu", weights_only=False)
    classe = MLP_B if nome == "MLP_B" else MLP_A
    modelo = classe(ckpt.get("n_classes", 47))
    modelo.load_state_dict(ckpt["modelo_state"])
    modelo.eval()
    info = {
        "nome": nome,
        "epoca": ckpt.get("epoca", "-"),
        "val_acc": ckpt.get("val_acc", None),
        "val_loss": ckpt.get("val_loss", None),
    }
    return modelo, info


@st.cache_resource(show_spinner=False)
def carregar_amostras_bundled():
    """Le o arquivo data/test_sample.npz com amostras reais do EMNIST
    (ja com orientacao corrigida). Retorna None se o arquivo nao existir."""
    if not os.path.exists(ARQ_AMOSTRA):
        return None
    pacote = np.load(ARQ_AMOSTRA, allow_pickle=True)
    return {
        "images": pacote["images"],
        "labels": pacote["labels"],
        "classes": [str(c) for c in pacote["classes"].tolist()],
    }


def tensor_normalizado(img28_uint8):
    """Recebe imagem 28x28 uint8 (fundo preto, traco branco) e devolve
    tensor [1, 1, 28, 28] normalizado com a media e desvio do treino."""
    t = torch.from_numpy(img28_uint8.astype(np.float32) / 255.0).unsqueeze(0)
    t = (t - MEDIA) / DESVIO
    return t.unsqueeze(0)


def canvas_para_28x28(image_data):
    """Converte a saida RGBA do canvas para 28x28 no formato EMNIST."""
    if image_data is None:
        return None
    rgba = np.array(image_data, dtype=np.uint8)
    if rgba[:, :, 3].max() == 0:
        return None

    cinza = ImageOps.grayscale(Image.fromarray(rgba[:, :, :3]))
    invertida = ImageOps.invert(cinza)
    arr = np.array(invertida)

    mascara = arr > 30
    if not mascara.any():
        return None
    linhas = np.where(np.any(mascara, axis=1))[0]
    colunas = np.where(np.any(mascara, axis=0))[0]
    recorte = Image.fromarray(arr[linhas[0]:linhas[-1] + 1,
                                  colunas[0]:colunas[-1] + 1])

    h, w = recorte.height, recorte.width
    if h > w:
        nh, nw = 20, max(1, int(round(w * 20 / h)))
    else:
        nh, nw = max(1, int(round(h * 20 / w))), 20
    recorte = recorte.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas28 = Image.new("L", (28, 28), color=0)
    canvas28.paste(recorte, ((28 - nw) // 2, (28 - nh) // 2))
    return np.array(canvas28, dtype=np.uint8)


def inferir(modelo, tensor):
    with torch.no_grad():
        return F.softmax(modelo(tensor), dim=1).squeeze(0).numpy()


def inferir_tta(modelo, img28):
    """5 versoes (original + 4 deslocadas em 1 pixel). Media das Softmaxes."""
    versoes = [img28]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        versoes.append(np.roll(img28, shift=(dy, dx), axis=(0, 1)))
    probs = [inferir(modelo, tensor_normalizado(v)) for v in versoes]
    return np.mean(probs, axis=0)


def predizer(modelos, img28, usar_tta):
    """Media de Softmax entre todos os modelos. Aplica TTA se ativo."""
    probs = []
    for m in modelos:
        if usar_tta:
            probs.append(inferir_tta(m, img28))
        else:
            probs.append(inferir(m, tensor_normalizado(img28)))
    return np.mean(probs, axis=0)

st.set_page_config(
    page_title="EMNIST MLP",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apenas dois estilos proprios: o card de info do checkpoint e o
# glyph grande da predicao. Todo o resto usa o tema nativo do
# Streamlit, configurado em .streamlit/config.toml (tema claro).

st.markdown(
    """
    <style>
    .info-card {
        background: var(--secondary-background-color, #F5F5F2);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 13px;
        line-height: 1.8;
    }
    .info-card code {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 12px;
    }
    .glyph {
        font-size: 96px;
        font-weight: 700;
        text-align: center;
        margin: 0;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Classificador EMNIST MLP")
st.caption("Trabalho P1 Vinicius de Lucena RA 00376040")

with st.sidebar:
    st.header("Configuração")

    disponiveis = [nome for nome, c in ARQUIVOS_CKPT.items() if os.path.exists(c)]
    if not disponiveis:
        st.error(
            "Nenhum checkpoint encontrado em `checkpoints/`. "
            "Rode o notebook `emnist_mlp.ipynb` para treinar os modelos."
        )
        st.stop()

    modelo_nome = st.selectbox("Modelo principal", disponiveis)

    modelo_principal, info_principal = carregar_modelo(modelo_nome)

    # Card com as informacoes do checkpoint carregado
    if info_principal is not None:
        val_acc = info_principal["val_acc"]
        val_loss = info_principal["val_loss"]
        linhas = [f"<b>{info_principal['nome']}</b>"]
        linhas.append(f"melhor epoca &nbsp; <code>{info_principal['epoca']}</code>")
        if val_acc is not None:
            linhas.append(f"val_acc &nbsp; <code>{val_acc * 100:.2f}%</code>")
        if val_loss is not None:
            linhas.append(f"val_loss &nbsp; <code>{val_loss:.4f}</code>")
        st.markdown(
            "<div class='info-card'>" + "<br>".join(linhas) + "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    usar_tta = st.checkbox(
        "Test-time augmentation (TTA)",
        value=False,
        help="Passa 5 versoes da imagem (original + 4 deslocadas em 1 pixel) "
             "e tira a media das Softmaxes. Mesma tecnica da secao 7.2 do notebook.",
    )
    usar_ensemble = st.checkbox(
        "Ensemble A + B",
        value=False,
        disabled=len(disponiveis) < 2,
        help="Media das Softmaxes de MLP_A e MLP_B. Mesma tecnica da secao 7.3 "
             "do notebook.",
    )

    if usar_ensemble:
        modelos_usados = [carregar_modelo(nome)[0] for nome in disponiveis]
    else:
        modelos_usados = [modelo_principal]

    st.divider()
    st.caption(
        "As amostras reais do EMNIST são carregadas de "
        f"`{ARQ_AMOSTRA}`."
    )


def exibir_predicao(img28, rotulo_real=None):
    """Mostra imagem, classe prevista em destaque e top-3."""
    probs = predizer(modelos_usados, img28, usar_tta)
    top_idx = int(np.argmax(probs))
    classe_top = CLASSES[top_idx]
    conf_top = float(probs[top_idx])

    if rotulo_real is None:
        titulo = "Predicao"
        cor = None
    elif classe_top == rotulo_real:
        titulo = "Acerto"
        cor = "#1F7A3D"
    else:
        titulo = "Erro"
        cor = "#C2410C"

    col_img, col_glyph, col_bars = st.columns([1, 1.1, 2])

    with col_img:
        st.image(img28, caption="entrada 28x28", width=120, clamp=True)

    with col_glyph:
        cor_titulo = cor if cor else "inherit"
        cor_glyph = cor if cor else "inherit"
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:10px; letter-spacing:2px;"
            f" text-transform:uppercase; color:{cor_titulo}; font-weight:600;'>"
            f"{titulo}</div>"
            f"<div class='glyph' style='color:{cor_glyph};'>{classe_top}</div>"
            f"<div style='font-size:13px; opacity:0.7;'>"
            f"{conf_top * 100:.2f}% confianca</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if rotulo_real is not None and classe_top != rotulo_real:
            st.markdown(
                f"<div style='text-align:center; font-size:12px; margin-top:6px;'>"
                f"real: <b style='color:#C2410C'>{rotulo_real}</b></div>",
                unsafe_allow_html=True,
            )

    with col_bars:
        st.markdown("**Top 3**")
        ord_idx = np.argsort(probs)[::-1][:3]
        for i in ord_idx:
            st.progress(
                float(probs[i]),
                text=f"**{CLASSES[i]}** - {probs[i] * 100:.2f}%",
            )

    with st.expander("Distribuição completa (47 classes)"):
        ord_full = np.argsort(probs)[::-1]
        tabela = {
            "rank": list(range(1, len(ord_full) + 1)),
            "classe": [CLASSES[i] for i in ord_full],
            "probabilidade": [f"{probs[i] * 100:.3f}%" for i in ord_full],
        }
        st.dataframe(tabela, use_container_width=True, height=320, hide_index=True)


aba_desenho, aba_emnist = st.tabs(["Desenhar", "EMNIST"])


# --- Aba Desenhar ---

with aba_desenho:
    col_canvas, col_resultado = st.columns([1, 1.5])

    with col_canvas:
        espessura = st.slider("Espessura do traco", 8, 40, 20, step=2)
        if st.button("Limpar canvas", use_container_width=True):
            st.session_state["canvas_rev"] = st.session_state.get("canvas_rev", 0) + 1

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=espessura,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.get('canvas_rev', 0)}",
        )

    with col_resultado:
        if canvas_result.image_data is not None:
            img28 = canvas_para_28x28(canvas_result.image_data)
            if img28 is not None:
                exibir_predicao(img28)
            else:
                st.info("Desenhe um caractere para ver a predicao.")
        else:
            st.info("Desenhe um caractere para ver a predicao.")


# --- Aba EMNIST: amostras reais do conjunto de teste ---

with aba_emnist:
    pacote = carregar_amostras_bundled()
    if pacote is None:
        st.warning(
            f"Arquivo `{ARQ_AMOSTRA}` nao encontrado. "
            f"Rode `python scripts/gerar_amostra_teste.py` para gera-lo."
        )
    else:
        imagens_ds = pacote["images"]
        labels_ds = pacote["labels"]
        classes_ds = pacote["classes"]

        st.caption(
            f"{len(labels_ds):,} amostras reais do conjunto de teste do "
            f"EMNIST (bundled em `{ARQ_AMOSTRA}`)."
        )

        col_f, col_n, col_b = st.columns([2, 2, 1])
        with col_f:
            filtro = st.selectbox(
                "Filtrar por classe",
                options=["(todas)"] + classes_ds,
                index=0,
            )
        with col_n:
            qtd = st.slider("Quantas amostras", 4, 24, 12, step=4)
        with col_b:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            sortear = st.button("Sortear", use_container_width=True)

        chave_filtro = f"{filtro}_{qtd}"
        if (
            "emnist_idx" not in st.session_state
            or sortear
            or chave_filtro != st.session_state.get("emnist_chave")
        ):
            st.session_state["emnist_chave"] = chave_filtro
            if filtro == "(todas)":
                pool = np.arange(len(labels_ds))
            else:
                alvo = classes_ds.index(filtro)
                pool = np.where(labels_ds == alvo)[0]

            if len(pool) == 0:
                st.session_state["emnist_idx"] = []
            else:
                rng = np.random.default_rng()
                n = min(qtd, len(pool))
                st.session_state["emnist_idx"] = rng.choice(
                    pool, size=n, replace=False
                ).tolist()

        indices = st.session_state.get("emnist_idx", [])
        if not indices:
            st.info("Nenhuma amostra para esse filtro.")
        else:
            resultados = []
            for idx in indices:
                img = imagens_ds[idx]
                real = classes_ds[labels_ds[idx]]
                probs = predizer(modelos_usados, img, usar_tta)
                pred_idx = int(np.argmax(probs))
                pred = CLASSES[pred_idx]
                conf = float(probs[pred_idx])
                resultados.append({
                    "img": img,
                    "real": real,
                    "pred": pred,
                    "conf": conf,
                    "correto": pred == real,
                    "probs": probs,
                })

            total = len(resultados)
            acertos = sum(1 for r in resultados if r["correto"])
            conf_media = sum(r["conf"] for r in resultados) / total if total else 0
            conf_acertos = (
                sum(r["conf"] for r in resultados if r["correto"])
                / max(acertos, 1)
            )

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Acertos", f"{acertos}/{total}")
            col_m2.metric("Taxa", f"{acertos / total * 100:.0f}%")
            col_m3.metric("Conf. media", f"{conf_media * 100:.1f}%")
            col_m4.metric("Conf. nos acertos", f"{conf_acertos * 100:.1f}%")

            st.markdown(" ")

            # Grid de miniaturas
            por_linha = 6
            opcoes_detalhe = []
            for inicio in range(0, len(resultados), por_linha):
                linha = resultados[inicio:inicio + por_linha]
                cols = st.columns(por_linha)
                for j, r in enumerate(linha):
                    with cols[j]:
                        simbolo = "OK" if r["correto"] else "X"
                        cor = "#1F7A3D" if r["correto"] else "#C2410C"
                        st.image(r["img"], use_container_width=True, clamp=True)
                        st.markdown(
                            f"<div style='text-align:center;'>"
                            f"<span style='font-weight:700; color:{cor}; font-size:18px;'>"
                            f"{simbolo} {r['pred']}</span>"
                            f"<div style='font-size:11px; opacity:0.7;'>"
                            f"real <b>{r['real']}</b> | {r['conf'] * 100:.0f}%</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        opcoes_detalhe.append(r)

            st.divider()

            # Drill-down em uma amostra
            rotulos_dd = [
                f"#{i + 1} | pred={r['pred']} | real={r['real']}"
                for i, r in enumerate(opcoes_detalhe)
            ]
            sel = st.selectbox("Analisar uma amostra em detalhe", options=rotulos_dd)
            if sel:
                r = opcoes_detalhe[rotulos_dd.index(sel)]
                exibir_predicao(r["img"], rotulo_real=r["real"])
