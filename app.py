"""
Aplicacao Streamlit para o trabalho EMNIST MLP.
Item 3.6 do enunciado (opcional, ponto extra).

Fluxo:
  1. Usuario desenha um caractere num canvas de 280x280 pixels (ou faz
     upload de uma imagem).
  2. O desenho e convertido para o formato 28x28 que a rede espera:
     escala de cinza, invertido (fundo preto, traco branco), recortado
     pela bounding box e centralizado num quadrado.
  3. A imagem normalizada (mesma media e desvio usados no treino) e
     passada para o modelo.
  4. Mostramos a classe prevista e a probabilidade Softmax das top 3
     classes.

As tecnicas opcionais de inferencia (TTA e ensemble) podem ser
ativadas na sidebar, para reproduzir o comportamento das secoes 7.2,
7.3 e 7.4 do notebook.
"""
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# ---------------------------------------------------------------------------
# Arquiteturas (copiadas do notebook emnist_mlp.ipynb, secao 3)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Constantes do EMNIST Balanced
# ---------------------------------------------------------------------------

# Media e desvio padrao calculados no proprio notebook (celula 2.2).
# Sao valores fixos do EMNIST Balanced, nao mudam entre execucoes.
MEDIA = 0.1751
DESVIO = 0.3331

# As 47 classes do EMNIST Balanced. A ordem e a mesma que o
# torchvision.datasets.EMNIST(split="balanced").classes retorna.
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

ARQUIVOS_CKPT = {
    "MLP_A": "checkpoints/MLP_A_melhor.pth",
    "MLP_B": "checkpoints/MLP_B_melhor.pth",
}


# ---------------------------------------------------------------------------
# Carregamento do checkpoint
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def carregar_modelo(nome):
    """Carrega um checkpoint da pasta checkpoints/ e devolve o modelo
    em modo eval. Retorna None se o arquivo nao existir."""
    caminho = ARQUIVOS_CKPT[nome]
    if not os.path.exists(caminho):
        return None
    ckpt = torch.load(caminho, map_location="cpu", weights_only=False)
    classe = MLP_B if nome == "MLP_B" else MLP_A
    modelo = classe(ckpt.get("n_classes", 47))
    modelo.load_state_dict(ckpt["modelo_state"])
    modelo.eval()
    return modelo


# ---------------------------------------------------------------------------
# Pipeline de pre-processamento da entrada do usuario
# ---------------------------------------------------------------------------

def tensor_normalizado(img28_uint8):
    """Recebe um array 28x28 uint8 (fundo preto, traco branco) e
    devolve um tensor [1, 1, 28, 28] normalizado com a mesma media e
    desvio usados no treino."""
    t = torch.from_numpy(img28_uint8.astype(np.float32) / 255.0).unsqueeze(0)
    t = (t - MEDIA) / DESVIO
    return t.unsqueeze(0)


def canvas_para_28x28(image_data):
    """Converte a saida RGBA do canvas (fundo branco, traco escuro)
    para uma imagem 28x28 no formato EMNIST (fundo preto, traco
    branco), centralizada."""
    if image_data is None:
        return None
    rgba = np.array(image_data, dtype=np.uint8)
    if rgba[:, :, 3].max() == 0:
        return None

    # converte RGB -> cinza e inverte (fundo preto, traco branco)
    cinza = ImageOps.grayscale(Image.fromarray(rgba[:, :, :3]))
    invertida = ImageOps.invert(cinza)
    arr = np.array(invertida)

    # bounding box do traco
    mascara = arr > 30
    if not mascara.any():
        return None
    linhas = np.where(np.any(mascara, axis=1))[0]
    colunas = np.where(np.any(mascara, axis=0))[0]
    recorte = Image.fromarray(arr[linhas[0]:linhas[-1] + 1,
                                  colunas[0]:colunas[-1] + 1])

    # redimensiona preservando proporcao para caber em 20x20
    h, w = recorte.height, recorte.width
    if h > w:
        nh, nw = 20, max(1, int(round(w * 20 / h)))
    else:
        nh, nw = max(1, int(round(h * 20 / w))), 20
    recorte = recorte.resize((nw, nh), Image.Resampling.LANCZOS)

    # cola no centro de um quadrado 28x28 preto
    canvas28 = Image.new("L", (28, 28), color=0)
    canvas28.paste(recorte, ((28 - nw) // 2, (28 - nh) // 2))
    return np.array(canvas28, dtype=np.uint8)


def upload_para_28x28(arquivo):
    """Converte um arquivo de imagem enviado pelo usuario para o mesmo
    formato 28x28 esperado pelo modelo. Assume que a imagem tem fundo
    claro e traco escuro (como o canvas)."""
    if arquivo is None:
        return None
    img = Image.open(arquivo).convert("RGBA")
    return canvas_para_28x28(np.array(img))


# ---------------------------------------------------------------------------
# Inferencia (com e sem TTA / ensemble)
# ---------------------------------------------------------------------------

def inferir(modelo, tensor):
    with torch.no_grad():
        return F.softmax(modelo(tensor), dim=1).squeeze(0).numpy()


def inferir_tta(modelo, img28):
    """Media de Softmax entre a imagem original e 4 versoes deslocadas
    em 1 pixel. Mesmo TTA da secao 7.2 do notebook."""
    versoes = [img28]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        versoes.append(np.roll(img28, shift=(dy, dx), axis=(0, 1)))
    probs = [inferir(modelo, tensor_normalizado(v)) for v in versoes]
    return np.mean(probs, axis=0)


def predizer(modelos, img28, usar_tta):
    """Aplica TTA (opcional) e media entre modelos (se houver mais de
    um). Retorna o vetor de probabilidades final."""
    probs = []
    for m in modelos:
        if usar_tta:
            probs.append(inferir_tta(m, img28))
        else:
            probs.append(inferir(m, tensor_normalizado(img28)))
    return np.mean(probs, axis=0)


# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EMNIST MLP",
    page_icon="A",
    layout="wide",
)

st.title("Classificador EMNIST - MLP")
st.caption("Trabalho P1 de Aprendizagem de Maquina | PUC-SP")

# --- Sidebar ---

with st.sidebar:
    st.header("Configuracao")

    disponiveis = [nome for nome, c in ARQUIVOS_CKPT.items() if os.path.exists(c)]
    if not disponiveis:
        st.error(
            "Nenhum checkpoint encontrado em `checkpoints/`. "
            "Rode o notebook `emnist_mlp.ipynb` para treinar os modelos "
            "antes de abrir a app."
        )
        st.stop()

    modelo_nome = st.selectbox("Modelo principal", disponiveis)

    usar_tta = st.checkbox(
        "Test-time augmentation (TTA)",
        value=False,
        help="Passa 5 versoes da imagem pela rede (original + 4 deslocadas) "
             "e tira a media das Softmaxes. Mesma tecnica da secao 7.2 do "
             "notebook.",
    )
    usar_ensemble = st.checkbox(
        "Ensemble A + B",
        value=False,
        disabled=len(disponiveis) < 2,
        help="Se os dois modelos estao disponiveis, faz a media das "
             "Softmaxes das duas arquiteturas. Mesma tecnica da secao 7.3 "
             "do notebook.",
    )

    st.divider()
    st.caption(
        "Para ver detalhes das tecnicas, abra `passo_a_passo.md`. "
        "Os modelos sao carregados do disco (item 3.4 do enunciado)."
    )


# --- Carrega modelos ---
modelo_principal = carregar_modelo(modelo_nome)
if usar_ensemble:
    modelos_usados = [carregar_modelo(nome) for nome in disponiveis]
else:
    modelos_usados = [modelo_principal]


# --- Duas formas de entrada: canvas e upload ---

aba_desenho, aba_upload = st.tabs(["Desenhar", "Upload de imagem"])


def mostrar_predicao(img28):
    """Roda inferencia e mostra classe prevista + top-3."""
    if img28 is None:
        st.info("Nenhuma imagem para classificar.")
        return

    probs = predizer(modelos_usados, img28, usar_tta)
    top_idx = int(np.argmax(probs))
    classe_top = CLASSES[top_idx]
    conf_top = float(probs[top_idx])

    col_glyph, col_bars = st.columns([1, 2])

    with col_glyph:
        st.markdown(
            f"<h1 style='text-align:center; font-size:96px; margin:0;'>"
            f"{classe_top}</h1>",
            unsafe_allow_html=True,
        )
        st.caption(f"Confianca: **{conf_top * 100:.2f}%**")
        st.image(img28, caption="imagem 28x28", width=120, clamp=True)

    with col_bars:
        st.subheader("Top 3")
        ord_idx = np.argsort(probs)[::-1][:3]
        for i in ord_idx:
            st.progress(
                float(probs[i]),
                text=f"**{CLASSES[i]}** - {probs[i] * 100:.2f}%",
            )


with aba_desenho:
    col_canvas, col_resultado = st.columns([1, 1])

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
            mostrar_predicao(img28)
        else:
            st.info("Desenhe um caractere para ver a predicao.")


with aba_upload:
    arq = st.file_uploader(
        "Envie uma imagem (PNG, JPG)",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Idealmente uma imagem com fundo claro e traco escuro, "
             "parecido com o que o canvas produz.",
    )
    if arq is not None:
        img28 = upload_para_28x28(arq)
        if img28 is not None:
            mostrar_predicao(img28)
        else:
            st.warning("Nao foi possivel processar a imagem. Verifique se "
                       "ela contem um traco visivel sobre fundo claro.")
