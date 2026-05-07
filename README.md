# Classificador EMNIST com MLP

Trabalho P1 de Aprendizagem de Maquina. Pontificia Universidade Catolica de Sao Paulo, 2026.

Autor: Vinicius de Lucena, RA 00376040.

Classificacao dos caracteres manuscritos do **EMNIST Balanced** (47 classes, digitos e letras) usando duas arquiteturas de rede totalmente conectada (MLP) em PyTorch.

## Destaques

- Duas arquiteturas MLP (com e sem BatchNorm) treinadas e comparadas.
- Pipeline completo: transforms, split treino/val/teste, TensorBoard, checkpoint, matriz de confusao e exemplos de predicao.
- Extensoes simples alem do basico: test-time augmentation (TTA) e ensemble das duas redes por media de probabilidades.
- Aplicacao Streamlit (item 3.6 opcional do enunciado) que recebe o desenho do usuario ou um upload de imagem e retorna a classe prevista + top 3.

## Estrutura

```
EMNIST_MLP_v2/
|-- emnist_mlp.ipynb       notebook principal (treino + avaliacao)
|-- app.py                 aplicacao Streamlit
|-- passo_a_passo.md       explicacao de origem de cada decisao tecnica
|-- requirements.txt       dependencias do projeto
|-- checkpoints/           .pth do melhor modelo (gerado pelo notebook)
|-- runs/                  logs do TensorBoard (gerado pelo notebook)
`-- data/                  dataset EMNIST baixado automaticamente
```

## Como rodar

### 1. Instalar dependencias

```
pip install -r requirements.txt
```

Recomendado usar um ambiente virtual (`python -m venv .venv` e depois ativar).

### 2. Treinar os modelos

Abrir o notebook no Jupyter e rodar todas as celulas:

```
jupyter notebook emnist_mlp.ipynb
```

No menu, `Kernel` -> `Restart & Run All`. O dataset e baixado na pasta `data/` na primeira execucao (pode demorar alguns minutos). O notebook gera dois checkpoints:

- `checkpoints/MLP_A_melhor.pth`
- `checkpoints/MLP_B_melhor.pth`

### 3. Visualizar o TensorBoard

As curvas de loss e acuracia dos dois modelos ficam em `runs/`:

```
tensorboard --logdir=runs
```

Abrir `http://localhost:6006` no navegador.

### 4. Rodar a aplicacao Streamlit

Com pelo menos um checkpoint salvo em `checkpoints/`, rodar:

```
streamlit run app.py
```

A app abre em `http://localhost:8501`. Voce pode:

- Desenhar um digito ou letra no canvas.
- Fazer upload de uma imagem (PNG ou JPG) com fundo claro e traco escuro.
- Ativar TTA e ensemble na sidebar para ver o impacto.

## Arquitetura dos modelos

Ambas compartilham a mesma topologia `784 -> 256 -> 128 -> 47`. A unica diferenca e o BatchNorm:

| Modelo | Camadas | BatchNorm |
|--------|---------|-----------|
| MLP_A  | Linear + ReLU + Dropout  | nao   |
| MLP_B  | Linear + **BN** + ReLU + Dropout | sim |

O objetivo e isolar o efeito do BatchNorm, que e explicitamente permitido pelo enunciado do trabalho (item 3.2).

## Hiperparametros de treino

- Otimizador: Adam (`lr=1e-3`, `weight_decay=1e-5`).
- Scheduler: `StepLR(step_size=8, gamma=0.5)`.
- Perda: `CrossEntropyLoss`.
- Batch size: 128.
- Epocas maximas: 40 (com early stopping de paciencia 5 em `val_loss`).
- Seed: 42.

## Referencias

O codigo segue os padroes dos notebooks da disciplina. Ver `passo_a_passo.md` para o mapeamento detalhado de cada decisao (qual notebook serviu de base, o que foi adaptado, o que e extensao propria).
