# SAM Change Detection MVP

MVP de detecção de mudanças usando o Segment Anything Model (SAM) com foco em rodar localmente (CPU, MPS ou CUDA). O app recebe um par de imagens *antes/depois*, roda o SAM duas vezes com o mesmo grid de prompts, compara as máscaras com métricas ajustáveis e retorna um mapa visual das mudanças acompanhado de um resumo textual.

## Visão geral do que está pronto

- Segmentação automática com SAM `vit_b` + patch para evitar tensores `float64` em Apple Silicon.
- Equalização opcional de histogramas (T1 → T0) para reduzir falsos positivos causados por iluminação.
- Grid fixo de prompts configurável (até 64x64) para garantir que o modelo observe os mesmos pontos nas duas datas.
- Classificação em objetos **novos**, **removidos**, **modificados** e **estáveis**, incluindo deltas de área e deslocamento de centróide.
- Interface Gradio com sliders para todos os limiares relevantes, além de overlays de máscaras e mapa de mudança colorido.

## Pipeline de detecção

1. Normalizamos as duas imagens (RGB, uint8 e resolução compatível). Opcionalmente aplicamos histogram matching em T1.
2. Limitamos a maior dimensão para `max_image_size` (default 1536 px) para manter a inferência viável.
3. O SAM gera máscaras para *antes* e *depois* usando o mesmo grid fixo de prompts.
4. Filtramos máscaras pequenas (`min_area`) e comparamos cada máscara de T0 com T1 usando IoU + Hausdorff.
5. Um par é aceito se atingir o limiar de IoU **ou** o limiar do score combinado. A partir daí:
   - sem par em T1 → objeto removido;
   - sem par em T0 → objeto novo;
   - par aceito + Δ área / Δ centróide acima dos limiares → modificado;
   - caso contrário → estável.
6. Geramos overlays, mapa colorido, lista de top mudanças e estatísticas agregadas para a interface.

## Componentes e estrutura

```
├─ app.py                     # Interface Gradio, sliders e orquestração
├─ mvp_sam/
│  ├─ sam_change_detector.py  # Carregamento do SAM, matching e pós-processamento
│  ├─ visualization.py        # Overlays e mapa de calor das mudanças
│  └─ sam_patches.py          # Patch para o generator do SAM rodar em MPS
├─ checkpoints/README.md      # Passo a passo para posicionar o .pth
└─ requirements.txt
```

`SAMChangeDetector` encapsula todo o fluxo: alinhamento das imagens, geração das máscaras, matching, classificação das mudanças e normalização dos pesos IoU/Hausdorff. O módulo `visualization` cuida apenas do pós-processamento gráfico.

## Requisitos rápidos

- Python 3.11
- PyTorch 2.x com backend compatível (`cpu`, `mps` ou `cuda`).
- Dependências listadas em `requirements.txt` (Gradio, OpenCV, scikit-image, etc.).
- Checkpoint `sam_vit_b_01ec64.pth` (≈360 MB). Outros modelos do SAM funcionam se ajustados manualmente.

## Instalação e setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Dica Apple Silicon: se ainda não tiver PyTorch instalado, use  
> `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu`

### Checkpoint do SAM

1. Baixe `sam_vit_b_01ec64.pth` do repositório oficial do SAM.
2. Coloque o arquivo em `checkpoints/` (já existe um README lá dentro com instruções rápidas).
3. Opcional: defina `SAM_CHECKPOINT_PATH=/caminho/para/sam_vit_b_01ec64.pth` se desejar usar outro local ou nome.

Se nenhum checkpoint for encontrado, o app lança um erro amigável na interface.

## Executando a interface Gradio

```bash
python app.py
```

- A UI abre no navegador padrão (Gradio) e também envia um link local.
- Faça upload das imagens **Antes** e **Depois**, ajuste os parâmetros e clique em **Detectar mudanças**.
- Saídas:
  - `Antes + Máscaras`: overlay colorido com todas as máscaras aceitas pelo SAM.
  - `Depois + Máscaras`: mesmo overlay para a imagem T1.
  - `Mapa de Mudanças`: legenda fixa (verde = novos, vermelho = removidos, amarelo = modificados, cinza = estável).
  - Resumo textual: contagens, pesos usados no matching e top 5 mudanças (IoU, Hausdorff, Δ área e Δ centróide).

## Parâmetros e métricas disponíveis

- **Limiar IoU para matching** – Interseção sobre União entre a máscara antes/depois (0–1). Serve como critério direto de similaridade.
- **Peso Hausdorff no matching** – Controla o quanto o score final privilegia o contorno (Hausdorff) versus a sobreposição (IoU).  
  Score final = `w_iou * IoU + w_hausdorff * SimilaridadeHausdorff`, com `w_iou + w_hausdorff = 1`.
- **Score final de matching (implícito)** – Além do limiar de IoU, o par também precisa passar pelo limiar de `score` configurado no detector (default 0.5).
- **Δ Área mínima (fração)** – Variação relativa entre as áreas das máscaras aceitas. Ex.: 0.15 = mudou 15% ou mais.
- **Δ centróide mínima** – Distância entre os centróides antes/depois normalizada pelo tamanho da imagem. Detecta deslocamentos mesmo quando a área permanece parecida.
- **Área mínima do objeto** – Remove componentes pequenos antes do matching para reduzir ruído (default 256 px²).
- **Histogram matching T1→T0** – Ajusta o histograma da imagem *depois* para se parecer com *antes*, reduzindo diferenças de sensor/iluminação.
- **Grid fixo de prompts (pontos/linha)** – Define o número de prompts regulares enviados ao SAM (entre 8 e 64). Mais pontos = mais máscaras e custo computacional.
- **Resolução máxima para processamento** – Reamostramos as imagens para que a maior borda não passe desse valor. Ajuda a manter a inferência dentro do limite de memória.

## Como funciona por baixo dos panos

1. **Preparação**: `SAMChangeDetector` garante imagens RGB `uint8`, iguala histogramas se solicitado e redimensiona para respeitar `max_image_edge`.
2. **Patch MPS**: `ensure_float32_mps_patch` substitui `_process_batch` do `SamAutomaticMaskGenerator` para trabalhar só com `float32` em MPS.
3. **Segmentação com grid fixo**: usamos `build_all_layer_point_grids` para gerar o mesmo conjunto de prompts em T0 e T1, garantindo alinhamento espacial.
4. **Matching**: para cada máscara de T0 buscamos a melhor de T1 baseada no score híbrido IoU + Hausdorff. O processo evita reuso de máscaras para manter o matching 1:1.
5. **Classificação**: comparamos variação de área e deslocamento de centróide; aplicamos filtros de área mínima e contabilizamos novos/removidos/modificados/estáveis.
6. **Visualização**: `overlay_masks` colore as máscaras originais e `render_change_map` pinta o mapa final por classe, reaproveitando as máscaras escaladas.
7. **Resumo textual**: `_build_summary` destaca contagens, pesos usados e até cinco principais mudanças (ordenadas pela lista de modificados).

## Dicas de uso e performance

- Ajuste `min_area` para reduzir ruído em cenas urbanas densas; para vegetação ou objetos grandes, valores entre 512 e 2048 costumam ajudar.
- `max_image_size` em 1024–2048 mantém o throughput viável em CPU/MPS sem perder muito detalhe.
- Para detectar mudanças sutis, aumente o grid e reduza `min_area`, mas monitore o tempo de inferência (matching é `O(n²)` no número de máscaras).
- Se o resultado estiver muito ruidoso:
  - reduza o peso Hausdorff (deixe mais IoU),
  - aumente o limiar de IoU,
  - desative histogram matching quando as imagens já estão balanceadas.

## Limitações conhecidas

- SAM não foi treinado especificamente para change detection; ainda precisamos de pós-processamento pesado para evitar falsos positivos.
- Diferenças de sombra, sazonalidade ou pequenos objetos muito próximos costumam confundir o matching.
- O processo de matching escala quadraticamente com o número de máscaras; grids grandes em imagens de alta resolução tornam o app lento.
- O modelo `vit_b` consome ~4 GB de RAM durante inferência. Feche aplicativos pesados antes de rodar no MacBook M4.

