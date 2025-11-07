# SAM Change Detection MVP

Sistema de detecção de mudanças baseado no Segment Anything Model (SAM) pensado para rodar localmente no seu MacBook M4. A estratégia segue o pipeline conceitual:

1. SAM segmenta a imagem **antes** → obtém máscaras A
2. SAM segmenta a imagem **depois** → obtém máscaras B
3. Comparamos A vs. B usando IoU
4. Classificamos os objetos em **novos**, **removidos** ou **modificados** e geramos um mapa visual interpretável

## Por que SAM?

**Vantagens**
- ✅ SAM é gratuito e oferece segmentação potente out-of-the-box
- ✅ Responde muito bem a mudanças estruturais (construções, estradas, desmatamento)
- ✅ Dispensa um modelo específico de change detection
- ✅ Totalmente interpretável: cada máscara é visível e ajustável

**Desvantagens**
- ❌ Não é otimizado para change detection — exige pós-processamento
- ❌ Pode produzir muitos falsos positivos se os limiares forem baixos
- ❌ Mudanças muito sutis (iluminação, objetos pequenos) são difíceis
- ❌ Precisa de tuning e filtragem extra para casos ruidosos

## Requisitos

- Python 3.10+
- Torch 2.x com suporte a CPU/MPS (Apple Silicon)
- Dependências Python listadas em `requirements.txt`
- Peso do modelo **SAM base (vit_b)** — recomendado `sam_vit_b_01ec64.pth`

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Baixe o checkpoint do SAM (aprox. 360 MB) no site oficial e informe o caminho via variável de ambiente:

```bash
export SAM_CHECKPOINT_PATH="$(pwd)/checkpoints/sam_vit_b_01ec64.pth"
```

> **Dica**: para Apple Silicon use `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu` se ainda não tiver torch instalado.

## Executando o app (Gradio)

```bash
python app.py
```

A interface abre no navegador. Faça upload das imagens **antes** e **depois**, ajuste os sliders e clique em **Detectar mudanças**. Você verá:

- Imagem antes com as máscaras identificadas pelo SAM
- Imagem depois com as novas máscaras
- Mapa de mudanças colorido (Verde = novos, Vermelho = removidos, Amarelo = modificados, Cinza = estáveis)
- Resumo textual com contagens e principais alterações (IoU, delta de área, deslocamento de centróide)

## Estratégias e parâmetros

- **IoU para matching**: máscara só é pareada se IoU ≥ limiar (default 0.45)
- **Objetos novos**: máscaras na imagem *depois* sem par correspondente
- **Objetos removidos**: máscaras na imagem *antes* sem par correspondente
- **Objetos modificados**: pares com IoU válido porém variação de área ≥ limiar ou deslocamento de centróide ≥ limiar
- **Filtro de área mínima**: descarta pequenos componentes (ruído) antes da comparação

Esses limiares ficam disponíveis como sliders na UI para você equilibrar cobertura x precisão dependendo do cenário.

## Estrutura do projeto

```
├─ app.py                # Interface Gradio e orquestração
├─ mvp_sam/
│  ├─ __init__.py
│  ├─ sam_change_detector.py  # Carrega SAM, segmenta e compara máscaras
│  └─ visualization.py        # Overlays e mapa de mudanças
└─ requirements.txt
```

## Extensões sugeridas

1. **SAM 2 com tracking**: formar um “pseudo-vídeo” (antes → depois) e usar o tracker para objetos persistentes vs. transitórios.
2. **Métricas adicionais**: combinar IoU com similaridade de contorno ou descritores shape-context para reduzir falsos positivos.
3. **Pós-processamento geoespacial**: unir máscaras vizinhas, aplicar buffers e exportar para shapefile/GeoJSON.

## Limitações conhecidas

- Dependemos de segmentações consistentes; sombras ou iluminação podem quebrar o matching.
- A etapa de matching é `O(n²)` com o número de máscaras; convém ajustar o `min_area` para áreas grandes.
- O modelo SAM base requer ~4 GB de RAM durante inferência — feche outros apps pesados ao usar no M4.

## Próximos passos

- Adicionar persistência das máscaras para inspeção manual (JSON/GeoJSON).
- Explorar integrações com SAM 2 ou métodos híbridos (SAM + UNet leve) para mudanças sutis.
- Criar testes automatizados com imagens sintéticas para validar o comparador.

Bom proveito! Qualquer ajuste específico (ex.: outro backend de UI, integração com pipeline geoespacial) é só pedir.
