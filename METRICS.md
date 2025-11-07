# Guia Rápido de Métricas e Conceitos

Documento curto para explicar, em linguagem simples, os termos que aparecem no app de Change Detection com SAM.

## IoU (Intersection over Union)
- **O que é:** mede o quanto duas máscaras se sobrepõem.
- **Como ler:** varia de 0 a 1. IoU = 1 significa que os objetos antes e depois são idênticos; IoU próximo de 0 indica que quase não há interseção.
- **Uso no projeto:** é a métrica primária para decidir se duas máscaras representam o mesmo objeto.

## Peso Hausdorff
- **O que é:** convertemos a distância de Hausdorff entre duas máscaras em um valor de similaridade (1 = formas iguais, 0 = formas muito diferentes).
- **Por que usar:** o IoU cai muito quando o objeto muda de forma, mesmo que ainda seja o mesmo alvo. A Hausdorff observa o contorno, sendo mais tolerante a deformações.
- **Slider “Peso Hausdorff”:** define o equilíbrio entre IoU e Hausdorff no score final (`score = w1 * IoU + w2 * Hausdorff`).

## Δ Área (Delta de Área)
- **O que é:** diferença relativa entre as áreas das máscaras antes/depois.
- **Como ler:** 0.15 significa que a área mudou 15%. Valores altos indicam crescimento ou redução significativa do objeto. Controlamos o limiar com o slider “Δ Área mínima”.

## Δ Centròide (Delta de Centròide)
- **O que é:** deslocamento do centro geométrico do objeto, normalizado pelo tamanho da imagem (varia de 0 a 1).
- **Uso:** se o objeto “andou” ou foi movido, o delta sobe. O slider “Δ centróide mínima” ajusta o limite para considerar que houve mudança.

## Histogram Matching
- **O que é:** ajuste automático para igualar o histograma de cores da imagem “depois” com o da imagem “antes”.
- **Por que importa:** diferenças de iluminação ou sensores podem gerar falsos positivos. Equalizando os histogramas, o SAM recebe imagens com distribuição de cor mais parecida.
- **Controle:** checkbox “Histogram matching T1→T0”.

## Grid Fixo de Prompts
- **O que é:** usamos o mesmo grid regular de pontos (por exemplo 32×32) como prompts para o SAM nas duas imagens.
- **Benefício:** força o modelo a olhar exatamente nos mesmos locais em T0 e T1, deixando as máscaras mais alinhadas.
- **Controle:** slider “Grid fixo de prompts (pontos/linha)”. Valores maiores = mais pontos = mais máscaras (mas também mais custo computacional).

## Score Final do Matching
- **Fórmula:** `score_final = w1 * IoU + w2 * Hausdorff`, onde `w1 + w2 = 1`.
- **Critério:** um par só é aceito se passar pelo limiar de IoU **ou** pelo limiar do `score_final`. Assim evitamos perder matches bons quando o IoU ou a Hausdorff ficam baixos isoladamente.

## Objetos Novos / Removidos / Modificados
- **Novos:** máscaras detectadas apenas na imagem “depois”.
- **Removidos:** máscaras que só aparecem na imagem “antes”.
- **Modificados:** pares de máscaras aceitos pelo matching, mas que ultrapassam o limiar de Δ área ou Δ centróide.

## Resumo Visual
- **Verde:** objetos novos.
- **Vermelho:** objetos removidos.
- **Amarelo:** objetos modificados.
- **Cinza:** nada mudou dentro dos limiares definidos.

> Dica: se o resultado estiver muito ruidoso, experimente aumentar `min_area`, reduzir o peso Hausdorff ou diminuir o grid de prompts. Para o contrário (captar detalhes), faça o oposto.