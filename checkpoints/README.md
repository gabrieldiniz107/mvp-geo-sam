# Checkpoints

Coloque aqui os pesos do SAM (ex.: `sam_vit_b_01ec64.pth`).

O app procura automaticamente por um arquivo `.pth` dentro desta pasta. Se houver mais de um, ele usará o que corresponde ao nome padrão `sam_vit_b_01ec64.pth`; caso contrário, escolhe o primeiro `.pth` encontrado. Você também pode forçar um caminho específico exportando a variável `SAM_CHECKPOINT_PATH`, se quiser.

Link do modelo no kaggle: https://www.kaggle.com/datasets/sacuscreed/sam-vit-b-01ec64-pth?resource=download
