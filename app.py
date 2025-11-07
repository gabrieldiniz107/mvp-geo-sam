from __future__ import annotations

import os
import gradio as gr

from mvp_sam import SAMChangeDetector
from mvp_sam.visualization import overlay_masks, render_change_map

SAM_CHECKPOINT_PATH = os.environ.get("SAM_CHECKPOINT_PATH", "checkpoints/sam_vit_b_01ec64.pth")
_detector: SAMChangeDetector | None = None


def _get_detector() -> SAMChangeDetector:
    global _detector
    if _detector is None:
        try:
            _detector = SAMChangeDetector(
                checkpoint_path=SAM_CHECKPOINT_PATH,
                model_type="vit_b",
            )
        except FileNotFoundError as exc:
            raise gr.Error(str(exc)) from exc
    return _detector


def _build_summary(result: dict) -> str:
    lines = [
        f"🟢 Objetos novos: **{len(result['new'])}**",
        f"🔴 Objetos removidos: **{len(result['removed'])}**",
        f"🟡 Objetos modificados: **{len(result['modified'])}**",
        f"⚪️ Objetos estáveis: **{len(result['unchanged'])}**",
    ]

    if result["modified"]:
        lines.append("\n### Principais mudanças detectadas")
        for idx, entry in enumerate(result["modified"][:5], start=1):
            lines.append(
                f"{idx}. IoU={entry['iou']:.2f} | Δárea={entry['area_delta']:.2f} | Δcentróide={entry['centroid_shift']:.2f}"
            )
    else:
        lines.append("\nNenhuma modificação significativa dentro dos limiares atuais.")
    return "\n".join(lines)


def run_change_detection(
    image_before,
    image_after,
    match_iou: float,
    area_delta: float,
    centroid_delta: float,
    min_area: int,
):
    if image_before is None or image_after is None:
        raise gr.Error("Envie as duas imagens (antes e depois) para continuar.")

    detector = _get_detector()
    detector.match_iou_threshold = match_iou
    detector.modification_area_threshold = area_delta
    detector.modification_shift_threshold = centroid_delta

    result = detector.detect_changes(
        image_before,
        image_after,
        min_area=int(min_area),
    )

    before_overlay = overlay_masks(result["before_image"], result["masks_before"])
    after_overlay = overlay_masks(result["after_image"], result["masks_after"])
    change_map = render_change_map(
        result["before_image"].shape[:2],
        result["new"],
        result["removed"],
        result["modified"],
        result["unchanged"],
    )

    summary = _build_summary(result)
    return before_overlay, after_overlay, change_map, summary


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="SAM Change Detection") as demo:
        gr.Markdown(
            """
            ## SAM Change Detection
            Pipeline: SAM ➜ Máscaras ➜ Matching ➜ Mapa de Mudanças. Ajuste os limiares para equilibrar cobertura e precisão.
            """
        )

        with gr.Row():
            before_input = gr.Image(type="numpy", label="Imagem Antes")
            after_input = gr.Image(type="numpy", label="Imagem Depois")

        with gr.Row():
            match_iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="Limiar IoU para matching")
            area_delta = gr.Slider(0.05, 0.5, value=0.15, step=0.05, label="Δ Área mínima (fração)")
            centroid_delta = gr.Slider(0.0, 0.2, value=0.05, step=0.01, label="Δ centróide mínima (normalizada)")
            min_area = gr.Slider(64, 5000, value=256, step=64, label="Área mínima do objeto (px)")

        run_button = gr.Button("Detectar mudanças", variant="primary")

        with gr.Row():
            before_out = gr.Image(label="Antes + Máscaras", show_label=True)
            after_out = gr.Image(label="Depois + Máscaras", show_label=True)
            change_out = gr.Image(label="Mapa de Mudanças", show_label=True)
        summary = gr.Markdown()

        run_button.click(
            fn=run_change_detection,
            inputs=[before_input, after_input, match_iou, area_delta, centroid_delta, min_area],
            outputs=[before_out, after_out, change_out, summary],
        )
    return demo


def main() -> None:
    demo = build_interface()
    demo.queue().launch()


if __name__ == "__main__":
    main()
