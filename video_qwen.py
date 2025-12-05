import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLVideoProcessor
from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLProcessor

if __name__ == "__main__":
    # Chargement automatique sur GPU/CPU
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="auto",
        # dtype=torch.float16,
        device_map="auto"
    )

    processor: Qwen3VLProcessor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    video_processor: Qwen3VLVideoProcessor = processor.video_processor
    # video_processor.size = {
    #     "longest_edge": 256 * 32 * 32,
    #     "shortest_edge": 256 * 32 * 32,
    # }

    # Chemin de la vidéo (local ou URL)
    video_path = "data/chat_1.mp4"

    # Message pour le modèle : "video" au lieu de "image"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": "Describe what is happening in this video."},
            ],
        }
    ]

    # Préparation + gère automatiquement l'extraction de frames
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        load_audio_from_video=False,
        num_frames=16,
        fps=None,
    ).to(model.device)

    # Génération de la réponse
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    # Suppression des tokens d'entrée
    generated_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # Décodage du texte
    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print("\n Model's response :\n")
    print(output_text[0])
