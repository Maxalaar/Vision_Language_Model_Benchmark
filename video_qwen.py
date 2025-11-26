import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

if __name__ == "__main__":
    # Chargement automatique sur GPU/CPU
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    # Chemin de la vidéo (local ou URL)
    video_path = "data/video.mp4"

    # Message pour le modèle : "video" au lieu de "image"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path
                },
                {"type": "text", "text": "Décris ce qu'il se passe dans cette vidéo."},
            ],
        }
    ]

    # Préparation + gère automatiquement l'extraction de frames
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
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

    print("\n Réponse du modèle :\n")
    print(output_text[0])
