import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLVideoProcessor
from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLProcessor


if __name__ == "__main__":
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="auto",
        device_map="auto"
    )

    processor: Qwen3VLProcessor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    video_processor: Qwen3VLVideoProcessor = processor.video_processor

    # Deux vidéos
    video1 = "data/chat_1.mp4"
    video2 = "data/chat_2.mp4"

    # Messages avec deux vidéos
    messages = [
        {
            "role": "user",
            "content": [
                { "type": "video", "video": video1 },
                { "type": "video", "video": video2 },
                {
                    "type": "text",
                    "text": "Compare these two videos and describe what happens in each one."
                }
            ],
        }
    ]

    # Préparation
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        load_audio_from_video=False,
        num_frames=8,
        fps=None,
    ).to(model.device)

    # Génération
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print("\nModel's response:\n")
    print(output_text[0])
