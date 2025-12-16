import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLVideoProcessor
from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLProcessor


if __name__ == "__main__":
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        dtype="auto",
        device_map="auto"
    )

    processor: Qwen3VLProcessor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    video_processor: Qwen3VLVideoProcessor = processor.video_processor

    # Deux vidéos
    # video1 = "data/chat_1.mp4"
    # video2 = "data/chat_2.mp4"
    # Please describe Video 1 in a way that clearly distinguishes it from Video 2.

    game_description: str = "Surround is a two-player game where each player controls a vehicle that leaves a solid trail behind it. The goal is to force the opponent to collide with trail or the edges of the screen. The last remaining player wins the round."
    # instruction: str = "Here is a description of game: " + game_description + ". The two videos show different phases of same the game, please describe videos, clearly highlighting the differences between them."
    # instruction: str = "The two videos show different phases of the game. Compare these two videos and describe what happens in each one."
    # instruction: str = "The two videos show different phases of same the game. Compare these two phases and describe what happens in each one."
    instruction: str = (
        "Game description: " + game_description +
        "You are given two videos showing different phases of the same game. "
        "For each video, provide a short, clear description and briefly mention how it differs from the other. "
        "Use the following format: 'Video 1: ...' and 'Video 2: ...'."
    )

    video1 = "data/video_1.mp4"
    video2 = "data/video_2.mp4"


    # Messages avec deux vidéos
    messages = [
        {
            "role": "user",
            "content": [
                { "type": "video", "video": video1 },
                { "type": "video", "video": video2 },
                {
                    "type": "text",
                    # "text": "Compare these two videos and describe what happens in each one."
                    "text": instruction,
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
        num_frames=16,
        fps=None,
    ).to(model.device)

    # Génération
    generated_ids = model.generate(**inputs, max_new_tokens=512)

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
