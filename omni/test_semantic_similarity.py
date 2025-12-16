from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F
from qwen_omni_utils import process_mm_info
from transformers import AutoModel, AutoProcessor


def embed_video(model, processor, video: Path):
    with torch.no_grad():
        documents = [
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "text",
                    #     "text": "passage: This is a passage to be embedded"
                    # },
                    {
                        "type": "video",
                        "video": str(video)
                    },
                    # {
                    #     "type": "audio",
                    #     "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
                    # }
                ]
            },
        ]

        documents_texts = processor.apply_chat_template(documents, add_generation_prompt=False, tokenize=False)
        audio, images, videos = process_mm_info(documents, use_audio_in_video=False)

        videos_kwargs = {
            "min_pixels": 32 * 14 * 14,
            "max_pixels": 64 * 28 * 28,
            "use_audio_in_video": False,
        }
        text_kwargs = {
            "truncation": True,
            "padding": True,
            "max_length": 204800,
        }
        batch_dict = processor(
            text=documents_texts,
            images=images,
            videos=videos,
            audio=audio,
            return_tensors="pt",
            text_kwargs=text_kwargs,
            videos_kwargs=videos_kwargs,
            audio_kwargs={"max_length": 2048000},
        )

        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
        last_hidden_states = model(**batch_dict, output_hidden_states=True).hidden_states[-1]
        # Average Pooling
        attention_mask = batch_dict["attention_mask"]
        last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embedding = F.normalize(embedding, dim=-1)
        return embedding

def embed_texts(model, processor, texts: List[str]):
    with torch.no_grad():
        text_kwargs = {
            "truncation": True,
            "padding": True,
            "max_length": 204800,
        }

        messages = []

        for text in texts:
            messages.append([{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                ]
            }])

        inputs = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        batch_dict = processor(
            text=inputs,
            return_tensors="pt",
            text_kwargs=text_kwargs,
        )

        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
        last_hidden_states = model(**batch_dict, output_hidden_states=True).hidden_states[-1]
        # Average Pooling
        attention_mask = batch_dict["attention_mask"]
        last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embedding = F.normalize(embedding, dim=-1)

        return list(embedding)



if __name__ == "__main__":
    model_name_or_path = "nvidia/omni-embed-nemotron-3b"
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    model = model.to("cuda:0")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    texts = [
        "rat is eating cheese",
        "cat is crying",
        "cat",
        "human",
        "cat and human",
        "rainbow is shining",
        "dog",
        "A small gray kitten is standing on its hind legs in a glass enclosure, looking up at the camera with wide eyes and a happy expression. The kitten is holding its front paws up, as if it is dancing or performing a dance move. The text \"free him!!!\" appears on the screen, indicating that the kitten is being held captive. The kitten's body is slightly tilted forward, and it appears to be moving its paws in a rhythmic motion. The enclosure is made of glass, and there are food and water bowls on the floor. The kitten's fur is soft and fluffy, and it has a playful and curious expression on its face. The video captures the kitten's adorable and energetic behavior, as it appears to be enjoying the attention and interaction with the camera.",
        "A person is holding a small gray kitten in their hands. The kitten is resting and scratching its face. It seems very happy and plays with the person’s hands. The kitten starts moving and fidgeting. The person continues to pet and comfort it. The kitten appears to be very happy and content.",
    ]

    video_path = Path('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Vision_Language_Model_Benchmark/data/chat_1.mp4')

    texts_embeddings = embed_texts(model, processor, texts)
    video_embedding = embed_video(model, processor, video_path).squeeze(0)


    # Stack text embeddings if they are a list
    if isinstance(texts_embeddings, list):
        texts_embeddings = torch.stack(texts_embeddings)  # shape: (num_texts, embedding_dim)

    # Normalize embeddings for cosine similarity
    texts_embeddings = F.normalize(texts_embeddings, dim=1)
    video_embedding = F.normalize(video_embedding, dim=0)

    # Compute cosine similarity
    cos_sim = torch.matmul(texts_embeddings, video_embedding)  # shape: (num_texts,)

    for text, score in zip(texts, cos_sim):
        print(f"{float(score):.2f}: {text}")

    # print(dict(zip(cos_sim, texts)))
    # Find the index of the most similar text
    best_idx = torch.argmax(cos_sim)

    # Return the closest text
    print()
    closest_text = texts[best_idx]
    print("Closest text:", closest_text)