import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


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

    documents = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
                },
            ]
        },
    ]

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    documents_texts = processor.apply_chat_template(documents, add_generation_prompt=False, tokenize=False)

    videos_kwargs = {
        "min_pixels": 32*14*14,
        "max_pixels": 64*28*28,
        "use_audio_in_video": False,
    }
    batch_dict = processor(
        text=documents_texts,
        videos=documents,
        return_tensors="pt",
        videos_kwargs=videos_kwargs,
    )

    batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
    last_hidden_states = model(**batch_dict, output_hidden_states=True).hidden_states[-1]
    # Average Pooling
    attention_mask = batch_dict["attention_mask"]
    last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    embedding = F.normalize(embedding, dim=-1)
    print(embedding)
    print(embedding.shape)