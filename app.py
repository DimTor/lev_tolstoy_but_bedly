import streamlit as st
import torch
import torch.nn.functional as F
import json
import os
from pathlib import Path


from model import TolstoyGPT

st.set_page_config(
    page_title="Генератор в стиле Толстого",
    page_icon="📚",
    layout="wide"
)

st.title("Генератор текста в стиле Льва Толстого")
st.markdown("Нейросеть, обученная на произведениях великого русского писателя")

@st.cache_resource
def load_model_and_vocab():
    model_path = Path('./checkpoints/tolstoy_final.pt')
    vocab_path = Path('./output/vocab.json')
    config_path = Path('./checkpoints/model_config.json')



    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    stoi = vocab['stoi']
    itos = {int(k): v for k, v in vocab['itos'].items()}
    vocab_size = len(stoi)

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        d_model = config.get('d_model', 128)
        num_heads = config.get('num_heads', 4)
        num_layers = config.get('num_layers', 3)
        d_ff = config.get('d_ff', 512)
        dropout = config.get('dropout', 0.1)
    else:
        d_model = 128
        num_heads = 4
        num_layers = 3
        d_ff = 512
        dropout = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TolstoyGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, stoi, itos, device


def generate_text(model, stoi, itos, device, prompt,
                  max_new_tokens=2000, temperature=0.8, top_k=40, seq_len=128):
    """Генерирует текст на основе промпта"""

    context = torch.tensor([stoi.get(ch, 0) for ch in prompt],
                           dtype=torch.long, device=device).unsqueeze(0)

    generated = list(prompt)

    with torch.no_grad():
        for _ in range(max_new_tokens):

            ctx = context[:, -seq_len:]
            seq_len_ctx = ctx.size(1)
            mask = torch.triu(torch.ones(seq_len_ctx, seq_len_ctx, device=device), diagonal=1).bool()
            mask = ~mask

            logits = model(ctx, mask)
            logits = logits[0, -1, :] / temperature

            if top_k is not None and top_k < logits.size(-1):
                top_probs, top_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits[top_indices] = top_probs

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            char = itos[next_token]
            generated.append(char)

            context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)
    return ''.join(generated)


with st.spinner("Загрузка модели Льва Николаевича..."):
    model, stoi, itos, device = load_model_and_vocab()

if model is None:
    st.stop()

with st.sidebar:
    st.header("⚙️ Настройки генерации")
    max_length = st.slider(
        "Максимальная длина (символов)",
        min_value=50,
        max_value=500,
        value=200,
        step=25,
        help="Максимальное количество генерируемых символов"
    )

    st.divider()

st.subheader("Введите начало текста")


examples = ["Однажды", "Душа моя", "Война", "Анна", "Счастье", "Человек есть"]

cols = st.columns(3)
for i, example in enumerate(examples):
    if cols[i % 3].button(example, key=f"ex_{i}"):
        st.session_state.prompt = example

prompt = st.text_area(
    "Начните писать...",
    value=st.session_state.get("prompt", "Однажды"),
    height=100,
    placeholder="Введите начало фразы..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button("Сгенерировать в стиле Толстого", type="primary", use_container_width=True)


if generate_button and prompt.strip():
    with st.spinner("Лев Николаевич размышляет..."):
        try:
            result = generate_text(
                model=model,
                stoi=stoi,
                itos=itos,
                device=device,
                prompt=prompt,
                max_new_tokens=max_length,
                seq_len=128
            )

            st.divider()
            st.subheader("Результат:")
            st.markdown(f"> {result}")

        except Exception as e:
            st.error(f"Ошибка при генерации: {e}")
            st.exception(e)

elif generate_button:
    st.warning("Пожалуйста, введите начало текста")