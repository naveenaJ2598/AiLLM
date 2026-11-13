
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# 1. MODEL CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 50

SYSTEM_PROMPT = (
    "You are a helpful AI Student Learning Assistant. "
    "Explain Python, Machine Learning, LLM, and technical "
    "concepts in simple English. Give one easy example. "
    "Keep answers short, clear, and beginner-friendly. "
    "If you are unsure, say so."
)

# --------------------------------------------------
# 2. STREAMLIT PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="AI Student Learning Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Student Learning Assistant")

st.write(
    "Ask questions about Python, Machine Learning, "
    "LLM, and other technical topics."
)

# --------------------------------------------------
# 3. LOAD MODEL ONCE AND CACHE IT
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model


try:
    with st.spinner("Loading AI model..."):
        tokenizer, model = load_model()

    st.success("AI model is ready!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# --------------------------------------------------
# 4. INITIALIZE CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

# --------------------------------------------------
# 5. DISPLAY PREVIOUS MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# 6. USER QUESTION INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask your question here..."
)

# --------------------------------------------------
# 7. GENERATE OPTIMIZED RESPONSE
# --------------------------------------------------

if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("assistant"):

        with st.spinner("AI is thinking..."):

            try:
                # Keep only recent conversation messages
                recent_messages = [
                    msg
                    for msg in st.session_state.messages
                    if msg["role"] != "system"
                ][-6:]

                prompt_messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    *recent_messages
                ]

                # Convert messages into model input
                inputs = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True
                )

                # Generate a shorter response
                with torch.inference_mode():

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Extract only newly generated tokens
                input_length = inputs["input_ids"].shape[-1]

                new_tokens = outputs[0][input_length:]

                # Decode generated answer
                answer = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True
                ).strip()

                if not answer:
                    answer = (
                        "I couldn't generate an answer. "
                        "Please try asking in a different way."
                    )

                st.markdown(answer)

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                st.error(
                    f"Response generation failed: {e}"
                )

# --------------------------------------------------
# 8. CLEAR CONVERSATION
# --------------------------------------------------

if st.button("Clear Conversation"):

    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    st.rerun()
