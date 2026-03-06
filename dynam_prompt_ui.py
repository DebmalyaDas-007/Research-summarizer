from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.header("Research Summarizer")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
        "AlphaFold: Highly Accurate Protein Structure Prediction with Deep Learning",
        "Neural Ordinary Differential Equations",
        "DALL-E: Creating Images from Text",
        "CLIP: Connecting Text and Images",
        "Reinforcement Learning with Human Feedback (RLHF)",
        "Transformers in Computer Vision: A Survey",
        "Graph Neural Networks: A Review of Methods and Applications",
        "Federated Learning: Challenges, Methods, and Future Directions",
        "Adversarial Attacks and Defenses in Deep Learning",
        "Meta-Learning: A Survey",
        "Self-Supervised Learning: An Overview",
        "Neural Architecture Search: A Survey",
        "Explainable AI: Interpreting Machine Learning Models",
        "AI Alignment: Ensuring Safe and Beneficial AI",
        "AI Ethics: Principles and Challenges",
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with:
"Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"]
)

if st.button("Summarize"):

    if paper_input != "Select...":

        st.write("Generating summary...")

        prompt = template.format(
            paper_input=paper_input,
            style_input=style_input,
            length_input=length_input
        )

        result = model.invoke(prompt)

        st.write(result.content)
        print(result.content)

    else:
        st.warning("Please select a research paper.")