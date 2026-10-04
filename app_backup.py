import streamlit as st


import requests


import os
import base64
from io import BytesIO


import random


import urllib.parse


from pathlib import Path


# Optional packages used by RAG / image upload


try:


    from pypdf import PdfReader


except Exception:


    PdfReader = None


try:


    import chromadb


    from sentence_transformers import SentenceTransformer


except Exception:


    chromadb = None


    SentenceTransformer = None


try:


    from PIL import Image


except Exception:


    Image = None


# ============================================================


# DEV LENS AI


# AI-Powered Developer & Creative Assistant


# ============================================================


st.set_page_config(


    page_title="DevLens AI",


    page_icon="🧠",


    layout="wide",


    initial_sidebar_state="expanded",


)


OLLAMA_URL = "http://localhost:11434/api/generate"


MODEL = "llama3.2:3b"


VISION_MODEL = "llava"


APP_DIR = Path(__file__).parent


DATA_DIR = APP_DIR / "devlens_data"


DATA_DIR.mkdir(exist_ok=True)


CHROMA_DIR = DATA_DIR / "chroma_db"


# ============================================================


# BASIC AI FUNCTION


# ============================================================


def ask_ollama(prompt, model=MODEL):


    try:


        response = requests.post(


            OLLAMA_URL,


            json={


                "model": model,


                "prompt": prompt,


                "stream": False,


                "options": {


                    "temperature": 0.3


                },


            },


            timeout=180,


        )


        if response.status_code != 200:


            return f"❌ Ollama error: HTTP {response.status_code}"


        data = response.json()


        return data.get("response", "❌ No response received.")


    except requests.exceptions.ConnectionError:


        return (


            "❌ Cannot connect to Ollama.\n\n"


            "Ollama is probably not running. "


            "If Ollama is already running in the background, "


            "do not start another `ollama serve` process."


        )


    except requests.exceptions.Timeout:


        return "❌ Ollama took too long to respond."


    except Exception as e:


        return f"❌ Unexpected error: {e}"


def ollama_model_exists(model_name):


    try:


        r = requests.get("http://localhost:11434/api/tags", timeout=10)


        if r.status_code != 200:


            return False


        names = [m.get("name", "") for m in r.json().get("models", [])]


        return model_name in names


    except Exception:


        return False


# ============================================================


# SESSION STATE


# ============================================================


defaults = {


    "chat_history": [],


    "interview_started": False,


    "interview_question": "",


    "interview_feedback": "",


    "question_number": 1,


    "uploaded_text": "",


    "rag_ready": False,


}


for key, value in defaults.items():


    if key not in st.session_state:


        st.session_state[key] = value


# ============================================================


# PERSONALITIES


# ============================================================


PERSONALITIES = {


    "👨‍💻 Developer": """


You are an expert software developer.


Give technically accurate, practical answers.


Use clean code and real-world examples.


""",


    "🎓 Beginner Tutor": """


You are a beginner-friendly teacher.


Explain difficult concepts using simple words,


small examples and step-by-step explanations.


""",


    "🤝 Friendly Assistant": """


You are a friendly general AI assistant.


Answer the actual question clearly and helpfully.


""",


    "💼 Professional Mentor": """


You are a professional technology mentor.


Give industry-oriented, interview-focused and practical advice.


""",


}


# ============================================================


# INTERVIEW TOPICS


# ============================================================


INTERVIEW_TOPICS = [


    "Python",


    "Java",


    "C",


    "C++",


    "JavaScript",


    "HTML",


    "CSS",


    "JavaScript & DOM",


    "Frontend Development",


    "React.js",


    "Backend Development",


    "Full Stack Development",


    "REST APIs",


    "SQL",


    "MySQL",


    "DBMS",


    "MongoDB",


    "Data Structures & Algorithms",


    "Object-Oriented Programming",


    "Operating Systems",


    "Computer Networks",


    "Computer Organization & Architecture",


    "Discrete Mathematics",


    "Artificial Intelligence",


    "Machine Learning",


    "Generative AI",


    "Large Language Models",


    "Prompt Engineering",


    "RAG",


    "Vector Databases",


    "Computer Vision",


    "Git & GitHub",


    "Django",


    "Flask",


    "Streamlit",


    "APIs & JSON",


    "Docker",


    "Cloud Computing",


    "Full Stack Project Interview",


    "Project Viva",


    "Resume-Based Interview",


    "HR Interview",


    "Aptitude",


    "Coding Interview",


    "🔥 Mixed Full-Stack Interview",


]


# ============================================================


# RAG HELPERS


# ============================================================


@st.cache_resource


def get_embedding_model():


    if SentenceTransformer is None:


        return None


    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource


def get_chroma_collection():


    if chromadb is None:


        return None


    client = chromadb.PersistentClient(path=str(CHROMA_DIR))


    return client.get_or_create_collection(


        name="devlens_knowledge"


    )


def split_text(text, chunk_size=900, overlap=150):


    text = text.replace("\x00", " ").strip()


    if not text:


        return []


    chunks = []


    start = 0


    while start < len(text):


        end = start + chunk_size


        chunk = text[start:end]


        if chunk.strip():


            chunks.append(chunk.strip())


        if end >= len(text):


            break


        start = end - overlap


    return chunks


def read_uploaded_file(uploaded_file):


    if uploaded_file is None:


        return ""


    suffix = Path(uploaded_file.name).suffix.lower()


    try:


        if suffix == ".pdf":


            if PdfReader is None:


                return ""


            reader = PdfReader(uploaded_file)


            pages = []


            for page in reader.pages:


                pages.append(page.extract_text() or "")


            return "\n".join(pages)


        if suffix in [".txt", ".md", ".py", ".java", ".c", ".cpp", ".html", ".css", ".js", ".json"]:


            return uploaded_file.getvalue().decode(


                "utf-8",


                errors="ignore"


            )


        return ""


    except Exception:


        return ""


def add_document_to_rag(text, source_name):


    collection = get_chroma_collection()


    embedder = get_embedding_model()


    if collection is None or embedder is None:


        return False, "RAG packages are not available."


    chunks = split_text(text)


    if not chunks:


        return False, "No readable text found."


    embeddings = embedder.encode(


        chunks,


        normalize_embeddings=True


    ).tolist()


    ids = [


        f"{source_name}_{i}_{random.randint(1000, 9999)}"


        for i in range(len(chunks))


    ]


    collection.add(


        ids=ids,


        documents=chunks,


        embeddings=embeddings,


        metadatas=[


            {"source": source_name}


            for _ in chunks


        ],


    )


    return True, f"Added {len(chunks)} knowledge chunks."


def search_rag(question, n_results=4):


    collection = get_chroma_collection()


    embedder = get_embedding_model()


    if collection is None or embedder is None:


        return []


    if collection.count() == 0:


        return []


    query_embedding = embedder.encode(


        [question],


        normalize_embeddings=True


    ).tolist()


    result = collection.query(


        query_embeddings=query_embedding,


        n_results=min(n_results, collection.count()),


    )


    documents = result.get("documents", [[]])[0]


    metadatas = result.get("metadatas", [[]])[0]


    return [


        {


            "text": doc,


            "source": meta.get("source", "Unknown")


        }


        for doc, meta in zip(documents, metadatas)


    ]


# ============================================================


# SIDEBAR


# ============================================================


with st.sidebar:


    st.markdown("# 🧠 DevLens AI")


    st.caption("AI-Powered Developer & Creative Assistant")


    st.divider()


    st.subheader("🎭 AI Personality")


    personality = st.selectbox(


        "Choose AI Personality",


        list(PERSONALITIES.keys()),


    )


    st.divider()


    st.subheader("🛠️ DevLens Modules")


    module = st.selectbox(


        "Choose a module",


        [


            "💬 AI Chat",


            "🐛 Error Explanation",


            "💻 Code Explanation",


            "📚 Project Document Q&A",


            "🖼️ Image Recognition",


            "🎨 AI Image Studio",


            "🎯 AI Interview Mode",


            "ℹ️ About Project",


        ],


    )


    st.divider()


    st.success(


        "Local AI: Ollama\n\n"


        "Web UI: Streamlit\n\n"


        "RAG: ChromaDB + Embeddings"


    )


# ============================================================


# HEADER


# ============================================================


st.title("🧠 DevLens AI")


st.subheader("AI-Powered Developer & Creative Assistant")


st.markdown(


    f"🎭 **Current AI Personality:** {personality}"


)


st.markdown("---")


# ============================================================


# 1. GENERAL AI CHAT


# ============================================================


if module == "💬 AI Chat":


    st.header("💬 AI Developer Chatbot")


    st.write(


        "Ask DevLens AI about programming, technology, "


        "projects, studies, career or general questions."


    )


    for message in st.session_state.chat_history:


        with st.chat_message(message["role"]):


            st.markdown(message["content"])


    user_input = st.chat_input(


        "Ask DevLens AI anything..."


    )


    if user_input:


        st.session_state.chat_history.append(


            {


                "role": "user",


                "content": user_input,


            }


        )


        prompt = f"""


{PERSONALITIES[personality]}


You are DevLens AI, a helpful general-purpose AI assistant.


You can answer questions about programming, AI, education,


career, projects, technology and general everyday topics.


Do NOT unnecessarily reject a question because it is not


about programming.


Give:


1\\. A direct answer


2\\. A simple explanation


3\\. Examples when useful


4\\. Code when required


5\\. Step-by-step instructions when appropriate


User question:


{user_input}


"""


        answer = ask_ollama(prompt)


        st.session_state.chat_history.append(


            {


                "role": "assistant",


                "content": answer,


            }


        )


        st.rerun()


# ============================================================


# 2. ERROR EXPLANATION


# ============================================================


elif module == "🐛 Error Explanation":


    st.header("🐛 Error Explanation")


    error_text = st.text_area(


        "Paste your error",


        height=220,


        placeholder=(


            "Example:\n"


            "ModuleNotFoundError: No module named 'streamlit'"


        ),


    )


    if st.button(


        "🔍 Analyze Error",


        use_container_width=True,


    ):


        if not error_text.strip():


            st.warning("Please paste an error first.")


        else:


            prompt = f"""


You are an expert programming debugger.


Analyze this error:


{error_text}


Give:


# 🔴 What is the error?


# 🔎 Why did it happen?


# 🛠️ How to fix it?


# ✅ Correct solution/code


# 💡 Beginner tip


Use simple language.


"""


            st.markdown(ask_ollama(prompt))


# ============================================================


# 3. CODE EXPLANATION


# ============================================================


elif module == "💻 Code Explanation":


    st.header("💻 Code Explanation")


    code = st.text_area(


        "Paste your code",


        height=330,


        placeholder="Paste Python, Java, C, C++, JavaScript or other code...",


    )


    if st.button(


        "🤖 Analyze Code",


        use_container_width=True,


    ):


        if not code.strip():


            st.warning("Please paste code first.")


        else:


            prompt = f"""


You are an expert programming teacher and code reviewer.


Analyze this code:


```text


{code}


```


Give:


# 📌 What does the code do?


# 🔍 Step-by-step explanation


# 🧠 Important concepts


# ⚠️ Errors or possible issues


# 🚀 Improvements


# ✅ Improved code


Explain it for a B.Tech student.


"""


            st.markdown(ask_ollama(prompt))


# ============================================================


# 4. RAG / PROJECT DOCUMENT Q&A


# ============================================================


elif module == "📚 Project Document Q&A":


    st.header("📚 Project Document Q&A")


    st.write(


        "Upload a PDF/text/code document. "


        "DevLens AI stores its knowledge in ChromaDB "


        "and answers questions using relevant chunks."


    )


    uploaded = st.file_uploader(


        "Upload your project document",


        type=[


            "pdf",


            "txt",


            "md",


            "py",


            "java",


            "c",


            "cpp",


            "html",


            "css",


            "js",


            "json",


        ],


    )


    if uploaded is not None:


        if st.button(


            "📥 Add Document to Knowledge Base",


            use_container_width=True,


        ):


            text = read_uploaded_file(uploaded)


            if not text:


                st.error(


                    "Could not extract readable text from this file."


                )


            else:


                with st.spinner(


                    "Creating embeddings and storing document..."


                ):


                    ok, message = add_document_to_rag(


                        text,


                        uploaded.name,


                    )


                if ok:


                    st.success(message)


                else:


                    st.error(message)


    st.divider()


    question = st.text_input(


        "Ask a question about your uploaded project",


        placeholder="Example: What database does this project use?",


    )


    if st.button(


        "🔎 Search Knowledge Base",


        use_container_width=True,


    ):


        if not question.strip():


            st.warning("Enter a question first.")


        else:


            with st.spinner("Searching ChromaDB..."):


                results = search_rag(question)


            if not results:


                st.info(


                    "No knowledge found. Upload and add a document first."


                )


            else:


                context = "\n\n".join(


                    [


                        f"Source: {item['source']}\n{item['text']}"


                        for item in results


                    ]


                )


                prompt = f"""


You are DevLens AI's project-document assistant.


Answer the user's question using ONLY the supplied document


context.


If the context does not contain the answer, clearly say that


the information was not found in the uploaded document.


Document context:


{context}


User question:


{question}


Give a clear answer and mention the relevant source name.


"""


                answer = ask_ollama(prompt)


                st.subheader("🤖 Answer")


                st.markdown(answer)


                with st.expander("📚 Retrieved Knowledge"):


                    for item in results:


                        st.markdown(


                            f"**Source:** {item['source']}"


                        )


                        st.write(item["text"])


# ============================================================


# 5. IMAGE RECOGNITION


# ============================================================


elif module == "🖼️ Image Recognition":


    st.header("🖼️ AI Image Recognition")


    st.write(


        "Upload an image and DevLens AI can analyze it using "


        "a local vision model."


    )


    image_file = st.file_uploader(


        "Upload an image",


        type=["png", "jpg", "jpeg", "webp"],


    )


    if image_file is not None:


        if Image is not None:


            image = Image.open(image_file)


            st.image(


                image,


                caption="Uploaded Image",


                use_container_width=True,


            )


        st.info(


            "This feature requires an Ollama vision model such as "


            "LLaVA."


        )


        if st.button(


            "🔍 Analyze Image",


            use_container_width=True,


        ):


            if not ollama_model_exists(VISION_MODEL):


                st.warning(


                    "Vision model is not installed yet."


                )


                st.code(


                    "ollama pull llava",


                    language="powershell",


                )


                st.write(


                    "After installing it, restart the app and "


                    "try the image again."


                )


            else:

                try:
                    image_bytes = image_file.getvalue()
                    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

                    vision_prompt = st.text_input(
                        "What do you want to know about this image?",
                        value="Describe this image clearly and identify important details.",
                    )

                    vision_response = requests.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": VISION_MODEL,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": vision_prompt,
                                    "images": [encoded_image],
                                }
                            ],
                            "stream": False,
                        },
                        timeout=180,
                    )

                    if vision_response.status_code == 200:
                        data = vision_response.json()
                        answer = data.get("message", {}).get(
                            "content",
                            "No description received."
                        )
                        st.subheader("🤖 Image Analysis")
                        st.markdown(answer)
                    else:
                        st.error(
                            f"Vision model error: HTTP {vision_response.status_code}"
                        )
                        st.code(vision_response.text)

                except requests.exceptions.Timeout:
                    st.error("⏱️ Image analysis timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to Ollama. Make sure Ollama is running."
                    )
                except Exception as e:
                    st.error(f"❌ Image analysis error: {e}")



# ============================================================


# ============================================================

# ============================================================
# 6. AI IMAGE STUDIO
# ============================================================

elif module == "🎨 AI Image Studio":

    st.header("🎨 AI Image Studio")
    st.write("Describe what you want to create.")

    image_idea = st.text_area(
        "Describe your image",
        placeholder="Example: A futuristic AI developer working in a college lab at night",
        height=120,
    )

    if st.button("✨ Create Image", use_container_width=True):
        if not image_idea.strip():
            st.warning("Please describe the image first.")
        else:
            hf_token = os.getenv("HF_TOKEN")

            if not hf_token:
                st.error("Hugging Face token is not available.")
                st.info("In PowerShell, set HF_TOKEN and restart Streamlit.")
            else:
                try:
                    from huggingface_hub import InferenceClient

                    with st.spinner("🎨 Creating your image..."):
                        client = InferenceClient(
                            provider="auto",
                            api_key=hf_token,
                        )

                        image = client.text_to_image(
                            image_idea.strip(),
                            model="black-forest-labs/FLUX.1-schnell",
                        )

                    st.success("✅ Image created successfully!")
                    st.image(image, caption="Generated by DevLens AI", use_container_width=True)

                    image_buffer = BytesIO()
                    image.save(image_buffer, format="PNG")

                    st.download_button(
                        "⬇️ Download Image",
                        data=image_buffer.getvalue(),
                        file_name="devlens_generated_image.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error("❌ Image generation failed.")
                    st.code(str(e))


# 7. AI INTERVIEW MODE

elif module == "🎯 AI Interview Mode":


    st.header("🎯 AI Interview Mode")


    st.write(


        "Practice real technical interviews with AI evaluation."


    )


    interview_topic = st.selectbox(


        "Choose interview topic",


        INTERVIEW_TOPICS,


    )


    difficulty = st.selectbox(


        "Choose difficulty",


        [


            "Beginner",


            "Intermediate",


            "Advanced",


        ],


    )


    if st.button(


        "🚀 Start Interview",


        use_container_width=True,


    ):


        if interview_topic == "🔥 Mixed Full-Stack Interview":


            topic_instruction = """


Ask from a mixture of Python, Java, C, C++, JavaScript,


HTML, CSS, React, Full Stack Development, SQL, DBMS,


DSA, OOP, Operating Systems, Computer Networks, AI,


Machine Learning, Generative AI, Git/GitHub, REST APIs


and project development.


"""


        else:


            topic_instruction = f"""


Ask specifically about:


{interview_topic}


"""


        prompt = f"""


You are a professional technical interviewer.


Difficulty:


{difficulty}


{topic_instruction}


Generate ONE interview question.


Rules:


- Ask only one question.


- Do not give the answer.


- Do not give hints.


- Make it realistic for a technical interview.


- Return only the question.


"""


        question = ask_ollama(prompt)


        st.session_state.interview_started = True


        st.session_state.interview_question = question


        st.session_state.interview_feedback = ""


        st.session_state.question_number = 1


        st.session_state.interview_topic = interview_topic


        st.session_state.interview_difficulty = difficulty


        st.rerun()


    if st.session_state.interview_started:


        st.markdown("---")


        st.subheader(


            f"📝 Question {st.session_state.question_number}"


        )


        st.info(


            st.session_state.interview_question


        )


        answer = st.text_area(


            "✍️ Your Answer",


            height=220,


            placeholder="Type your interview answer here...",


        )


        if st.button(


            "✅ Submit Answer",


            use_container_width=True,


        ):


            if not answer.strip():


                st.warning(


                    "Please type your answer before submitting."


                )


            else:


                evaluation_prompt = f"""


You are an experienced technical interviewer.


Topic:


{st.session_state.interview_topic}


Difficulty:


{st.session_state.interview_difficulty}


Question:


{st.session_state.interview_question}


Candidate answer:


{answer}


Evaluate the answer.


Give:


# ⭐ Score


Score out of 10.


# ✅ What You Did Well


Correct points.


# ❌ What Needs Improvement


Mistakes or missing points.


# 💡 Better Answer


Give an ideal interview answer.


# 🎯 Interview Tip


Give one useful interview tip.


Be honest, professional and encouraging.


"""


                st.session_state.interview_feedback = ask_ollama(


                    evaluation_prompt


                )


        if st.session_state.interview_feedback:


            st.markdown("---")


            st.subheader("🤖 AI Interview Evaluation")


            st.markdown(


                st.session_state.interview_feedback


            )


            if st.button(


                "➡️ Next Question",


                use_container_width=True,


            ):


                if (


                    st.session_state.interview_topic


                    == "🔥 Mixed Full-Stack Interview"


                ):


                    mixed_topics = [


                        "Python",


                        "Java",


                        "C",


                        "C++",


                        "JavaScript",


                        "HTML",


                        "CSS",


                        "React.js",


                        "Full Stack Development",


                        "SQL",


                        "DBMS",


                        "DSA",


                        "OOP",


                        "Operating Systems",


                        "Computer Networks",


                        "Artificial Intelligence",


                        "Machine Learning",


                        "Generative AI",


                        "Git & GitHub",


                        "REST APIs",


                        "Project Development",


                    ]


                    next_topic = random.choice(


                        mixed_topics


                    )


                    topic_instruction = (


                        f"Ask the next question about {next_topic}."


                    )


                else:


                    topic_instruction = (


                        f"Ask the next question about "


                        f"{st.session_state.interview_topic}."


                    )


                next_prompt = f"""


You are conducting a professional technical interview.


Difficulty:


{st.session_state.interview_difficulty}


{topic_instruction}


Previous question:


{st.session_state.interview_question}


Generate ONE new question.


Rules:


- Do not repeat the previous question.


- Ask only one question.


- Do not give the answer.


- Return only the question.


"""


                st.session_state.interview_question = ask_ollama(


                    next_prompt


                )


                st.session_state.interview_feedback = ""


                st.session_state.question_number += 1


                st.rerun()


# ============================================================


# 8. ABOUT PROJECT


# ============================================================


elif module == "ℹ️ About Project":


    st.header("🚀 About DevLens AI")


    st.markdown(


        """


# 🧠 DevLens AI


### AI-Powered Developer & Creative Assistant


DevLens AI is a multi-feature AI platform designed for


students and developers.


### 🔥 Main Features


- 💬 AI Developer Chatbot


- 🌐 General-purpose AI questions


- 🐛 Error Explanation


- 💻 Code Explanation


- 📚 Project Document Q&A


- 🔎 RAG Pipeline


- 🗄️ ChromaDB Knowledge Base


- 🧠 Local LLM using Ollama


- 🎭 AI Personality Modes


- 🎯 AI Interview Mode


- 📝 Interview Answer Evaluation


- ⭐ Interview Scoring


- 🖼️ Image Recognition module


- 🎨 AI Image Studio


- 📄 PDF/Text/Code document support


- 🔥 Mixed Full-Stack Interview


- 🧩 Project-specific knowledge


### 🛠️ Technologies


**Python**  


**Streamlit**  


**Ollama**  


**Llama 3.2**  


**ChromaDB**  


**Sentence Transformers**  


**PyPDF**  


**REST APIs**


### 🎯 Purpose


DevLens AI brings multiple developer and learning tools


into one application instead of requiring students to use


many separate AI tools.


"""


    )


# ============================================================


# FOOTER


# ============================================================


st.markdown("---")


st.caption(


    "🧠 DevLens AI | AI Chat + Developer Tools + RAG + "


    "Interview Mode + Creative AI | Powered by Ollama, "


    "Streamlit & ChromaDB"


)
