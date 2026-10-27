import streamlit as st
import json

from io import BytesIO
from pypdf import PdfReader
from openai import OpenAI


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Interviewer",
    page_icon="🎙️",
    layout="centered"
)


# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)


# =========================================================
# SESSION STATE
# =========================================================

if "skills" not in st.session_state:
    st.session_state.skills = []

if "show_skill_input" not in st.session_state:
    st.session_state.show_skill_input = False

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if "cv_profile" not in st.session_state:
    st.session_state.cv_profile = None

if "job_title" not in st.session_state:
    st.session_state.job_title = ""

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "current_skill_index" not in st.session_state:
    st.session_state.current_skill_index = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "transcript" not in st.session_state:
    st.session_state.transcript = []

if "question_number" not in st.session_state:
    st.session_state.question_number = 1


# =========================================================
# READ CV
# =========================================================

def read_pdf(uploaded_file):

    reader = PdfReader(
        BytesIO(uploaded_file.getvalue())
    )

    text = ""

    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    return text


# =========================================================
# ANALYSE CV
# =========================================================

def extract_cv_profile(cv_text, skills):

    skills_text = ", ".join(skills)

    prompt = f"""
Analyse the following candidate CV in preparation for a
professional job interview.

JOB SKILLS TO BE INTERVIEWED:

{skills_text}

Extract the following:

1. Candidate name
2. Education
3. Employment history
4. Projects
5. Evidence relating to each interview skill

For every interview skill, identify specific claims,
jobs, projects, technologies, responsibilities or
achievements from the CV that could reasonably be
explored during the interview.

Return valid JSON using exactly this general structure:

{{
    "candidate_name": "",
    "education": [],
    "employment_history": [],
    "projects": [],
    "skill_evidence": {{
        "skill name": []
    }}
}}

CANDIDATE CV:

{cv_text}
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )

    result = response.output_text

    result = result.replace("```json", "")
    result = result.replace("```", "")
    result = result.strip()

    return json.loads(result)


# =========================================================
# BUILD INTERVIEW CONTEXT
# =========================================================

def build_interview_context(skill):

    skill_evidence = (
        st.session_state.cv_profile
        .get("skill_evidence", {})
        .get(skill, [])
    )

    return f"""
You are conducting a professional structured job interview.

JOB TITLE:

{st.session_state.job_title}

JOB DESCRIPTION:

{st.session_state.job_description}

CURRENT SKILL BEING INTERVIEWED:

{skill}

RELEVANT EVIDENCE FROM THE CANDIDATE CV:

{json.dumps(skill_evidence, indent=2)}

Interview the candidate specifically about this skill.

Use information from the candidate's CV where relevant.

Explore areas such as:

- what the candidate actually did
- their individual contribution
- their technical decisions
- why they chose a particular approach
- problems encountered
- how they solved those problems
- results achieved
- lessons learned
- what they would change
- conceptual understanding of the skill

Ask ONE question at a time.

Never ask several questions in one response.

The interview should feel natural and conversational.

Use follow-up questions when an answer contains something
worth exploring more deeply.

Do not tell the candidate whether an answer is good or bad.

Do not provide feedback.

Only ask the next interview question.
"""


# =========================================================
# GENERATE QUESTION
# =========================================================

def generate_question(skill):

    interview_context = build_interview_context(skill)

    conversation = [
        {
            "role": "system",
            "content": interview_context
        }
    ]

    conversation.extend(
        st.session_state.messages
    )

    response = client.responses.create(
        model="gpt-5.4",
        input=conversation
    )

    return response.output_text.strip()


# =========================================================
# TRANSCRIBE CANDIDATE AUDIO
# =========================================================

def transcribe_audio(audio_file):

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=(
            "candidate_answer.wav",
            audio_file.getvalue(),
            "audio/wav"
        )
    )

    return transcription.text


# =========================================================
# GENERATE INTERVIEWER SPEECH
# =========================================================

def generate_speech(text):

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="cedar",
        input=text,
        instructions=(
            "Speak like a friendly professional interviewer. "
            "Sound natural, calm and conversational. "
            "Do not sound like you are reading from a script."
        )
    )

    return response.content


# =========================================================
# SETUP SCREEN
# =========================================================

def show_setup_screen():

    st.title("🎙️ AI Interviewer")

    st.write(
        "Configure the interview, upload the candidate's CV "
        "and add the skills you want to explore."
    )

    st.divider()

    # JOB TITLE

    job_title = st.text_input(
        "Job Title",
        placeholder="e.g. AI Analyst"
    )

    # JOB DESCRIPTION

    job_description = st.text_area(
        "Job Description",
        placeholder="Paste the complete job description here...",
        height=250
    )

    # CV

    uploaded_cv = st.file_uploader(
        "Candidate CV",
        type=["pdf"],
        help="Upload the candidate's CV as a PDF."
    )

    st.divider()

    # SKILLS

    st.subheader("Interview Skills")

    # Existing skills

    for index, skill in enumerate(
        st.session_state.skills
    ):

        col1, col2 = st.columns([8, 1])

        with col1:

            st.markdown(
                f"**{skill}**"
            )

        with col2:

            if st.button(
                "✕",
                key=f"delete_skill_{index}"
            ):

                st.session_state.skills.pop(index)

                st.rerun()

    # Add Skill control

    col1, col2 = st.columns([4, 1])

    with col1:

        st.markdown(
            "### Add Skill"
        )

    with col2:

        if st.button(
            "➕",
            use_container_width=True
        ):

            st.session_state.show_skill_input = True

    # Skill input

    if st.session_state.show_skill_input:

        with st.form(
            "add_skill_form",
            clear_on_submit=True
        ):

            new_skill = st.text_input(
                "Skill",
                placeholder="e.g. Python"
            )

            submitted = st.form_submit_button(
                "Add Skill",
                type="primary",
                use_container_width=True
            )

            if submitted:

                new_skill = new_skill.strip()

                if not new_skill:

                    st.warning(
                        "Please enter a skill."
                    )

                else:

                    existing_skills = [
                        skill.lower()
                        for skill in st.session_state.skills
                    ]

                    if (
                        new_skill.lower()
                        not in existing_skills
                    ):

                        st.session_state.skills.append(
                            new_skill
                        )

                    st.session_state.show_skill_input = False

                    st.rerun()

    st.divider()

    # START INTERVIEW

    if st.button(
        "Start Interview",
        type="primary",
        use_container_width=True
    ):

        if not job_title.strip():

            st.error(
                "Please enter a job title."
            )

            return

        if not job_description.strip():

            st.error(
                "Please enter a job description."
            )

            return

        if uploaded_cv is None:

            st.error(
                "Please upload the candidate CV."
            )

            return

        if not st.session_state.skills:

            st.error(
                "Please add at least one interview skill."
            )

            return

        st.session_state.job_title = (
            job_title.strip()
        )

        st.session_state.job_description = (
            job_description.strip()
        )

        st.session_state.cv_text = read_pdf(
            uploaded_cv
        )

        with st.spinner(
            "Analysing candidate CV..."
        ):

            try:

                st.session_state.cv_profile = (
                    extract_cv_profile(
                        st.session_state.cv_text,
                        st.session_state.skills
                    )
                )

            except Exception as error:

                st.error(
                    f"CV analysis failed: {error}"
                )

                return

        st.session_state.interview_started = True
        st.session_state.current_skill_index = 0
        st.session_state.messages = []
        st.session_state.transcript = []
        st.session_state.question_number = 1
        st.session_state.current_question = None

        st.rerun()


# =========================================================
# INTERVIEW SCREEN
# =========================================================

def show_interview_screen():

    st.title("🎙️ AI Interview")

    # Check whether all skills have been completed

    if (
        st.session_state.current_skill_index
        >= len(st.session_state.skills)
    ):

        show_finished_screen()

        return

    current_skill = st.session_state.skills[
        st.session_state.current_skill_index
    ]

    candidate_name = (
        st.session_state.cv_profile
        .get(
            "candidate_name",
            "Candidate"
        )
    )

    st.caption(
        f"Candidate: {candidate_name} | "
        f"Position: {st.session_state.job_title}"
    )

    st.divider()

    st.subheader(current_skill)

    st.caption(
        f"Interview area "
        f"{st.session_state.current_skill_index + 1} "
        f"of {len(st.session_state.skills)}"
    )

    # Generate question

    if st.session_state.current_question is None:

        with st.spinner(
            "Preparing the next question..."
        ):

            try:

                st.session_state.current_question = (
                    generate_question(
                        current_skill
                    )
                )

            except Exception as error:

                st.error(
                    f"Question generation failed: {error}"
                )

                return

    question = (
        st.session_state.current_question
    )

    # Show interviewer question

    st.markdown(
        "### Interviewer"
    )

    st.write(
        question
    )

    # Speak the question

    try:

        speech = generate_speech(
            question
        )

        st.audio(
            speech,
            format="audio/mpeg",
            autoplay=True
        )

    except Exception as error:

        st.warning(
            f"Could not generate speech: {error}"
        )

    st.divider()

    # Candidate answer

    st.markdown(
        "### Your Answer"
    )

    st.write(
        "Click the microphone below, answer the question, "
        "then stop the recording."
    )

    audio = st.audio_input(
        "Record your answer",
        sample_rate=16000,
        key=(
            f"answer_"
            f"{st.session_state.question_number}"
        )
    )

    # Submit recording

    if audio is not None:

        if st.button(
            "Submit Answer",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Transcribing your answer..."
            ):

                try:

                    answer = transcribe_audio(
                        audio
                    )

                except Exception as error:

                    st.error(
                        f"Transcription failed: {error}"
                    )

                    return

            # Save to transcript

            st.session_state.transcript.append(
                {
                    "skill": current_skill,
                    "question_number":
                        st.session_state.question_number,
                    "question": question,
                    "answer": answer
                }
            )

            # Add conversation history

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": question
                }
            )

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": answer
                }
            )

            st.session_state.question_number += 1

            # Force generation of next question

            st.session_state.current_question = None

            st.rerun()

    st.divider()

    # Move onto next skill manually

    if st.button(
        f"Finish {current_skill} and move to next skill",
        use_container_width=True
    ):

        st.session_state.current_skill_index += 1

        st.session_state.messages = []

        st.session_state.current_question = None

        st.rerun()


# =========================================================
# FINISHED SCREEN
# =========================================================

def show_finished_screen():

    st.title(
        "✅ Interview Complete"
    )

    st.success(
        "The interview has been completed."
    )

    st.divider()

    st.subheader(
        "Interview Transcript"
    )

    transcript_text = ""

    for item in st.session_state.transcript:

        section = f"""
SKILL: {item["skill"]}

QUESTION {item["question_number"]}

INTERVIEWER:
{item["question"]}

CANDIDATE:
{item["answer"]}

{"=" * 60}

"""

        transcript_text += section

        st.markdown(
            f"### {item['skill']} "
            f"— Question "
            f"{item['question_number']}"
        )

        st.markdown(
            "**Interviewer**"
        )

        st.write(
            item["question"]
        )

        st.markdown(
            "**Candidate**"
        )

        st.write(
            item["answer"]
        )

        st.divider()

    # TXT DOWNLOAD

    st.download_button(
        "Download Transcript",
        data=transcript_text,
        file_name="interview_transcript.txt",
        mime="text/plain",
        use_container_width=True
    )

    # JSON DOWNLOAD

    json_data = json.dumps(
        st.session_state.transcript,
        indent=2
    )

    st.download_button(
        "Download Structured JSON",
        data=json_data,
        file_name="interview_results.json",
        mime="application/json",
        use_container_width=True
    )

    # NEW INTERVIEW

    if st.button(
        "Start New Interview",
        use_container_width=True
    ):

        for key in list(
            st.session_state.keys()
        ):

            del st.session_state[key]

        st.rerun()


# =========================================================
# MAIN APP
# =========================================================

if st.session_state.interview_started:

    show_interview_screen()

else:

    show_setup_screen()