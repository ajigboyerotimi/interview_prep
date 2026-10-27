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
# MODEL SETTINGS
# =========================================================

# You can change these later without changing the rest
# of the application.

INTERVIEW_MODEL = "gpt-5.4"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "cedar"


# =========================================================
# SESSION STATE
# =========================================================

def initialise_session_state():

    defaults = {
        "skills": [],
        "show_skill_input": False,
        "interview_started": False,
        "cv_text": "",
        "cv_profile": None,
        "job_title": "",
        "job_description": "",
        "current_skill_index": 0,
        "messages": [],
        "current_question": None,
        "current_question_audio": None,
        "transcript": [],
        "question_number": 1
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()


# =========================================================
# PDF READER
# =========================================================

def read_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF CV.
    """

    reader = PdfReader(
        BytesIO(uploaded_file.getvalue())
    )

    pages = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


# =========================================================
# CV ANALYSIS
# =========================================================

def extract_cv_profile(cv_text, skills):
    """
    Ask the model to analyse the candidate's CV and identify
    evidence relating to the skills that will be interviewed.
    """

    skills_text = "\n".join(
        f"- {skill}"
        for skill in skills
    )

    prompt = f"""
You are analysing a candidate CV before conducting
a professional job interview.

The candidate is applying for:

JOB TITLE:
{st.session_state.job_title}

JOB DESCRIPTION:
{st.session_state.job_description}

The interview will specifically cover these skills:

{skills_text}

Analyse the candidate CV.

Extract:

1. The candidate's name.
2. Education.
3. Employment history.
4. Relevant projects.
5. Evidence relating to each interview skill.

For each skill, identify specific claims in the CV that
could reasonably be explored during the interview.

Relevant evidence can include:

- jobs
- responsibilities
- projects
- technologies
- achievements
- qualifications
- technical claims
- leadership claims
- tools used
- outcomes delivered

Do not invent experience that does not appear in the CV.

Return ONLY valid JSON.

Use exactly this general structure:

{{
    "candidate_name": "",
    "education": [],
    "employment_history": [],
    "projects": [],
    "skill_evidence": {{
        "Python": [],
        "SQL": []
    }}
}}

The skill_evidence keys should correspond to the actual skills
provided above.

CANDIDATE CV:

{cv_text}
"""

    response = client.responses.create(
        model=INTERVIEW_MODEL,
        input=prompt
    )

    result = response.output_text.strip()

    # Remove Markdown fences if the model includes them.

    if result.startswith("```json"):
        result = result[7:]

    elif result.startswith("```"):
        result = result[3:]

    if result.endswith("```"):
        result = result[:-3]

    result = result.strip()

    return json.loads(result)


# =========================================================
# GET CV EVIDENCE FOR CURRENT SKILL
# =========================================================

def get_skill_evidence(skill):
    """
    Retrieve the CV evidence associated with the current skill.
    """

    if not st.session_state.cv_profile:
        return []

    evidence = (
        st.session_state.cv_profile
        .get("skill_evidence", {})
    )

    # First try exact match.

    if skill in evidence:
        return evidence[skill]

    # Then try case-insensitive matching.

    for key, value in evidence.items():

        if key.lower().strip() == skill.lower().strip():
            return value

    return []


# =========================================================
# INTERVIEW CONTEXT
# =========================================================

def build_interview_context(skill):
    """
    Build the system instructions for the interview.
    """

    skill_evidence = get_skill_evidence(skill)

    candidate_name = (
        st.session_state.cv_profile
        .get("candidate_name", "Candidate")
    )

    return f"""
You are conducting a professional structured job interview.

CANDIDATE:
{candidate_name}

JOB TITLE:
{st.session_state.job_title}

JOB DESCRIPTION:
{st.session_state.job_description}

CURRENT INTERVIEW SKILL:
{skill}

RELEVANT EVIDENCE FROM THE CANDIDATE'S CV:
{json.dumps(skill_evidence, indent=2)}

Your job is to interview the candidate specifically about:

{skill}

Use relevant information from the CV where appropriate.

The purpose of the interview is to understand the candidate's
genuine experience, knowledge and competence.

Explore areas such as:

- what the candidate actually did
- the candidate's individual contribution
- technical choices
- reasoning behind their approach
- problems encountered
- how problems were solved
- results achieved
- lessons learned
- what they would change
- technical understanding
- conceptual understanding
- practical application of the skill

Where the candidate's CV contains relevant experience, use it
to make questions specific to the candidate.

Where appropriate, ask follow-up questions based on what the
candidate has just said.

IMPORTANT INTERVIEW RULES:

1. Ask ONE question at a time.

2. Never ask several questions bundled together.

3. Keep questions concise and conversational.

4. Do not provide the candidate with the answer.

5. Do not tell the candidate whether an answer was correct,
   incorrect, strong or weak.

6. Do not score the candidate during the interview.

7. Do not repeatedly ask essentially the same question.

8. Use previous answers to make follow-up questions feel
   natural.

9. Stay focused on the current skill.

10. Behave like a professional human interviewer rather than
    a quiz application.

Your response must contain ONLY the next thing the interviewer
should say to the candidate.

Do not include labels such as "Interviewer:".

Do not explain your reasoning.

Do not include quotation marks around the question.
"""


# =========================================================
# GENERATE QUESTION
# =========================================================

def generate_question(skill):
    """
    Generate the next interview question using the
    conversation so far.
    """

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
        model=INTERVIEW_MODEL,
        input=conversation
    )

    return response.output_text.strip()


# =========================================================
# TEXT TO SPEECH
# =========================================================

def generate_speech(text):
    """
    Convert the interviewer's question into speech.
    """

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions=(
            "Speak like a friendly, experienced professional "
            "interviewer. Use a natural conversational pace. "
            "Sound calm, engaged and human. Avoid sounding "
            "like you are reading from a script."
        )
    )

    return response.content


# =========================================================
# TRANSCRIBE CANDIDATE AUDIO
# =========================================================

def transcribe_audio(audio_file):
    """
    Transcribe an answer recorded using the candidate's
    browser microphone.
    """

    transcription = client.audio.transcriptions.create(
        model=TRANSCRIPTION_MODEL,
        file=(
            "candidate_answer.wav",
            audio_file.getvalue(),
            "audio/wav"
        )
    )

    return transcription.text.strip()


# =========================================================
# SETUP SCREEN
# =========================================================

def show_setup_screen():

    st.title("🎙️ AI Interviewer")

    st.write(
        "Set up the interview, upload the candidate's CV "
        "and add the skills you want to explore."
    )

    st.divider()

    # -----------------------------------------------------
    # JOB TITLE
    # -----------------------------------------------------

    job_title = st.text_input(
        "Job Title",
        value=st.session_state.job_title,
        placeholder="e.g. AI Analyst"
    )

    # -----------------------------------------------------
    # JOB DESCRIPTION
    # -----------------------------------------------------

    job_description = st.text_area(
        "Job Description",
        value=st.session_state.job_description,
        placeholder=(
            "Paste the complete job description here..."
        ),
        height=250
    )

    # -----------------------------------------------------
    # CV
    # -----------------------------------------------------

    uploaded_cv = st.file_uploader(
        "Candidate CV",
        type=["pdf"],
        help="Upload the candidate's CV as a PDF."
    )

    st.divider()

    # =====================================================
    # SKILLS
    # =====================================================

    st.subheader("Interview Skills")

    st.caption(
        "Add each skill you want the AI interviewer "
        "to explore."
    )

    # -----------------------------------------------------
    # DISPLAY EXISTING SKILLS
    # -----------------------------------------------------

    for index, skill in enumerate(
        st.session_state.skills
    ):

        col_skill, col_delete = st.columns(
            [8, 1]
        )

        with col_skill:

            st.markdown(
                f"**{skill}**"
            )

        with col_delete:

            if st.button(
                "✕",
                key=f"delete_skill_{index}",
                help=f"Remove {skill}",
                use_container_width=True
            ):

                st.session_state.skills.pop(
                    index
                )

                st.rerun()

    # -----------------------------------------------------
    # ADD SKILL BUTTON
    # -----------------------------------------------------

    col_label, col_button = st.columns(
        [4, 1]
    )

    with col_label:

        st.markdown(
            "### Add Skill"
        )

    with col_button:

        if st.button(
            "➕",
            key="open_add_skill",
            use_container_width=True,
            help="Add another interview skill"
        ):

            st.session_state.show_skill_input = True

    # -----------------------------------------------------
    # NEW SKILL FORM
    # -----------------------------------------------------

    if st.session_state.show_skill_input:

        with st.form(
            "add_skill_form",
            clear_on_submit=True
        ):

            new_skill = st.text_input(
                "Skill",
                placeholder=(
                    "e.g. Python, SQL, Machine Learning..."
                )
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
                        skill.lower().strip()
                        for skill
                        in st.session_state.skills
                    ]

                    if (
                        new_skill.lower()
                        not in existing_skills
                    ):

                        st.session_state.skills.append(
                            new_skill
                        )

                    else:

                        st.warning(
                            "That skill has already "
                            "been added."
                        )

                    st.session_state.show_skill_input = False

                    st.rerun()

    st.divider()

    # =====================================================
    # START INTERVIEW
    # =====================================================

    if st.button(
        "Start Interview",
        type="primary",
        use_container_width=True
    ):

        # ---------------------------------------------
        # VALIDATION
        # ---------------------------------------------

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
                "Please upload the candidate's CV."
            )

            return

        if not st.session_state.skills:

            st.error(
                "Please add at least one interview skill."
            )

            return

        # ---------------------------------------------
        # SAVE SETUP
        # ---------------------------------------------

        st.session_state.job_title = (
            job_title.strip()
        )

        st.session_state.job_description = (
            job_description.strip()
        )

        # ---------------------------------------------
        # READ CV
        # ---------------------------------------------

        try:

            st.session_state.cv_text = (
                read_pdf(uploaded_cv)
            )

        except Exception as error:

            st.error(
                f"Could not read the CV: {error}"
            )

            return

        if not st.session_state.cv_text.strip():

            st.error(
                "I could not extract any text from "
                "this PDF. The CV may be an image-only "
                "or scanned PDF."
            )

            return

        # ---------------------------------------------
        # ANALYSE CV
        # ---------------------------------------------

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

            except json.JSONDecodeError:

                st.error(
                    "The CV analysis returned an invalid "
                    "response. Please try again."
                )

                return

            except Exception as error:

                st.error(
                    f"CV analysis failed: {error}"
                )

                return

        # ---------------------------------------------
        # INITIALISE INTERVIEW
        # ---------------------------------------------

        st.session_state.interview_started = True

        st.session_state.current_skill_index = 0

        st.session_state.messages = []

        st.session_state.transcript = []

        st.session_state.question_number = 1

        st.session_state.current_question = None

        st.session_state.current_question_audio = None

        st.rerun()


# =========================================================
# INTERVIEW PROGRESS
# =========================================================

def show_interview_progress():

    completed = (
        st.session_state.current_skill_index
    )

    total = len(
        st.session_state.skills
    )

    if total > 0:

        progress = completed / total

        st.progress(progress)

        st.caption(
            f"Interview area "
            f"{completed + 1} of {total}"
        )


# =========================================================
# INTERVIEW HISTORY
# =========================================================

def show_current_skill_history(current_skill):
    """
    Show previous questions and answers from the
    current skill.
    """

    current_history = [
        item
        for item in st.session_state.transcript
        if item["skill"] == current_skill
    ]

    if not current_history:
        return

    with st.expander(
        "Previous questions and answers"
    ):

        for item in current_history:

            st.markdown(
                f"**Interviewer:** "
                f"{item['question']}"
            )

            st.markdown(
                f"**Candidate:** "
                f"{item['answer']}"
            )

            st.divider()


# =========================================================
# INTERVIEW SCREEN
# =========================================================

def show_interview_screen():

    # -----------------------------------------------------
    # CHECK WHETHER INTERVIEW IS COMPLETE
    # -----------------------------------------------------

    if (
        st.session_state.current_skill_index
        >= len(st.session_state.skills)
    ):

        show_finished_screen()

        return

    # -----------------------------------------------------
    # CURRENT SKILL
    # -----------------------------------------------------

    current_skill = (
        st.session_state.skills[
            st.session_state.current_skill_index
        ]
    )

    candidate_name = (
        st.session_state.cv_profile
        .get(
            "candidate_name",
            "Candidate"
        )
    )

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.title("🎙️ AI Interview")

    st.caption(
        f"Candidate: {candidate_name}"
    )

    st.caption(
        f"Position: "
        f"{st.session_state.job_title}"
    )

    st.divider()

    show_interview_progress()

    st.subheader(current_skill)

    show_current_skill_history(
        current_skill
    )

    # =====================================================
    # GENERATE QUESTION
    # =====================================================

    if st.session_state.current_question is None:

        with st.spinner(
            "Preparing the next question..."
        ):

            try:

                question = generate_question(
                    current_skill
                )

                st.session_state.current_question = (
                    question
                )

            except Exception as error:

                st.error(
                    f"Could not generate the "
                    f"question: {error}"
                )

                return

    question = (
        st.session_state.current_question
    )

    # =====================================================
    # QUESTION DISPLAY
    # =====================================================

    st.markdown("### Interviewer")

    st.write(question)

    # =====================================================
    # GENERATE SPEECH ONLY ONCE
    # =====================================================

    if (
        st.session_state.current_question_audio
        is None
    ):

        with st.spinner(
            "Preparing interviewer audio..."
        ):

            try:

                st.session_state.current_question_audio = (
                    generate_speech(question)
                )

            except Exception as error:

                st.warning(
                    "The question was generated, but "
                    "the interviewer audio could not "
                    f"be created: {error}"
                )

    # -----------------------------------------------------
    # PLAY QUESTION
    # -----------------------------------------------------

    if st.session_state.current_question_audio:

        st.audio(
            st.session_state.current_question_audio,
            format="audio/mpeg",
            autoplay=True
        )

    st.divider()

    # =====================================================
    # CANDIDATE ANSWER
    # =====================================================

    st.markdown("### Your Answer")

    st.write(
        "Record your answer below. You can listen "
        "back to your recording before submitting it."
    )

    audio = st.audio_input(
        "Record your answer",
        sample_rate=16000,
        key=(
            f"answer_"
            f"{st.session_state.question_number}"
        )
    )

    # =====================================================
    # SUBMIT ANSWER
    # =====================================================

    if audio is not None:

        if st.button(
            "Submit Answer",
            type="primary",
            use_container_width=True,
            key=(
                f"submit_answer_"
                f"{st.session_state.question_number}"
            )
        ):

            # -----------------------------------------
            # TRANSCRIPTION
            # -----------------------------------------

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

            if not answer:

                st.warning(
                    "No speech could be detected. "
                    "Please record your answer again."
                )

                return

            # -----------------------------------------
            # SAVE TRANSCRIPT
            # -----------------------------------------

            st.session_state.transcript.append(
                {
                    "skill": current_skill,
                    "question_number":
                        st.session_state.question_number,
                    "question": question,
                    "answer": answer
                }
            )

            # -----------------------------------------
            # SAVE AI CONVERSATION
            # -----------------------------------------

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

            # -----------------------------------------
            # PREPARE NEXT QUESTION
            # -----------------------------------------

            st.session_state.question_number += 1

            st.session_state.current_question = None

            st.session_state.current_question_audio = None

            st.rerun()

    st.divider()

    # =====================================================
    # MOVE TO NEXT SKILL
    # =====================================================

    if (
        st.session_state.current_skill_index
        < len(st.session_state.skills) - 1
    ):

        button_text = (
            f"Finish {current_skill} "
            f"and move to next skill"
        )

    else:

        button_text = (
            f"Finish {current_skill} "
            f"and complete interview"
        )

    if st.button(
        button_text,
        use_container_width=True
    ):

        st.session_state.current_skill_index += 1

        # Reset skill-specific conversation.
        st.session_state.messages = []

        st.session_state.current_question = None

        st.session_state.current_question_audio = None

        st.rerun()


# =========================================================
# BUILD READABLE TRANSCRIPT
# =========================================================

def build_transcript_text():

    candidate_name = (
        st.session_state.cv_profile
        .get(
            "candidate_name",
            "Candidate"
        )
    )

    transcript_text = f"""
AI INTERVIEW TRANSCRIPT
============================================================

Candidate: {candidate_name}
Job Title: {st.session_state.job_title}

Interview Skills:
{", ".join(st.session_state.skills)}

============================================================

"""

    for item in st.session_state.transcript:

        transcript_text += f"""
SKILL: {item["skill"]}
QUESTION {item["question_number"]}

INTERVIEWER:
{item["question"]}

CANDIDATE:
{item["answer"]}

============================================================

"""

    return transcript_text.strip()


# =========================================================
# FINISHED SCREEN
# =========================================================

def show_finished_screen():

    st.title("✅ Interview Complete")

    candidate_name = (
        st.session_state.cv_profile
        .get(
            "candidate_name",
            "Candidate"
        )
    )

    st.success(
        f"The interview with {candidate_name} "
        f"has been completed."
    )

    st.divider()

    # =====================================================
    # TRANSCRIPT
    # =====================================================

    st.subheader(
        "Interview Transcript"
    )

    if not st.session_state.transcript:

        st.info(
            "No interview answers were recorded."
        )

    else:

        previous_skill = None

        for item in st.session_state.transcript:

            if item["skill"] != previous_skill:

                st.markdown(
                    f"## {item['skill']}"
                )

                previous_skill = item["skill"]

            st.markdown(
                f"### Question "
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

    # =====================================================
    # DOWNLOADS
    # =====================================================

    transcript_text = (
        build_transcript_text()
    )

    st.download_button(
        "📄 Download Transcript",
        data=transcript_text,
        file_name=(
            "interview_transcript.txt"
        ),
        mime="text/plain",
        use_container_width=True
    )

    structured_data = {
        "candidate": candidate_name,
        "job_title":
            st.session_state.job_title,
        "job_description":
            st.session_state.job_description,
        "skills":
            st.session_state.skills,
        "cv_profile":
            st.session_state.cv_profile,
        "interview":
            st.session_state.transcript
    }

    json_data = json.dumps(
        structured_data,
        indent=2
    )

    st.download_button(
        "📊 Download Structured JSON",
        data=json_data,
        file_name=(
            "interview_results.json"
        ),
        mime="application/json",
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # NEW INTERVIEW
    # =====================================================

    if st.button(
        "Start New Interview",
        use_container_width=True
    ):

        st.session_state.clear()

        st.rerun()


# =========================================================
# MAIN APPLICATION
# =========================================================

if st.session_state.interview_started:

    show_interview_screen()

else:

    show_setup_screen()
