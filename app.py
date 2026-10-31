import base64
import json
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from pypdf import PdfReader


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

CHAT_MODEL = "gpt-5.4"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
TTS_MODEL = "gpt-4o-mini-tts-2025-12-15"
TTS_VOICE = "marin"

# Interview flow
# Questions asked per skill, decided by the skill's importance (1-5)
QUESTIONS_BY_IMPORTANCE = {
    1: 2,
    2: 3,
    3: 5,
    4: 6,
    5: 7,
}
PREP_COUNTDOWN_SECONDS = 30      # pause between question audio and recording
MAX_ANSWER_SECONDS = 180         # maximum recording length (3 minutes)

# Importance slider (1-5) labels and definitions
IMPORTANCE_LABELS = {
    1: "Nice to have",
    2: "Useful",
    3: "Important",
    4: "Very important",
    5: "Essential",
}
IMPORTANCE_MEANINGS = {
    1: "a bonus, but the role works fine without it",
    2: "helpful day to day, can be learned on the job",
    3: "expected at a competent level",
    4: "core to success in the role",
    5: "the role can't be done without it",
}
IMPORTANCE_MIN = 1
IMPORTANCE_MAX = 5
DEFAULT_IMPORTANCE = 3

# Score (0-100) -> competency level. Checked from the top down.
# These match the bands in the scoring guide given to the model.
LEVEL_THRESHOLDS = [
    (81, "Expert"),
    (61, "Advanced"),
    (41, "Intermediate"),
    (21, "Beginner"),
    (0, "Unqualified"),
]
LEVEL_COLOURS = {
    "Unqualified": "#dc2626",
    "Beginner": "#ea580c",
    "Intermediate": "#d97706",
    "Advanced": "#65a30d",
    "Expert": "#15803d",
}

# Makes the importance slider thicker and easier to see.
SLIDER_CSS = """
<style>
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child,
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div {
    height: 14px !important;
    border-radius: 7px !important;
}
div[data-testid="stSlider"] [role="slider"] {
    height: 30px !important;
    width: 30px !important;
    box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.35) !important;
}
div[data-testid="stSlider"] {
    padding-top: 0.5rem;
}
.imp-markers {
    position: relative;
    height: 2.8rem;
    margin: 0 0 0.5rem 0;
    font-size: 0.8rem;
    line-height: 1.2;
    opacity: 0.75;
}
.imp-marker {
    position: absolute;
    top: 0;
    text-align: center;
    transform: translateX(-50%);
    white-space: nowrap;
}
.imp-marker.first { transform: none; text-align: left; }
.imp-marker.last { transform: translateX(-100%); text-align: right; }
.imp-marker.active { font-weight: 700; opacity: 1; color: #ff4b4b; }
</style>
"""


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
# CUSTOM RECORDER COMPONENT
# =========================================================
# Plays the question audio, runs the preparation countdown, then
# records automatically. The front end lives in recorder_component/.

interview_recorder = components.declare_component(
    "interview_recorder",
    path=str(Path(__file__).parent / "recorder_component")
)


# =========================================================
# SESSION STATE
# =========================================================

def initialise_session_state():

    default_values = {
        "skills": [],              # list of {"name": str, "importance": int}
        "show_skill_input": False,
        "skill_form_message": "",
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
        "question_number": 1,
        "recording_attempt": 0,
        "interview_result": None,
        "answer_processing": False
    }

    for key, value in default_values.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()


# =========================================================
# SKILL HELPERS
# =========================================================

def skill_names():

    return [skill["name"] for skill in st.session_state.skills]


def max_questions_for(skill_name):
    """Number of questions to ask for a skill, based on its importance."""

    for skill in st.session_state.skills:

        if skill["name"] == skill_name:

            return QUESTIONS_BY_IMPORTANCE.get(
                skill["importance"],
                QUESTIONS_BY_IMPORTANCE[DEFAULT_IMPORTANCE]
            )

    return QUESTIONS_BY_IMPORTANCE[DEFAULT_IMPORTANCE]


def importance_shares():
    """Return each skill's share of the overall score, in percent."""

    total = sum(
        skill["importance"] for skill in st.session_state.skills
    )

    if total == 0:
        return [0 for _ in st.session_state.skills]

    return [
        round(skill["importance"] / total * 100)
        for skill in st.session_state.skills
    ]


def format_importance(value):

    return f"{value} - {IMPORTANCE_LABELS[value]}"


# =========================================================
# PDF READER
# =========================================================

def read_pdf(uploaded_file):

    reader = PdfReader(
        BytesIO(uploaded_file.getvalue())
    )

    extracted_pages = []

    for page in reader.pages:

        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(page_text)

    cv_text = "\n\n".join(extracted_pages)

    if not cv_text.strip():

        raise ValueError(
            "No readable text was found in the PDF. "
            "The CV may be a scanned image rather than a text PDF."
        )

    return cv_text


# =========================================================
# JSON CLEANING
# =========================================================

def clean_json_response(response_text):

    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]

    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]

    cleaned_text = cleaned_text.strip()

    return json.loads(cleaned_text)


# =========================================================
# CV ANALYSIS
# =========================================================

def extract_cv_profile(cv_text, skills):

    skills_text = ", ".join(skills)

    prompt = f"""
You are analysing a candidate CV before a structured job interview.

LANGUAGE REQUIREMENT:

Return the complete response in English only.
Do not switch to another language.

The interview will focus on the following skills:

{skills_text}

Extract:

1. The candidate's name
2. Education
3. Employment history
4. Projects
5. Specific evidence relating to each interview skill

For every interview skill, identify relevant claims, jobs,
projects, technologies, responsibilities or achievements from
the CV that could be explored during the interview.

Do not invent information.

If there is no evidence for a skill, return an empty list for
that skill.

Return ONLY valid JSON using this structure:

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
        model=CHAT_MODEL,
        input=prompt
    )

    return clean_json_response(
        response.output_text
    )


# =========================================================
# MATCH SKILL EVIDENCE
# =========================================================

def get_skill_evidence(skill):

    if not st.session_state.cv_profile:
        return []

    skill_evidence = st.session_state.cv_profile.get(
        "skill_evidence",
        {}
    )

    # Try an exact match first
    if skill in skill_evidence:
        return skill_evidence[skill]

    # Try a case-insensitive match
    for evidence_skill, evidence in skill_evidence.items():

        if evidence_skill.lower() == skill.lower():
            return evidence

    return []


# =========================================================
# INTERVIEW CONTEXT
# =========================================================

def build_interview_context(skill, question_in_skill):

    skill_evidence = get_skill_evidence(skill)

    max_questions = max_questions_for(skill)

    if question_in_skill >= max_questions:

        progress_instruction = (
            f"This is question {question_in_skill} of "
            f"{max_questions} for this skill, and it is the "
            "LAST one. Ask the single most valuable remaining question "
            "for judging the candidate's competence in this skill. "
            "Do not mention that it is the final question."
        )

    else:

        progress_instruction = (
            f"This is question {question_in_skill} of "
            f"{max_questions} for this skill. You have a "
            "limited number of questions, so cover the most important "
            "ground and avoid repeating topics already discussed."
        )

    return f"""
You are conducting a professional structured job interview.

LANGUAGE RULE:

The entire interview MUST be conducted in English.

Always ask every question in English.

Never switch to another language, even if:

- the CV contains non-English words
- a transcription contains foreign-language text
- the candidate uses a foreign-language phrase
- a previous message contains another language
- the candidate's answer appears to have been incorrectly transcribed

If a candidate answer appears to have been incorrectly transcribed
into another language, do not copy that language. Continue the
interview in English.


JOB TITLE:

{st.session_state.job_title}


JOB DESCRIPTION:

{st.session_state.job_description}


CURRENT INTERVIEW SKILL:

{skill}


RELEVANT CV EVIDENCE:

{json.dumps(skill_evidence, indent=2)}


QUESTION PROGRESS:

{progress_instruction}


INTERVIEW INSTRUCTIONS:

Interview the candidate specifically about the current skill.

Use CV evidence where it is relevant.

Explore:

- what the candidate personally did
- the candidate's individual contribution
- technical choices
- reasoning behind the approach
- problems encountered
- results achieved
- lessons learned
- what the candidate would change
- conceptual understanding of the skill

Ask exactly ONE question at a time.

Do not combine several questions into one message.

Questions should sound like a natural professional interview,
not a written examination. Candidates have about three minutes
to answer each question, so keep questions answerable in that time.

Where the candidate gives an interesting or incomplete answer,
ask a relevant follow-up question.

Do not tell the candidate whether an answer was good or bad.

Do not provide feedback, a score or an assessment during the
interview.

Only return the next interview question.

The question MUST be written entirely in English. Include punctation marks and indications of prosody.
"""


# =========================================================
# QUESTION GENERATION
# =========================================================

def generate_question(skill):

    question_in_skill = len(
        [
            item for item in st.session_state.transcript
            if item["skill"] == skill
        ]
    ) + 1

    interview_context = build_interview_context(
        skill,
        question_in_skill
    )

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
        model=CHAT_MODEL,
        input=conversation
    )

    question = response.output_text.strip()

    if not question:

        raise ValueError(
            "The interviewer did not generate a question."
        )

    return question


# =========================================================
# TEXT-TO-SPEECH
# =========================================================

def generate_speech(text):

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions=(
            "Speak entirely in English. "
            "Speak like a friendly and professional job interviewer. "
            "Use a calm, natural and conversational delivery. "
            "Do not sound robotic or as if reading from a script."
        )
    )

    return response.content


# =========================================================
# AUDIO TRANSCRIPTION
# =========================================================

def transcribe_audio(audio_bytes, mime_type):

    if not audio_bytes:

        raise ValueError(
            "The recording was empty."
        )

    # Browsers record in different formats (Chrome/Edge: webm,
    # Firefox: ogg, Safari: mp4). Give OpenAI a matching filename.
    mime_type = (mime_type or "").lower()

    if "mp4" in mime_type:
        extension = "mp4"
    elif "ogg" in mime_type:
        extension = "ogg"
    elif "wav" in mime_type:
        extension = "wav"
    else:
        extension = "webm"

    transcription_file = BytesIO(audio_bytes)
    transcription_file.name = f"candidate_answer.{extension}"

    transcription = client.audio.transcriptions.create(
        model=TRANSCRIPTION_MODEL,
        file=transcription_file,
        language="en"
    )

    transcript = transcription.text.strip()

    if not transcript:

        raise ValueError(
            "No speech could be identified in the recording."
        )

    return transcript


# =========================================================
# SCORING HELPERS
# =========================================================

def clamp_score(value):

    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0

    return max(0, min(100, score))


def level_for_score(score):

    for threshold, label in LEVEL_THRESHOLDS:

        if score >= threshold:
            return label

    return LEVEL_THRESHOLDS[-1][1]


def level_badge(level, large=False):
    """Return an HTML pill badge for a competency level."""

    colour = LEVEL_COLOURS.get(level, "#6b7280")

    padding = "0.4rem 1.2rem" if large else "0.15rem 0.7rem"
    font_size = "1.2rem" if large else "0.8rem"

    return (
        f'<span style="display:inline-block;background:{colour};'
        f"color:#fff;font-weight:700;border-radius:999px;"
        f"padding:{padding};font-size:{font_size};"
        f'letter-spacing:0.03em;">{level}</span>'
    )


def apply_weighted_scoring(result):
    """
    Attach importance and weight to each skill assessment, then
    compute the overall score (0-100) as an importance-weighted
    average of the skill scores. The overall score and the
    competency level are calculated here, not by the model, so the
    maths is consistent and reproducible.
    """

    skills = st.session_state.skills

    assessments = result.get("skill_assessments", [])

    by_name = {
        str(item.get("skill", "")).strip().lower(): item
        for item in assessments
    }

    total_importance = sum(
        skill["importance"] for skill in skills
    )

    ordered_assessments = []
    weighted_total = 0
    weight_used = 0

    for index, skill in enumerate(skills):

        item = by_name.get(skill["name"].lower())

        # Fall back to position if the model altered a skill name.
        if item is None and len(assessments) == len(skills):
            item = assessments[index]

        if item is None:
            continue

        item["skill"] = skill["name"]
        item["score"] = clamp_score(item.get("score"))
        item["level"] = level_for_score(item["score"])
        item["importance"] = skill["importance"]
        item["weight_percent"] = round(
            skill["importance"] / total_importance * 100
        )

        weighted_total += item["score"] * skill["importance"]
        weight_used += skill["importance"]

        ordered_assessments.append(item)

    overall_score = (
        round(weighted_total / weight_used)
        if weight_used
        else 0
    )

    result["skill_assessments"] = ordered_assessments
    result["overall_score"] = overall_score
    result["level"] = level_for_score(
        overall_score
    )

    return result


# =========================================================
# FINAL INTERVIEW ASSESSMENT
# =========================================================

def generate_interview_result():

    if not st.session_state.transcript:

        raise ValueError(
            "There are no interview answers to assess."
        )

    transcript_json = json.dumps(
        st.session_state.transcript,
        indent=2
    )

    cv_profile_json = json.dumps(
        st.session_state.cv_profile,
        indent=2
    )

    skills_json = json.dumps(
        [
            {
                "skill": skill["name"],
                "importance_out_of_5": skill["importance"]
            }
            for skill in st.session_state.skills
        ],
        indent=2
    )

    prompt = f"""
You are assessing a candidate after a structured job interview.

LANGUAGE REQUIREMENT:

The entire assessment MUST be written in English.
Never switch to another language.

JOB TITLE:

{st.session_state.job_title}


JOB DESCRIPTION:

{st.session_state.job_description}


SKILLS ASSESSED (with the importance the employer gave each skill):

{skills_json}


CANDIDATE CV INFORMATION:

{cv_profile_json}


INTERVIEW TRANSCRIPT:

{transcript_json}


ASSESSMENT RULES:

The interview answers must be the primary source of evidence.

The CV may provide supporting context, but do not treat a CV claim
as demonstrated competence unless the candidate's interview answers
support it.

Do not invent evidence.

Do not assume competence that was not demonstrated.

Do not penalise a candidate merely because the interview did not
ask about a particular subject.

If there is insufficient evidence for a skill, state that clearly.

ZERO-SCORE RULE:

Some transcripts are not genuine answers. They may be nonsensical,
irrelevant to the question, unintelligible, extremely short, written
in a language other than English, or unrelated sentences that look like
transcription errors (for example from silence or background noise).
Such text demonstrates no competence.

If ALL of the candidate's answers for a skill are of this kind, the
score for that skill MUST be exactly 0. Do not award any points on the
basis of the CV, the job description or the wording of the questions.
State clearly in the assessment that no relevant answer was given.

If only SOME answers for a skill are of this kind, ignore those answers
and score the skill using only the genuine answers.

Score each skill independently on its own merits. The importance
values are applied to the overall score separately, so do NOT
adjust a skill's score because of its importance and do NOT
calculate an overall score. Use the full range of the scale.

For each skill:

1. Assess the candidate's demonstrated knowledge.
2. Identify specific strengths.
3. Identify weaknesses, concerns or gaps.
4. Cite concise evidence from the candidate's actual answers.
5. Give the skill an integer score from 0 to 100.

Then provide:

- an overall assessment (a short narrative that gives more
  attention to the skills with higher importance)
- key strengths
- key concerns

SCORING GUIDE (per skill, 0-100):

0-20   = Little or no relevant competence demonstrated
21-40  = Limited competence with substantial gaps
41-60  = Acceptable competence with some gaps
61-80  = Strong competence demonstrated
81-100 = Exceptional and detailed competence demonstrated

Return ONLY valid JSON using this exact structure:

{{
    "overall_assessment": "",
    "key_strengths": [],
    "key_concerns": [],
    "skill_assessments": [
        {{
            "skill": "",
            "score": 0,
            "assessment": "",
            "strengths": [],
            "concerns": [],
            "evidence": []
        }}
    ]
}}
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt
    )

    result = clean_json_response(
        response.output_text
    )

    return apply_weighted_scoring(result)


# =========================================================
# CREATE TRANSCRIPT TEXT
# =========================================================

def create_transcript_text():

    candidate_name = (
        st.session_state.cv_profile or {}
    ).get(
        "candidate_name",
        "Candidate"
    )

    lines = [
        "STRUCTURED AI INTERVIEW TRANSCRIPT",
        "=" * 60,
        "",
        f"Candidate: {candidate_name}",
        f"Job title: {st.session_state.job_title}",
        f"Skills: {', '.join(skill_names())}",
        "",
        "=" * 60,
        ""
    ]

    for item in st.session_state.transcript:

        lines.extend(
            [
                f'SKILL: {item["skill"]}',
                f'QUESTION {item["question_number"]}',
                "",
                "INTERVIEWER:",
                item["question"],
                "",
                "CANDIDATE:",
                item["answer"],
                "",
                "=" * 60,
                ""
            ]
        )

    return "\n".join(lines)


# =========================================================
# RESET THE APPLICATION
# =========================================================

def reset_interview():

    keys_to_clear = list(
        st.session_state.keys()
    )

    for key in keys_to_clear:
        del st.session_state[key]

    st.rerun()


# =========================================================
# MOVE TO THE NEXT SKILL
# =========================================================

def advance_to_next_skill():

    st.session_state.current_skill_index += 1
    st.session_state.messages = []
    st.session_state.current_question = None
    st.session_state.current_question_audio = None
    st.session_state.recording_attempt = 0
    st.session_state.answer_processing = False


# =========================================================
# ADD SKILL (button callback)
# =========================================================

def add_skill_callback():

    cleaned_skill = st.session_state.get(
        "new_skill_name", ""
    ).strip()

    importance = st.session_state.get(
        "new_skill_importance", DEFAULT_IMPORTANCE
    )

    if not cleaned_skill:

        st.session_state.skill_form_message = (
            "Please enter a skill."
        )
        return

    if cleaned_skill.lower() in [
        name.lower() for name in skill_names()
    ]:

        st.session_state.skill_form_message = (
            "That skill has already been added."
        )
        return

    st.session_state.skills.append(
        {
            "name": cleaned_skill,
            "importance": importance
        }
    )

    st.session_state.skill_form_message = ""
    st.session_state.show_skill_input = False


# =========================================================
# SETUP SCREEN
# =========================================================

def show_setup_screen():

    st.title("🎙️ AI Interviewer")

    st.write(
        "Configure the interview, upload the candidate's CV "
        "and add the skills you want to assess."
    )

    st.divider()

    job_title = st.text_input(
        "Job Title",
        value=st.session_state.job_title,
        placeholder="e.g. AI Analyst"
    )

    job_description = st.text_area(
        "Job Description",
        value=st.session_state.job_description,
        placeholder="Paste the complete job description here...",
        height=250
    )

    uploaded_cv = st.file_uploader(
        "Candidate CV",
        type=["pdf"],
        help="Upload a text-based PDF containing the candidate's CV."
    )

    st.divider()

    st.subheader("Interview Skills")

    if st.session_state.skills:

        shares = importance_shares()

        for index, skill in enumerate(
            st.session_state.skills
        ):

            skill_column, delete_column = st.columns(
                [8, 1]
            )

            with skill_column:

                st.markdown(
                    f"**{skill['name']}** &nbsp;·&nbsp; "
                    f"importance {skill['importance']}/{IMPORTANCE_MAX} "
                    f"({IMPORTANCE_LABELS[skill['importance']]}) "
                    f"&nbsp;·&nbsp; {shares[index]}% of overall score"
                )

            with delete_column:

                if st.button(
                    "✕",
                    key=f"delete_skill_{index}",
                    help=f"Remove {skill['name']}"
                ):

                    st.session_state.skills.pop(
                        index
                    )

                    st.rerun()

    else:

        st.caption(
            "No skills have been added yet."
        )

    label_column, button_column = st.columns(
        [4, 1]
    )

    with label_column:

        st.markdown("### Add Skill")

    with button_column:

        if st.button(
            "➕",
            use_container_width=True,
            help="Add another interview skill"
        ):

            st.session_state.show_skill_input = True

    if st.session_state.show_skill_input:

        st.markdown(SLIDER_CSS, unsafe_allow_html=True)

        st.text_input(
            "Skill",
            placeholder="e.g. Python",
            key="new_skill_name"
        )

        new_importance = st.slider(
            "Importance of this skill for the role",
            min_value=IMPORTANCE_MIN,
            max_value=IMPORTANCE_MAX,
            value=DEFAULT_IMPORTANCE,
            step=1,
            key="new_skill_importance"
        )

        span = IMPORTANCE_MAX - IMPORTANCE_MIN

        markers = "".join(
            '<span class="imp-marker'
            #+ (" first" if value == IMPORTANCE_MIN else "")
            #+ (" last" if value == IMPORTANCE_MAX else "")
            + (" active" if value == new_importance else "")
            + f'" style="left:{(value - IMPORTANCE_MIN) / span * 100}%">'
            + f"{value}<br>{IMPORTANCE_LABELS[value]}</span>"
            for value in range(IMPORTANCE_MIN, IMPORTANCE_MAX + 1)
        )

        st.markdown(
            f'<div class="imp-markers">{markers}</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"**{IMPORTANCE_LABELS[new_importance]}:** {IMPORTANCE_MEANINGS[new_importance]}"
        )

        if st.session_state.skill_form_message:

            st.warning(
                st.session_state.skill_form_message
            )

        st.button(
            "Add Skill",
            type="primary",
            use_container_width=True,
            on_click=add_skill_callback
        )

    st.divider()

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
                "Please upload the candidate's CV."
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

        try:

            st.session_state.cv_text = read_pdf(
                uploaded_cv
            )

        except Exception as error:

            st.error(
                f"Could not read the CV: {error}"
            )

            return

        with st.spinner(
            "Analysing the candidate's CV..."
        ):

            try:

                st.session_state.cv_profile = (
                    extract_cv_profile(
                        st.session_state.cv_text,
                        skill_names()
                    )
                )

            except json.JSONDecodeError:

                st.error(
                    "The CV analysis was returned in an unexpected "
                    "format. Please try again."
                )

                return

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
        st.session_state.recording_attempt = 0
        st.session_state.current_question = None
        st.session_state.current_question_audio = None
        st.session_state.interview_result = None
        st.session_state.answer_processing = False

        st.rerun()


# =========================================================
# INTERVIEW SCREEN
# =========================================================

def show_interview_screen():

    if (
        st.session_state.current_skill_index
        >= len(st.session_state.skills)
    ):

        show_finished_screen()
        return

    current_skill = st.session_state.skills[
        st.session_state.current_skill_index
    ]["name"]

    candidate_name = (
        st.session_state.cv_profile or {}
    ).get(
        "candidate_name",
        "Candidate"
    )

    st.title("🎙️ AI Interview")

    st.caption(
        f"Candidate: {candidate_name} | "
        f"Position: {st.session_state.job_title}"
    )

    st.progress(
        (
            st.session_state.current_skill_index + 1
        )
        / len(st.session_state.skills)
    )

    st.divider()

    st.subheader(current_skill)

    current_skill_answers = [
        item
        for item in st.session_state.transcript
        if item["skill"] == current_skill
    ]

    st.caption(
        f"Interview skill "
        f"Question {len(current_skill_answers) + 1}"
    )

    # Generate the current question only once.
    if st.session_state.current_question is None:

        with st.spinner(
            "Preparing the next interview question..."
        ):

            try:

                st.session_state.current_question = (
                    generate_question(
                        current_skill
                    )
                )

                st.session_state.current_question_audio = (
                    generate_speech(
                        st.session_state.current_question
                    )
                )

            except Exception as error:

                st.error(
                    f"Could not generate the question: {error}"
                )

                if st.button(
                    "Try Again",
                    use_container_width=True
                ):

                    st.session_state.current_question = None
                    st.session_state.current_question_audio = None
                    st.rerun()

                return

    question = st.session_state.current_question

    st.markdown("### Interviewer")

    st.info(question)

    st.divider()

    st.markdown("### Candidate")

    st.caption(
        f"The question plays automatically. After it finishes you "
        f"have {PREP_COUNTDOWN_SECONDS} seconds to think, then "
        f"recording starts by itself (up to "
        f"{MAX_ANSWER_SECONDS // 60} minutes). Press Done when "
        f"you have finished answering."
    )

    audio_b64 = ""

    if st.session_state.current_question_audio:

        audio_b64 = base64.b64encode(
            st.session_state.current_question_audio
        ).decode("utf-8")

    # Identifies this exact question + attempt. The recorder echoes it
    # back, so an old recording is never mistaken for a new answer.
    recorder_nonce = (
        f"{st.session_state.question_number}_"
        f"{st.session_state.recording_attempt}"
    )

    recording = interview_recorder(
        question_audio=audio_b64,
        countdown_seconds=PREP_COUNTDOWN_SECONDS,
        max_record_seconds=MAX_ANSWER_SECONDS,
        nonce=recorder_nonce,
        key="interview_recorder",
        default=None
    )

    if (
        recording
        and recording.get("audio")
        and recording.get("nonce") == recorder_nonce
    ):

        with st.spinner(
            "Transcribing your answer..."
        ):

            try:

                answer = transcribe_audio(
                    base64.b64decode(recording["audio"]),
                    recording.get("mime", "")
                )

            except Exception as error:

                st.error(
                    f"Transcription failed: {error}"
                )

                if st.button(
                    "Record this answer again",
                    use_container_width=True
                ):

                    st.session_state.recording_attempt += 1
                    st.rerun()

                return

        st.session_state.transcript.append(
            {
                "skill": current_skill,
                "question_number":
                    st.session_state.question_number,
                "question": question,
                "answer": answer
            }
        )

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
        st.session_state.recording_attempt = 0
        st.session_state.current_question = None
        st.session_state.current_question_audio = None
        st.session_state.answer_processing = False

        # Maximum questions reached: move on automatically.
        if (
            len(current_skill_answers) + 1
            >= max_questions_for(current_skill)
        ):

            advance_to_next_skill()

        st.rerun()

    # Show previous answers for the current skill.
    if current_skill_answers:

        with st.expander(
            f"View recorded answers for {current_skill}"
        ):

            for item in current_skill_answers:

                st.markdown(
                    f"**Question {item['question_number']}**"
                )

                st.write(
                    item["question"]
                )

                st.markdown(
                    "**Candidate answer**"
                )

                st.write(
                    item["answer"]
                )

                st.divider()

    st.divider()

    is_last_skill = (
        st.session_state.current_skill_index
        == len(st.session_state.skills) - 1
    )

    if is_last_skill:

        finish_button_text = (
            "Skip remaining questions and Generate Result"
        )

    else:

        next_skill = st.session_state.skills[
            st.session_state.current_skill_index + 1
        ]["name"]

        finish_button_text = (
            f"Skip remaining questions and Move to {next_skill}"
        )

    if st.button(
        finish_button_text,
        use_container_width=True
    ):

        if not current_skill_answers:

            st.warning(
                "Please record at least one answer for this skill "
                "before moving on."
            )

            return

        advance_to_next_skill()

        st.rerun()


# =========================================================
# FINISHED SCREEN
# =========================================================

def show_finished_screen():

    st.title("✅ Interview Complete")

    st.success(
        "The interview has been completed."
    )

    if not st.session_state.transcript:

        st.error(
            "No interview answers were recorded, so an assessment "
            "cannot be generated."
        )

        if st.button(
            "Start New Interview",
            use_container_width=True
        ):

            reset_interview()

        return

    # Generate the result only once.
    if st.session_state.interview_result is None:

        with st.spinner(
            "Analysing interview performance and generating the result..."
        ):

            try:

                st.session_state.interview_result = (
                    generate_interview_result()
                )

            except json.JSONDecodeError:

                st.error(
                    "The assessment was returned in an unexpected "
                    "format. Select Retry Assessment."
                )

                if st.button(
                    "Retry Assessment",
                    use_container_width=True
                ):

                    st.session_state.interview_result = None
                    st.rerun()

                return

            except Exception as error:

                st.error(
                    f"Could not generate the interview result: {error}"
                )

                if st.button(
                    "Retry Assessment",
                    use_container_width=True
                ):

                    st.session_state.interview_result = None
                    st.rerun()

                return

    result = st.session_state.interview_result

    st.header("Interview Result")

    overall_level = result.get(
        "level",
        "Unqualified"
    )

    overall_score = result.get(
        "overall_score",
        0
    )

    st.markdown(
        level_badge(overall_level, large=True),
        unsafe_allow_html=True
    )

    st.metric(
        "Overall Score (importance-weighted)",
        f"{overall_score}%"
    )

    st.progress(
        max(0, min(100, overall_score)) / 100
    )

    st.markdown("### Overall Assessment")

    st.write(
        result.get(
            "overall_assessment",
            "No overall assessment was returned."
        )
    )

    strength_column, concern_column = st.columns(
        2
    )

    with strength_column:

        st.markdown("### Key Strengths")

        key_strengths = result.get(
            "key_strengths",
            []
        )

        if key_strengths:

            for strength in key_strengths:
                st.markdown(
                    f"- {strength}"
                )

        else:

            st.write(
                "No clear strengths were identified."
            )

    with concern_column:

        st.markdown("### Key Concerns")

        key_concerns = result.get(
            "key_concerns",
            []
        )

        if key_concerns:

            for concern in key_concerns:
                st.markdown(
                    f"- {concern}"
                )

        else:

            st.write(
                "No significant concerns were identified."
            )

    st.divider()

    st.header("Score Breakdown")

    skill_assessments = result.get(
        "skill_assessments",
        []
    )

    if not skill_assessments:

        st.warning(
            "The assessment did not contain individual skill results."
        )

    else:

        st.table(
            [
                {
                    "Skill": item["skill"],
                    "Importance": (
                        f"{item['importance']}/{IMPORTANCE_MAX}"
                    ),
                    "Weight": f"{item['weight_percent']}%",
                    "Score": f"{item['score']}%",
                    "Level": item["level"]
                }
                for item in skill_assessments
            ]
        )

    st.header("Skill Assessments")

    for skill_result in skill_assessments:

        skill_name = skill_result.get(
            "skill",
            "Skill"
        )

        skill_score = skill_result.get(
            "score",
            0
        )

        with st.container(
            border=True
        ):

            st.subheader(
                f"{skill_name}: {skill_score}%"
            )

            st.markdown(
                level_badge(
                    skill_result.get(
                        "level",
                        level_for_score(skill_score)
                    )
                ),
                unsafe_allow_html=True
            )

            st.progress(
                max(0, min(100, skill_score)) / 100
            )

            st.caption(
                f"Importance {skill_result.get('importance', '-')}"
                f"/{IMPORTANCE_MAX} "
                f"· {skill_result.get('weight_percent', '-')}% "
                f"of the overall score"
            )

            st.write(
                skill_result.get(
                    "assessment",
                    "No skill assessment was returned."
                )
            )

            st.markdown("**Strengths**")

            skill_strengths = skill_result.get(
                "strengths",
                []
            )

            if skill_strengths:

                for strength in skill_strengths:
                    st.markdown(
                        f"- {strength}"
                    )

            else:

                st.write(
                    "No clear strengths were demonstrated."
                )

            st.markdown("**Concerns or gaps**")

            skill_concerns = skill_result.get(
                "concerns",
                []
            )

            if skill_concerns:

                for concern in skill_concerns:
                    st.markdown(
                        f"- {concern}"
                    )

            else:

                st.write(
                    "No significant concerns were identified."
                )

            st.markdown("**Interview evidence**")

            evidence_items = skill_result.get(
                "evidence",
                []
            )

            if evidence_items:

                for evidence in evidence_items:
                    st.markdown(
                        f"- {evidence}"
                    )

            else:

                st.write(
                    "No specific interview evidence was returned."
                )

    # Create the text transcript before the expander.
    transcript_text = create_transcript_text()

    st.divider()

    with st.expander(
        "View Full Interview Transcript"
    ):

        for item in st.session_state.transcript:

            st.markdown(
                f"### {item['skill']}: "
                f"Question {item['question_number']}"
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

    st.header("Downloads")

    st.download_button(
        "📄 Download Transcript",
        data=transcript_text,
        file_name="interview_transcript.txt",
        mime="text/plain",
        use_container_width=True
    )

    complete_result = {
        "job_title":
            st.session_state.job_title,

        "job_description":
            st.session_state.job_description,

        "skills":
            st.session_state.skills,

        "candidate":
            st.session_state.cv_profile,

        "assessment":
            st.session_state.interview_result,

        "transcript":
            st.session_state.transcript
    }

    complete_result_json = json.dumps(
        complete_result,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        "📊 Download Complete Results",
        data=complete_result_json,
        file_name="interview_results.json",
        mime="application/json",
        use_container_width=True
    )

    st.divider()

    if st.button(
        "Start New Interview",
        use_container_width=True
    ):

        reset_interview()


# =========================================================
# MAIN APPLICATION
# =========================================================

if st.session_state.interview_started:

    show_interview_screen()

else:

    show_setup_screen()
