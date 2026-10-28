import json
from io import BytesIO

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

CHAT_MODEL = "gpt-5.4"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "cedar"


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

def initialise_session_state():

    default_values = {
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
        "question_number": 1,
        "interview_result": None,
        "answer_processing": False
    }

    for key, value in default_values.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()


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

def build_interview_context(skill):

    skill_evidence = get_skill_evidence(skill)

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
not a written examination.

Where the candidate gives an interesting or incomplete answer,
ask a relevant follow-up question.

Do not tell the candidate whether an answer was good or bad.

Do not provide feedback, a score or an assessment during the
interview.

Only return the next interview question.

The question MUST be written entirely in English.
"""


# =========================================================
# QUESTION GENERATION
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

def transcribe_audio(audio_file):

    # Streamlit's UploadedFile-like object may already have a name,
    # but giving OpenAI an explicit filename helps it identify the format.
    audio_bytes = audio_file.getvalue()

    transcription_file = BytesIO(audio_bytes)
    transcription_file.name = "candidate_answer.wav"

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
        st.session_state.skills,
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


SKILLS ASSESSED:

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

For each skill:

1. Assess the candidate's demonstrated knowledge.
2. Identify specific strengths.
3. Identify weaknesses, concerns or gaps.
4. Cite concise evidence from the candidate's actual answers.
5. Give the skill an integer score from 1 to 5.

Then provide:

- an overall integer score from 1 to 5
- an overall assessment
- key strengths
- key concerns
- a final recommendation

The recommendation MUST be exactly one of:

"Strongly Recommended"
"Recommended"
"Proceed with Caution"
"Not Recommended"

SCORING GUIDE:

1 = Little or no relevant competence demonstrated
2 = Limited competence with substantial gaps
3 = Acceptable competence with some gaps
4 = Strong competence demonstrated
5 = Exceptional and detailed competence demonstrated

Return ONLY valid JSON using this exact structure:

{{
    "overall_score": 0,
    "recommendation": "",
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

    return result


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
        f"Skills: {', '.join(st.session_state.skills)}",
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

        for index, skill in enumerate(
            st.session_state.skills
        ):

            skill_column, delete_column = st.columns(
                [8, 1]
            )

            with skill_column:

                st.markdown(
                    f"**{skill}**"
                )

            with delete_column:

                if st.button(
                    "✕",
                    key=f"delete_skill_{index}",
                    help=f"Remove {skill}"
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

        with st.form(
            "add_skill_form",
            clear_on_submit=True
        ):

            new_skill = st.text_input(
                "Skill",
                placeholder="e.g. Python"
            )

            add_skill_submitted = st.form_submit_button(
                "Add Skill",
                type="primary",
                use_container_width=True
            )

            if add_skill_submitted:

                cleaned_skill = new_skill.strip()

                if not cleaned_skill:

                    st.warning(
                        "Please enter a skill."
                    )

                else:

                    existing_skills = [
                        skill.lower()
                        for skill in st.session_state.skills
                    ]

                    if (
                        cleaned_skill.lower()
                        in existing_skills
                    ):

                        st.warning(
                            "That skill has already been added."
                        )

                    else:

                        st.session_state.skills.append(
                            cleaned_skill
                        )

                        st.session_state.show_skill_input = False

                        st.rerun()

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
                        st.session_state.skills
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
    ]

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

    st.caption(
        f"Interview skill "
        f"{st.session_state.current_skill_index + 1} "
        f"of {len(st.session_state.skills)}"
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

    if st.session_state.current_question_audio:

        st.audio(
            st.session_state.current_question_audio,
            format="audio/mpeg",
            autoplay=True
        )

    st.divider()

    st.markdown("### Candidate")

    audio = st.audio_input(
        "Record your answer",
        sample_rate=None,
        key=f"answer_{st.session_state.question_number}"
    )

    if audio is not None:

        st.caption(
            "Your recording is ready. Select Submit Answer "
            "when you are satisfied with it."
        )

        if st.button(
            "Submit Answer",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.answer_processing
        ):

            st.session_state.answer_processing = True

            with st.spinner(
                "Transcribing your answer..."
            ):

                try:

                    answer = transcribe_audio(
                        audio
                    )

                except Exception as error:

                    st.session_state.answer_processing = False

                    st.error(
                        f"Transcription failed: {error}"
                    )

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
            st.session_state.current_question = None
            st.session_state.current_question_audio = None
            st.session_state.answer_processing = False

            st.rerun()

    # Show previous answers for the current skill.
    current_skill_answers = [
        item
        for item in st.session_state.transcript
        if item["skill"] == current_skill
    ]

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
            "Finish Interview and Generate Result"
        )

    else:

        next_skill = st.session_state.skills[
            st.session_state.current_skill_index + 1
        ]

        finish_button_text = (
            f"Finish {current_skill} and Move to {next_skill}"
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

        st.session_state.current_skill_index += 1
        st.session_state.messages = []
        st.session_state.current_question = None
        st.session_state.current_question_audio = None
        st.session_state.answer_processing = False

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

    recommendation = result.get(
        "recommendation",
        "Assessment unavailable"
    )

    overall_score = result.get(
        "overall_score",
        0
    )

    st.subheader(recommendation)

    st.metric(
        "Overall Score",
        f"{overall_score} / 5"
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

    st.header("Skill Assessments")

    skill_assessments = result.get(
        "skill_assessments",
        []
    )

    if not skill_assessments:

        st.warning(
            "The assessment did not contain individual skill results."
        )

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
                f"{skill_name}: {skill_score}/5"
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
