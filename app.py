import json
import time
from io import BytesIO

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from pypdf import PdfReader

CHAT_MODEL = "gpt-5.4"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "cedar"
QUESTIONS_PER_SKILL = 3
COUNTDOWN_SECONDS = 3
SILENCE_TO_STOP_SECONDS = 2.5

st.set_page_config(page_title="AI Interviewer", page_icon="🎙️", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


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
        "skill_question_count": 0,
        "messages": [],
        "current_question": None,
        "current_question_audio": None,
        "transcript": [],
        "question_number": 1,
        "interview_result": None,
        "answer_processing": False,
        "recording_ready": False,
        "processed_recording_ids": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()


def read_pdf(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    cv_text = "\n\n".join(pages)
    if not cv_text.strip():
        raise ValueError(
            "No readable text was found. The CV may be a scanned image rather than a text PDF."
        )
    return cv_text


def clean_json_response(response_text):
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def extract_cv_profile(cv_text, skills):
    prompt = f"""
You are analysing a candidate CV before a structured job interview.
Return the complete response in English only.

Interview skills:
{json.dumps(skills, ensure_ascii=False)}

Extract the candidate's name, education, employment history, projects, and
specific evidence relating to every interview skill. Do not invent information.
If no evidence exists for a skill, return an empty list for that skill.

Return ONLY valid JSON with this exact structure:
{{
  "candidate_name": "",
  "education": [],
  "employment_history": [],
  "projects": [],
  "skill_evidence": {{"skill name": []}}
}}

CANDIDATE CV:
{cv_text}
"""
    response = client.responses.create(model=CHAT_MODEL, input=prompt)
    return clean_json_response(response.output_text)


def get_skill_evidence(skill):
    profile = st.session_state.cv_profile or {}
    evidence_map = profile.get("skill_evidence", {})
    if skill in evidence_map:
        return evidence_map[skill]
    for evidence_skill, evidence in evidence_map.items():
        if evidence_skill.lower() == skill.lower():
            return evidence
    return []


def build_interview_context(skill):
    evidence = get_skill_evidence(skill)
    return f"""
You are conducting a professional structured job interview.

LANGUAGE RULE:
The entire interview MUST be conducted in English. Always ask every question
in English. Never switch languages, even if the CV or a transcription contains
another language. If a transcription appears incorrect or non-English, continue
in English and ask a clear English follow-up.

JOB TITLE:
{st.session_state.job_title}

JOB DESCRIPTION:
{st.session_state.job_description}

CURRENT SKILL:
{skill}

RELEVANT CV EVIDENCE:
{json.dumps(evidence, indent=2, ensure_ascii=False)}

Ask exactly ONE concise, natural interview question at a time. Use CV evidence
where relevant. Explore the candidate's personal contribution, technical choices,
reasoning, problems, results, lessons, conceptual understanding, and what the
candidate would change. Follow up on incomplete answers. Do not provide feedback,
scores, or assessments during the interview. Return only the next question in English.
"""


def generate_question(skill):
    conversation = [{"role": "system", "content": build_interview_context(skill)}]
    conversation.extend(st.session_state.messages)
    response = client.responses.create(model=CHAT_MODEL, input=conversation)
    question = response.output_text.strip()
    if not question:
        raise ValueError("The interviewer did not generate a question.")
    return question


def generate_speech(text):
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions=(
            "Speak entirely in English like a friendly professional interviewer. "
            "Use a calm, natural, conversational delivery."
        ),
    )
    return response.content


def transcribe_audio(audio_bytes):
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "candidate_answer.wav"
    transcription = client.audio.transcriptions.create(
        model=TRANSCRIPTION_MODEL,
        file=audio_file,
        language="en",
        prompt=(
            "This is a professional job interview conducted entirely in English. "
            "Transcribe the candidate's spoken English accurately. Do not translate "
            "the speech into another language."
        ),
    )
    text = transcription.text.strip()
    if not text:
        raise ValueError("No speech could be identified in the recording.")
    return text


def generate_interview_result():
    if not st.session_state.transcript:
        raise ValueError("There are no interview answers to assess.")

    prompt = f"""
You are assessing a candidate after a structured job interview.
Write the entire assessment in English only.

JOB TITLE:
{st.session_state.job_title}

JOB DESCRIPTION:
{st.session_state.job_description}

SKILLS:
{json.dumps(st.session_state.skills, ensure_ascii=False)}

CANDIDATE CV INFORMATION:
{json.dumps(st.session_state.cv_profile, indent=2, ensure_ascii=False)}

INTERVIEW TRANSCRIPT:
{json.dumps(st.session_state.transcript, indent=2, ensure_ascii=False)}

Use interview answers as the primary evidence. The CV is supporting context only.
Do not invent evidence or assume competence that was not demonstrated. Do not
penalise the candidate for something the interview did not ask. If evidence is
insufficient, say so clearly.

Score every skill and the overall interview with an integer from 1 to 5:
1 = little or no competence demonstrated
2 = limited competence with substantial gaps
3 = acceptable competence with some gaps
4 = strong competence demonstrated
5 = exceptional and detailed competence demonstrated

The recommendation must be exactly one of:
"Strongly Recommended", "Recommended", "Proceed with Caution", "Not Recommended"

Return ONLY valid JSON with this exact structure:
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
    response = client.responses.create(model=CHAT_MODEL, input=prompt)
    return clean_json_response(response.output_text)


def create_transcript_text():
    candidate = (st.session_state.cv_profile or {}).get("candidate_name", "Candidate")
    lines = [
        "STRUCTURED AI INTERVIEW TRANSCRIPT",
        "=" * 60,
        "",
        f"Candidate: {candidate}",
        f"Job title: {st.session_state.job_title}",
        f"Skills: {', '.join(st.session_state.skills)}",
        "",
        "=" * 60,
        "",
    ]
    for item in st.session_state.transcript:
        lines.extend(
            [
                f"SKILL: {item['skill']}",
                f"QUESTION {item['question_number']}",
                "",
                "INTERVIEWER:",
                item["question"],
                "",
                "CANDIDATE:",
                item["answer"],
                "",
                "=" * 60,
                "",
            ]
        )
    return "\n".join(lines)


def reset_interview():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def advance_after_answer():
    st.session_state.skill_question_count += 1
    st.session_state.question_number += 1
    st.session_state.current_question = None
    st.session_state.current_question_audio = None
    st.session_state.recording_ready = False
    st.session_state.answer_processing = False

    if st.session_state.skill_question_count >= QUESTIONS_PER_SKILL:
        st.session_state.current_skill_index += 1
        st.session_state.skill_question_count = 0
        st.session_state.messages = []


def show_setup_screen():
    st.title("🎙️ AI Interviewer")
    st.write(
        "Upload the candidate's CV, enter the job description, and add the skills "
        "to assess"
    )
    st.divider()

    job_title = st.text_input(
        "Job Title", value=st.session_state.job_title, placeholder="e.g. AI Analyst"
    )
    job_description = st.text_area(
        "Job Description",
        value=st.session_state.job_description,
        placeholder="Paste the complete job description here...",
        height=250,
    )
    uploaded_cv = st.file_uploader(
        "Candidate CV", type=["pdf"], help="Upload a text-based PDF CV."
    )

    st.divider()
    st.subheader("Required Skills")

    if st.session_state.skills:
        for index, skill in enumerate(st.session_state.skills):
            skill_col, delete_col = st.columns([8, 1])
            with skill_col:
                st.markdown(f"**{skill}**")
            with delete_col:
                if st.button("✕", key=f"delete_skill_{index}", help=f"Remove {skill}"):
                    st.session_state.skills.pop(index)
                    st.rerun()
    else:
        st.caption("No skills have been added yet.")

    label_col, button_col = st.columns([4, 1])
    with label_col:
        st.markdown("### Add Skill")
    with button_col:
        if st.button("➕", use_container_width=True, help="Add another skill"):
            st.session_state.show_skill_input = True

    if st.session_state.show_skill_input:
        with st.form("add_skill_form", clear_on_submit=True):
            new_skill = st.text_input("Skill", placeholder="e.g. Python")
            submitted = st.form_submit_button(
                "Add Skill", type="primary", use_container_width=True
            )
            if submitted:
                cleaned = new_skill.strip()
                existing = [skill.lower() for skill in st.session_state.skills]
                if not cleaned:
                    st.warning("Please enter a skill.")
                elif cleaned.lower() in existing:
                    st.warning("That skill has already been added.")
                else:
                    st.session_state.skills.append(cleaned)
                    st.session_state.show_skill_input = False
                    st.rerun()

    st.divider()
    if st.button("Start Interview", type="primary", use_container_width=True):
        if not job_title.strip():
            st.error("Please enter a job title.")
            return
        if not job_description.strip():
            st.error("Please enter a job description.")
            return
        if uploaded_cv is None:
            st.error("Please upload the candidate's CV.")
            return
        if not st.session_state.skills:
            st.error("Please add at least one interview skill.")
            return

        st.session_state.job_title = job_title.strip()
        st.session_state.job_description = job_description.strip()

        try:
            st.session_state.cv_text = read_pdf(uploaded_cv)
        except Exception as error:
            st.error(f"Could not read the CV: {error}")
            return

        with st.spinner("Analysing the candidate's CV..."):
            try:
                st.session_state.cv_profile = extract_cv_profile(
                    st.session_state.cv_text, st.session_state.skills
                )
            except json.JSONDecodeError:
                st.error("The CV analysis had an unexpected format. Please try again.")
                return
            except Exception as error:
                st.error(f"CV analysis failed: {error}")
                return

        st.session_state.interview_started = True
        st.session_state.current_skill_index = 0
        st.session_state.skill_question_count = 0
        st.session_state.messages = []
        st.session_state.transcript = []
        st.session_state.question_number = 1
        st.session_state.current_question = None
        st.session_state.current_question_audio = None
        st.session_state.interview_result = None
        st.session_state.answer_processing = False
        st.session_state.recording_ready = False
        st.session_state.processed_recording_ids = []
        st.rerun()


def show_countdown():
    countdown_box = st.empty()
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        countdown_box.markdown(
            f"<div style='text-align:center;font-size:52px;font-weight:700'>"
            f"Recording in {remaining}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(1)
    countdown_box.markdown(
        "<div style='text-align:center;font-size:30px;font-weight:700;color:#d62728'>"
        "🔴 Recording now. Start speaking.</div>",
        unsafe_allow_html=True,
    )


def process_recording(recording, current_skill, question):
    recording_id = str(recording.get("id", ""))
    if recording_id and recording_id in st.session_state.processed_recording_ids:
        return

    st.session_state.answer_processing = True
    with st.spinner("Transcribing your answer and preparing the next question..."):
        try:
            answer = transcribe_audio(recording["bytes"])
        except Exception as error:
            st.session_state.answer_processing = False
            st.error(f"Transcription failed: {error}")
            return

    if recording_id:
        st.session_state.processed_recording_ids.append(recording_id)

    st.session_state.transcript.append(
        {
            "skill": current_skill,
            "question_number": st.session_state.question_number,
            "question": question,
            "answer": answer,
        }
    )
    st.session_state.messages.extend(
        [
            {"role": "assistant", "content": question},
            {"role": "user", "content": answer},
        ]
    )
    advance_after_answer()
    st.rerun()


def show_interview_screen():
    if st.session_state.current_skill_index >= len(st.session_state.skills):
        show_finished_screen()
        return

    current_skill = st.session_state.skills[st.session_state.current_skill_index]
    candidate = (st.session_state.cv_profile or {}).get("candidate_name", "Candidate")

    st.title("🎙️ AI Interview")
    st.caption(f"Candidate: {candidate} | Position: {st.session_state.job_title}")
    total_questions = len(st.session_state.skills) * QUESTIONS_PER_SKILL
    completed_questions = len(st.session_state.transcript)
    st.progress(min(completed_questions / total_questions, 1.0))
    st.caption(
        f"Skill {st.session_state.current_skill_index + 1} of "
        f"{len(st.session_state.skills)} | Question "
        f"{st.session_state.skill_question_count + 1} of {QUESTIONS_PER_SKILL}"
    )
    st.divider()
    st.subheader(current_skill)

    if st.session_state.current_question is None:
        with st.spinner("Preparing the next interview question..."):
            try:
                st.session_state.current_question = generate_question(current_skill)
                st.session_state.current_question_audio = generate_speech(
                    st.session_state.current_question
                )
            except Exception as error:
                st.error(f"Could not generate the question: {error}")
                if st.button("Try Again", use_container_width=True):
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
            autoplay=True,
        )

    st.divider()
    st.markdown("### Candidate")

    if not st.session_state.recording_ready:
        st.write(
            "When the question has finished playing, select **Begin Answer**. "
            "A three-second countdown will run, recording will start, and it will "
            "stop automatically after you finish speaking."
        )
        if st.button("Begin Answer", type="primary", use_container_width=True):
            st.session_state.recording_ready = True
            st.rerun()
        return

    show_countdown()

    recorder_key = f"recorder_{st.session_state.question_number}"
    recording = audio_recorder(
        text="",
        recording_color="#e53935",
        neutral_color="#1f77b4",
        icon_name="microphone",
        icon_size="4x",
        energy_threshold=0.01,
        pause_threshold=SILENCE_TO_STOP_SECONDS,
        sample_rate=16000,
        key=recorder_key,
    )

    st.caption(
        "Speak naturally. Recording stops automatically after approximately "
        f"{SILENCE_TO_STOP_SECONDS:g} seconds of silence. Your answer is then "
        "submitted automatically."
    )

    if recording and not st.session_state.answer_processing:
        if isinstance(recording, bytes):
            recording = {
                "bytes": recording,
                "id": f"{st.session_state.question_number}-{len(recording)}",
            }
        process_recording(recording, current_skill, question)

    if st.button("Cancel and replay question", use_container_width=True):
        st.session_state.recording_ready = False
        st.rerun()


# The named function is defined before runtime reaches show_interview_screen().
def show_finished_screen():
    st.title("✅ Interview Complete")
    st.success("The interview has been completed.")

    if not st.session_state.transcript:
        st.error("No answers were recorded, so an assessment cannot be generated.")
        if st.button("Start New Interview", use_container_width=True):
            reset_interview()
        return

    if st.session_state.interview_result is None:
        with st.spinner("Analysing performance and generating the result..."):
            try:
                st.session_state.interview_result = generate_interview_result()
            except json.JSONDecodeError:
                st.error("The assessment had an unexpected format. Select Retry.")
                if st.button("Retry Assessment", use_container_width=True):
                    st.session_state.interview_result = None
                    st.rerun()
                return
            except Exception as error:
                st.error(f"Could not generate the interview result: {error}")
                if st.button("Retry Assessment", use_container_width=True):
                    st.session_state.interview_result = None
                    st.rerun()
                return

    result = st.session_state.interview_result
    st.header("Interview Result")
    st.subheader(result.get("recommendation", "Assessment unavailable"))
    st.metric("Overall Score", f"{result.get('overall_score', 0)} / 5")
    st.markdown("### Overall Assessment")
    st.write(result.get("overall_assessment", "No overall assessment was returned."))

    strength_col, concern_col = st.columns(2)
    with strength_col:
        st.markdown("### Key Strengths")
        strengths = result.get("key_strengths", [])
        if strengths:
            for strength in strengths:
                st.markdown(f"- {strength}")
        else:
            st.write("No clear strengths were identified.")
    with concern_col:
        st.markdown("### Key Concerns")
        concerns = result.get("key_concerns", [])
        if concerns:
            for concern in concerns:
                st.markdown(f"- {concern}")
        else:
            st.write("No significant concerns were identified.")

    st.divider()
    st.header("Skill Assessments")
    assessments = result.get("skill_assessments", [])
    if not assessments:
        st.warning("The assessment did not contain individual skill results.")

    for skill_result in assessments:
        with st.container(border=True):
            st.subheader(
                f"{skill_result.get('skill', 'Skill')}: "
                f"{skill_result.get('score', 0)}/5"
            )
            st.write(skill_result.get("assessment", "No assessment was returned."))
            st.markdown("**Strengths**")
            for item in skill_result.get("strengths", []):
                st.markdown(f"- {item}")
            if not skill_result.get("strengths"):
                st.write("No clear strengths were demonstrated.")
            st.markdown("**Concerns or gaps**")
            for item in skill_result.get("concerns", []):
                st.markdown(f"- {item}")
            if not skill_result.get("concerns"):
                st.write("No significant concerns were identified.")
            st.markdown("**Interview evidence**")
            for item in skill_result.get("evidence", []):
                st.markdown(f"- {item}")
            if not skill_result.get("evidence"):
                st.write("No specific interview evidence was returned.")

    transcript_text = create_transcript_text()
    st.divider()
    with st.expander("View Full Interview Transcript"):
        for item in st.session_state.transcript:
            st.markdown(f"### {item['skill']}: Question {item['question_number']}")
            st.markdown("**Interviewer**")
            st.write(item["question"])
            st.markdown("**Candidate**")
            st.write(item["answer"])
            st.divider()

    st.header("Downloads")
    st.download_button(
        "📄 Download Transcript",
        data=transcript_text,
        file_name="interview_transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )

    complete_result = {
        "job_title": st.session_state.job_title,
        "job_description": st.session_state.job_description,
        "skills": st.session_state.skills,
        "candidate": st.session_state.cv_profile,
        "assessment": st.session_state.interview_result,
        "transcript": st.session_state.transcript,
    }
    st.download_button(
        "📊 Download Complete Results",
        data=json.dumps(complete_result, indent=2, ensure_ascii=False),
        file_name="interview_results.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    if st.button("Start New Interview", use_container_width=True):
        reset_interview()


if st.session_state.interview_started:
    show_interview_screen()
else:
    show_setup_screen()
