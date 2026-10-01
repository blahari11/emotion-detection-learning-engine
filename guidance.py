import os

FALLBACK = {
    "Bored": """### Supportive response
It sounds like the topic is not holding your attention right now.

### Personalized strategy
Turn the topic into a short challenge. Study for 15 minutes, then solve one practical example connected to something you already like.

### Next steps
1. Identify one real-world use of the topic.
2. Learn only one small concept.
3. Test yourself with a quick question.

**Encouragement:** Small, active sessions can make a dull topic easier to engage with.""",
    "Confident": """### Supportive response
You appear confident with the current concept. This is a good time to test and deepen your understanding.

### Personalized strategy
Move from recognition to application by explaining the idea without notes and solving a slightly harder problem.

### Next steps
1. Explain the concept in your own words.
2. Solve one intermediate problem.
3. Identify one edge case or limitation.

**Encouragement:** Use your confidence to build depth, not just speed.""",
    "Confused": """### Supportive response
The concept seems unclear right now, so the best approach is to reduce it into smaller pieces.

### Personalized strategy
Start with the basic definition, then trace one simple example line by line. Avoid moving to advanced problems until the first example makes sense.

### Next steps
1. Write the concept in one simple sentence.
2. Follow one beginner example step by step.
3. Explain each step aloud and note the exact point that becomes unclear.

**Encouragement:** Clarity often comes from isolating one unclear step at a time.""",
    "Curious": """### Supportive response
Your question shows curiosity and a desire to understand the idea beyond memorization.

### Personalized strategy
Use a 'why → how → example' learning path. First ask why the concept exists, then how it works, and finally test it with an example.

### Next steps
1. Find the problem the concept was designed to solve.
2. Study its core mechanism.
3. Compare it with one alternative approach.

**Encouragement:** Curiosity is useful when you turn each question into a small experiment.""",
    "Frustrated": """### Supportive response
You seem frustrated after repeated difficulty. Repeating the same approach may add more pressure without improving clarity.

### Personalized strategy
Stop retrying the full problem. Find the first step where your reasoning breaks and practice only that smaller skill with an easy example.

### Next steps
1. Write down the exact step where you get stuck.
2. Review one prerequisite connected to that step.
3. Solve one easier version before returning to the original problem.

**Encouragement:** Change the learning approach before increasing the number of attempts."""
}

def _api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None

def generate_guidance(student_text, emotion, use_gemini=False):
    if not use_gemini:
        return FALLBACK[emotion]

    key = _api_key()
    if not key:
        return FALLBACK[emotion] + "\n\n> Gemini is disabled because no API key was configured."

    prompt = f"""
You are an empathetic academic learning support assistant.
The learner wrote: {student_text!r}
Detected learning emotion: {emotion}

Give a concise response with exactly these headings:
### Supportive response
### Personalized strategy
### Next steps
### Encouragement

Under Next steps, provide exactly 3 numbered steps.
Focus on academic learning. Do not make medical or psychological diagnoses.
Do not shame the learner. Keep the total response under 220 words.
"""
    try:
        from google import genai
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as exc:
        return FALLBACK[emotion] + f"\n\n> Gemini request was unavailable; local guidance was used."
