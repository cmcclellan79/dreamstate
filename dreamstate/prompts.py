"""All prompt templates in one place, so they're easy to review and version."""

# ---------- Corpus generation (teacher, run once) ----------

WORLD_DOCS = """You are writing source documents for a memory experiment. Invent a fictional setting with \
people, places, organizations, dates, numbers and events that do NOT exist in the real world. Every fact must \
be invented and specific (names, years, quantities), so a reader could not guess it from general knowledge.

Write {n} short documents (150-250 words each) for day {day} of a chronicle. Mix genres: news report, \
encyclopedia entry, letter, meeting minutes, field report. Each document should contain 6-10 distinct, \
checkable facts.

Do not reuse any of these names already used on earlier days: {used_names}

Return ONLY JSON: a list of objects with keys "title", "genre", "text", "entities" (list of the proper \
names you invented in that document)."""

EVAL_QUESTIONS = """Below is a document. Write {n} short-answer questions that test facts stated in it.

Rules:
- Each question must be answerable from the document alone, with a short answer (a name, number, date or phrase).
- Paraphrase: do NOT copy the document's sentence structure or distinctive wording into the question.
- Cover different facts; no two questions about the same fact.
- Each question must make sense on its own without the document (name the entity it's about).

Document:
\"\"\"{text}\"\"\"

Return ONLY JSON: a list of objects with keys "question" and "answer"."""

# ---------- Card making (student, with the document in context = "wake") ----------

MAKE_CARDS = """Read the document below carefully. You will NOT be able to see it again. Write {n} flashcards \
that would help you remember its most important facts later.

Each flashcard has a question on the front and a short answer on the back. Questions must name the entity \
they're about so they make sense without the document.

Document:
\"\"\"{text}\"\"\"

Return ONLY JSON: a list of objects with keys "q" and "a"."""

# ---------- Card verification (judge, with source) ----------

VERIFY_CARD = """You are checking a flashcard against its source document.

Document:
\"\"\"{text}\"\"\"

Flashcard question: {q}
Flashcard answer: {a}

Is the answer fully supported by the document and a correct answer to the question? \
Reply ONLY with JSON: {{"valid": true}} or {{"valid": false}}."""

# ---------- Closed-book quiz (student, no document) ----------

QUIZ_SYSTEM = "Answer the question with a short, direct answer. No explanation."

# ---------- Grading (judge) ----------

GRADE_ANSWER = """Grade a student's answer against a reference answer.

Question: {q}
Reference answer: {ref}
Student answer: {pred}

The student is correct if their answer means the same thing as the reference (ignore phrasing, \
capitalization and minor formatting; numbers and names must match). \
Reply ONLY with JSON: {{"correct": true}} or {{"correct": false}}."""
