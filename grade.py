"""
CBSE AI GRADING SYSTEM
Fair grading version (no annotation)
"""

import base64
import json
import os
import sys
import io

from openai import AzureOpenAI
from pdf2image import convert_from_path


# =============================
# AZURE CONFIG
# =============================

AZURE_ENDPOINT = "https://cbse-grading-resource.openai.azure.com/"
AZURE_API_KEY = "22LlsoRrwbZUSOU5fEqLLtVnSWgm6MzN6Knww9dgJM4gxeCCXBgkJQQJ99CCACHYHv6XJ3w3AAAAACOGuvUB"
DEPLOYMENT = "gpt-4o"


# =============================
# SAFE JSON PARSER
# =============================

def safe_json(raw):

    raw = raw.replace("```json", "").replace("```", "").strip()

    start = raw.find("{")
    end = raw.rfind("}") + 1

    if start != -1 and end != -1:
        raw = raw[start:end]

    try:
        return json.loads(raw)

    except:
        print("⚠ Invalid JSON returned:")
        print(raw)
        return None


# =============================
# LOAD FILES
# =============================

def file_to_images(path):

    images = []

    # folder input
    if os.path.isdir(path):

        print("📂 Loading images from folder")

        files = sorted(os.listdir(path))

        for f in files:

            if f.lower().endswith(("jpg", "jpeg", "png")):

                full = os.path.join(path, f)

                with open(full, "rb") as img:

                    images.append(
                        base64.b64encode(img.read()).decode()
                    )

        print(f"✅ {len(images)} page(s) loaded")

        return images


    ext = path.split(".")[-1].lower()

    # pdf input
    if ext == "pdf":

        print("📄 Converting PDF to images")

        pages = convert_from_path(path, dpi=200)

        for p in pages:

            buf = io.BytesIO()
            p.save(buf, "JPEG")

            images.append(
                base64.b64encode(buf.getvalue()).decode()
            )

        print(f"✅ {len(images)} page(s) converted")

        return images


    # single image
    with open(path, "rb") as f:

        images.append(base64.b64encode(f.read()).decode())

    return images


# =============================
# IMAGE FORMAT FOR GPT
# =============================

def make_blocks(images):

    return [

        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}",
                "detail": "high"
            }
        }

        for img in images
    ]


# =============================
# EXTRACT QUESTIONS
# =============================

def extract_questions(client, paper_path):

    print("\n📋 Reading question paper...")

    images = file_to_images(paper_path)

    prompt = """
Read this CBSE question paper carefully.

Extract all questions.

Return JSON only:

{
 "questions":[
   {
    "q_no":"Q1",
    "question":"",
    "max_marks":5
   }
 ]
}
"""

    res = client.chat.completions.create(

        model=DEPLOYMENT,

        messages=[{
            "role": "user",
            "content": make_blocks(images) + [
                {"type": "text", "text": prompt}
            ]
        }],

        temperature=0
    )

    raw = res.choices[0].message.content

    paper = safe_json(raw)

    print("✅ Questions detected:", len(paper["questions"]))

    return paper


# =============================
# GRADE QUESTION
# =============================

def grade_question(client, q, answer_images):

    prompt = f"""
You are a fair CBSE mathematics examiner.

Evaluate the student's handwritten answer.

Reward partial effort and correct steps.

Question:
{q['question']}

Maximum Marks: {q['max_marks']}

Return JSON:

{{
 "q_no": "{q['q_no']}",
 "marks_awarded": 0,
 "max_marks": {q['max_marks']},

 "positives": "",
 "weakness": "",
 "error_step": "",
 "correct_solution": ""
}}
"""

    res = client.chat.completions.create(

        model=DEPLOYMENT,

        messages=[
            {"role": "system", "content": "You are a CBSE examiner awarding step marks."},

            {
                "role": "user",
                "content": make_blocks(answer_images) + [
                    {"type": "text", "text": prompt}
                ]
            }
        ],

        temperature=0.2
    )

    raw = res.choices[0].message.content

    return safe_json(raw)


# =============================
# GRADE ALL QUESTIONS
# =============================

def grade_all(client, paper, answer_path):

    answer_images = file_to_images(answer_path)

    results = []

    total = 0
    max_total = 0

    print("\n✏️ Grading answers...\n")

    for q in paper["questions"]:

        print("Grading", q["q_no"])

        r = grade_question(
            client,
            q,
            answer_images
        )

        if r:

            results.append(r)

            total += r["marks_awarded"]
            max_total += r["max_marks"]

    return results, total, max_total


# =============================
# REPORT
# =============================

def print_report(results, total, max_total):

    print("\n================================")
    print("CBSE AI GRADING REPORT")
    print("================================\n")

    for r in results:

        print(r["q_no"])
        print("Marks:", r["marks_awarded"], "/", r["max_marks"])

        print("\nPositive Points:")
        print(r["positives"])

        print("\nWeakness:")
        print(r["weakness"])

        print("\nError Step:")
        print(r["error_step"])

        print("\nCorrect Solution:")
        print(r["correct_solution"])

        print("\n--------------------------------\n")

    pct = round((total / max_total) * 100, 1)

    print("FINAL SCORE:", total, "/", max_total, f"({pct}%)")


# =============================
# MAIN
# =============================

if __name__ == "__main__":

    client = AzureOpenAI(

        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version="2024-08-01-preview"
    )

    print("\n🎓 CBSE AI Grader\n")

    if len(sys.argv) == 3:

        question_paper = sys.argv[1]
        answer_sheet = sys.argv[2]

    else:

        question_paper = input("Question paper path: ")
        answer_sheet = input("Answer sheet path: ")

    paper = extract_questions(client, question_paper)

    results, total, max_total = grade_all(
        client,
        paper,
        answer_sheet
    )

    print_report(results, total, max_total)