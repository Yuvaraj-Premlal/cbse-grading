from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
from sqlalchemy import create_engine, text
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from datetime import datetime, timedelta
import urllib, os, hashlib, hmac, uuid, json, base64, threading
from jose import jwt, JWTError

# Lazy imports — only loaded when needed so app starts even if not yet installed
def _get_blob_service_client():
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient

def _get_azure_openai():
    from openai import AzureOpenAI
    return AzureOpenAI

load_dotenv()
app = Flask(__name__)

# ── SECRETS ───────────────────────────────────────────
def get_secret_safe(client, name, fallback_env):
    """Read one secret from Key Vault, fall back to env var if missing."""
    try:
        return client.get_secret(name).value
    except Exception as e:
        print(f"Key Vault secret '{name}' failed: {e}")
        return os.getenv(fallback_env)

def get_secrets():
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(
            vault_url=os.getenv("KEY_VAULT_URL"),
            credential=credential
        )
        return {
            "db"           : get_secret_safe(client, "DB-CONNECTION-STRING",              "DB_CONNECTION_STRING"),
            "jwt"          : get_secret_safe(client, "JWT-SECRET",                        "JWT_SECRET"),
            "storage"      : get_secret_safe(client, "AZURE-STORAGE-CONNECTION-STRING",   "AZURE_STORAGE_CONNECTION_STRING"),
            "oai_endpoint" : get_secret_safe(client, "AZURE-OPENAI-ENDPOINT",             "AZURE_OPENAI_ENDPOINT"),
            "oai_key"      : get_secret_safe(client, "AZURE-OPENAI-KEY",                  "AZURE_OPENAI_KEY"),
            "oai_deploy"   : get_secret_safe(client, "AZURE-OPENAI-DEPLOYMENT",           "AZURE_OPENAI_DEPLOYMENT"),
        }
    except Exception as e:
        print(f"Key Vault connection failed entirely: {e}")
        return {
            "db"           : os.getenv("DB_CONNECTION_STRING"),
            "jwt"          : os.getenv("JWT_SECRET"),
            "storage"      : os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            "oai_endpoint" : os.getenv("AZURE_OPENAI_ENDPOINT"),
            "oai_key"      : os.getenv("AZURE_OPENAI_KEY"),
            "oai_deploy"   : os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        }

secrets = get_secrets()
JWT_SECRET    = secrets["jwt"]
JWT_ALGORITHM = "HS256"
JWT_EXPIRY    = 7  # days

# ── DATABASE ──────────────────────────────────────────
def get_engine():
    params = urllib.parse.quote_plus(secrets["db"])
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        pool_pre_ping=True
    )

# ── BLOB STORAGE ─────────────────────────────────────
BLOB_CONTAINER = "answer-sheets"

def get_blob_client():
    BlobServiceClient = _get_blob_service_client()
    return BlobServiceClient.from_connection_string(secrets["storage"])

def upload_answer_sheet(file_bytes, filename, content_type):
    """Upload file to Azure Blob, return blob name (not URL — use get_sas_url to read)."""
    from azure.storage.blob import ContentSettings
    import re
    blob_service  = get_blob_client()
    container     = blob_service.get_container_client(BLOB_CONTAINER)
    # Sanitize filename — replace spaces and special chars with underscores
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    blob_name     = f"{uuid.uuid4()}_{safe_filename}"
    container.upload_blob(
        blob_name, file_bytes,
        content_settings=ContentSettings(content_type=content_type)
    )
    # Return internal URL — not publicly accessible
    account_name = blob_service.account_name
    return f"https://{account_name}.blob.core.windows.net/{BLOB_CONTAINER}/{blob_name}"

def get_sas_url(blob_url, expiry_hours=2):
    """Generate a time-limited SAS URL for a private blob."""
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions
    from datetime import timezone
    import urllib.parse
    # Parse blob name from URL — decode any URL encoding first
    parts     = blob_url.split(f".blob.core.windows.net/{BLOB_CONTAINER}/")
    blob_name = urllib.parse.unquote(parts[1]) if len(parts) > 1 else blob_url
    blob_service = get_blob_client()
    # Get account key from connection string
    conn_str = secrets["storage"]
    key = dict(part.split("=", 1) for part in conn_str.split(";") if "=" in part).get("AccountKey", "")
    account_name = blob_service.account_name
    sas_token = generate_blob_sas(
        account_name   = account_name,
        container_name = BLOB_CONTAINER,
        blob_name      = blob_name,
        account_key    = key,
        permission     = BlobSasPermissions(read=True),
        expiry         = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
    )
    return f"{blob_url}?{sas_token}"

# ── OPENAI ────────────────────────────────────────────
def get_openai_client():
    AzureOpenAI = _get_azure_openai()
    return AzureOpenAI(
        azure_endpoint = secrets["oai_endpoint"],
        api_key        = secrets["oai_key"],
        api_version    = "2024-02-01"
    )

def wrap_latex(text):
    """
    Safety net: if GPT-4o returned raw LaTeX without $ delimiters,
    wrap continuous math expressions. Prompt instructs GPT-4o to use $
    delimiters — this handles cases where it doesn't comply.
    """
    if not text:
        return text
    import re
    # Already has $ delimiters — return as-is
    if '$' in text or r'\(' in text:
        return text
    # No LaTeX commands — plain English
    if not re.search(r'\\[a-zA-Z]', text):
        return text
    # Wrap sequences of LaTeX tokens: \cmd{...} possibly chained with operators
    # This matches things like: \int x \cdot e^x dx = x \cdot e^x - \int e^x dx + C
    result = re.sub(
        r'((?:\\[a-zA-Z]+(?:\{[^{}]*\})*'   # \command{optional args}
        r'|[a-zA-Z0-9](?:_|\^)\{[^{}]+\}'   # x_{sub} or x^{sup}
        r'|[a-zA-Z0-9](?:_|\^)[a-zA-Z0-9]'  # x_n or x^2
        r')(?:\s*(?:[=+\-/*]|\\[a-zA-Z]+(?:\{[^{}]*\})*'
        r'|[a-zA-Z0-9](?:_|\^)[{a-zA-Z0-9][^{}]*}?'
        r'|\d+(?:\.\d+)?)\s*)*)',
        lambda m: '$' + m.group(0).strip() + '$'
        if not m.group(0).strip().startswith('$') else m.group(0),
        text
    )
    return result


def grade_submission(questions, answer_sheet_urls):
    """
    Grade student answer sheet using GPT-4o vision — 2-pass approach.
    Pass 1: Grade with full prompt + images.
    Pass 2: Verify consistency, mathematical equivalence, alternate methods.
    Returns verified list of per-question results.
    """
    client = get_openai_client()

    # Build q_text — conditional model solution instruction
    q_text = ""
    for i, q in enumerate(questions, 1):
        model_sol = (q.get('model_solution') or '').strip()
        if model_sol:
            model_sol_text = f"Model Solution (one valid approach only — award full marks for any correct alternate method): {model_sol}"
        else:
            model_sol_text = "No model solution provided — solve this question independently before grading. Award full marks for any correct method that reaches the right answer."
        q_text += f"""
Q{i}. [{q['chapter']} · {q['max_marks']} marks]
Question: {q['latex_content']}
{model_sol_text}
---"""

    # ── PASS 1 — GRADE ────────────────────────────────────────────────────────
    p1_system = """You are an expert CBSE Mathematics and Science examiner with deep pedagogical knowledge.
You will be given a question paper and a student's handwritten answer sheet image.

GRADING PROCESS — follow this for every question:
1. Read the question carefully.
2. If a Model Solution is provided, use it to understand the correct final answer and one valid approach. If not, solve the question mentally.
3. Locate the student's answer for this question on the answer sheet.
4. Identify all logical steps in the solution BEFORE assigning marks.
5. Assign marks proportionally to each step based on importance.
6. Evaluate each step in order and award marks.
7. Deduct marks from the exact step where the first error occurs and all dependent steps after it.
8. Verify that the student has reached a complete and correct final answer.
--------------------------------------------------
STEP STRUCTURE RULE:
- Treat dependent results as a single step.
- If a question requires multiple values (e.g., two roots), the final answer step is correct ONLY if all required values are correct.
- Do not over-credit partially correct final results unless explicitly justified.
--------------------------------------------------
DEPENDENCY RULE:
- If an early mistake affects later steps, do NOT award marks for dependent steps even if they appear correct.
--------------------------------------------------
CONSTRAINT VALIDATION RULE:
- If the question includes constraints (e.g., positive integers, length must be positive):
  - The student MUST explicitly apply these constraints.
- If multiple solutions exist:
  - The student must identify and justify the valid solution(s).
- If constraint filtering is missing or only implied:
  - Deduct marks even if the final answer is correct.
--------------------------------------------------
WORKING CLARITY RULE:
- Solutions must be logically structured and readable.
- If the working contains excessive cancellations, overwriting, or unclear transitions:
  - Deduct up to 25% of marks.
- Clean step-by-step presentation is required for full marks.
--------------------------------------------------
STEP FLOW RULE:
- Each step must logically follow from the previous one.
- If steps are skipped, rewritten unclearly, or disorganized:
  - Deduct marks even if mathematically correct.
--------------------------------------------------
RECOVERY RULE:
- If the student makes incorrect or unclear attempts but later reaches the correct answer:
  - Do NOT award full marks.
  - Deduct marks for earlier incorrect or unclear steps.
--------------------------------------------------
SAME METHOD AS MODEL SOLUTION:
- Compare step-by-step with the model solution.
- Award marks for each correct step.
- Deduct from the exact step where the error first occurs.
--------------------------------------------------
ALTERNATE METHOD EVALUATION:
If the student uses a different method:
1. Evaluate it independently.
2. Check each step is mathematically valid.
3. Check it logically leads to the final answer.
4. Check final answer correctness.
5. If all are correct — award full marks.
6. If method is valid but unfamiliar — set ai_confidence < 0.75 and ai_flag_review = true.
7. If invalid — apply normal deductions.
--------------------------------------------------
NEUTRAL GRADE (ai_marks_awarded):
- Full marks ONLY if all steps correct, final answer correct, constraints correctly applied, working clear and structured.
- Partial credit for partially correct solutions.
- If final answer is incorrect, total marks should generally NOT exceed 50% unless strongly justified.
--------------------------------------------------
STRICT GRADE (ai_strict_marks):
- Wrong final answer = 0 marks for final answer step.
- Incomplete or unclear steps = 0 for that step.
- Correct method + correct final answer = full marks.
- MUST be less than ai_marks_awarded if final answer is wrong.
--------------------------------------------------
CONFIDENCE SCORE — start at 1.0 and subtract:
- 0.20 if handwriting unclear
- 0.15 if final answer wrong but partial marks awarded
- 0.10 if borderline marking
- 0.15 if wrong question attempted
- 0.10 if unfamiliar method used
- Minimum value is 0.10
--------------------------------------------------
IRRELEVANT ANSWER:
- If no relation to question, set ai_irrelevant = true.
- Skip concept, formula and calculation analysis.
- Still provide model solution.
--------------------------------------------------
LATEX RULES:
- Use LaTeX ONLY for mathematical expressions, equations, symbols and numbers within equations.
- Wrap inline math with single dollar signs: $expression$
- NEVER use LaTeX delimiters around English words or phrases.
- CORRECT: "The student correctly set up $x(x+1) = 156$"
- WRONG: "$positiveinteger$" or "$factorization$" or "$e.g.$"
--------------------------------------------------
OUTPUT:
Respond ONLY with a valid JSON array. No extra text, no markdown, no backticks."""

    p1_user = f"""Grade this student's answer sheet for the following questions:

{q_text}

For each question return a JSON object with EXACT fields:
- question_number: "Q1", "Q2" etc
- max_marks: integer
- ai_marks_awarded: integer (0 to max_marks) — NEUTRAL grade, the final score
- ai_strict_marks: integer (0 to max_marks) — STRICT grade
- ai_strict_reason: string — one sentence why strict < neutral, or "Strict and neutral agree"
- ai_irrelevant: boolean
- ai_concept: string — correct steps, exact step of mistake, wrong concept, correct concept. If alternate valid method, state it. English with LaTeX for math only.
- ai_formula: string — correct formula vs student formula. English with LaTeX for formulas only.
- ai_calculation: string — exact error line. If no errors, state "No calculation errors found". English with LaTeX for expressions only.
- ai_model_solution: string — complete correct solution. LaTeX for math only.
- ai_coaching_tip: string — one specific actionable tip. English with LaTeX for math only.
- ai_confidence: float 0 to 1 — per CONFIDENCE SCORE rules above
- ai_flag_review: boolean — true if confidence < 0.85, final answer wrong, irrelevant, or alternate method needs verification

Follow all grading rules strictly.
Return ONLY the JSON array, nothing else."""

    # Build image content for Pass 1
    image_contents = [{"type": "text", "text": p1_user}]
    for item in answer_sheet_urls:
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": item["url"], "detail": "high"}
        })

    p1_response = client.chat.completions.create(
        model       = secrets["oai_deploy"],
        messages    = [
            {"role": "system", "content": p1_system},
            {"role": "user",   "content": image_contents}
        ],
        max_tokens  = 4000,
        temperature = 0.1
    )

    raw = p1_response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    pass1_results = json.loads(raw.strip())
    print(f"Pass 1 complete — {len(pass1_results)} questions graded", flush=True)

    # ── PASS 2 — VERIFY ───────────────────────────────────────────────────────
    p2_system = """You are an expert CBSE examiner verifying a previous grading result.
DO NOT re-grade from scratch. Only verify and correct.
--------------------------------------------------
MATHEMATICAL EQUIVALENCE RULE:
- Answers are equivalent if mathematically identical, even if form differs.
- Use mathematical knowledge to determine equivalence, not textual matching.
- If unsure, set ai_flag_review = true rather than penalising the student.
--------------------------------------------------
ALTERNATE METHOD RULE:
- If student used a different method, evaluate their final answer independently.
- Do NOT compare against model solution steps.
- If final answer is mathematically correct, do not reduce marks.
- If uncertain, flag for teacher — do not reduce marks.
--------------------------------------------------
OUTPUT:
Return corrected JSON array only. No preamble, no markdown, no backticks."""

    p2_user = f"""You previously graded a student's answer sheet and returned this result:

{json.dumps(pass1_results, indent=2)}

The original questions were:
{q_text}

Verify each question by checking ALL steps below in order:

STEP 1 — FINAL ANSWER CHECK:
- What is the final stated answer in the model solution?
- What is the final stated answer the student wrote — not intermediate steps, the actual conclusion?
- Are these mathematically equivalent?
- If student only reached an intermediate result and did not state the final conclusion — answer is incomplete, reduce ai_marks_awarded.
- If not equivalent and full marks awarded — reduce ai_marks_awarded.
- If alternate method used — evaluate final answer independently, not against model solution steps.

STEP 2 — CONSTRAINT CHECK:
- Did the question include any constraints (e.g., positive integers, real values, domain restrictions)?
- Did the student explicitly apply and state these constraints?
- If constraints are missing or only implied — reduce marks.

STEP 3 — OVER-GRADING CHECK:
- If full marks were awarded, verify ALL of the following:
  - All steps are correct
  - Working is clear and structured
  - No missing reasoning steps
  - Constraints were applied
- If any of the above fail — reduce marks.

STEP 4 — INTERNAL CONSISTENCY CHECK:
- Does ai_concept feedback describe the student getting the right answer?
- Does ai_calculation mention errors?
- Does ai_model_solution show a different answer than what was credited?
- If feedback contradicts marks — correct marks to match feedback.

STEP 5 — STRICT MARKS CHECK:
- If final answer is wrong, ai_strict_marks MUST be less than ai_marks_awarded. Fix if violated.
- If final answer is correct, ai_strict_marks can equal ai_marks_awarded.
- Verify ai_strict_reason accurately reflects the difference.

STEP 6 — CONFIDENCE AND FLAG UPDATE:
- If any marks were changed in steps 1-5, subtract 0.15 from ai_confidence.
- Clamp ai_confidence between 0.10 and 1.0.
- Set ai_flag_review = true if:
  - marks changed
  - confidence < 0.85
  - final answer wrong
  - alternate method used needing teacher confirmation

Return the verified and corrected JSON array with the exact same field structure as the input.
Only update fields that need correction.
Return ONLY the JSON array, nothing else."""

    p2_response = client.chat.completions.create(
        model       = secrets["oai_deploy"],
        messages    = [
            {"role": "system", "content": p2_system},
            {"role": "user",   "content": p2_user}
        ],
        max_tokens  = 4000,
        temperature = 0.1
    )

    raw2 = p2_response.choices[0].message.content.strip()
    if raw2.startswith("```"):
        raw2 = raw2.split("```")[1]
        if raw2.startswith("json"):
            raw2 = raw2[4:]
    pass2_results = json.loads(raw2.strip())
    print(f"Pass 2 complete — verification done", flush=True)

    return pass2_results

def calculate_grade(percentage):
    if percentage >= 90: return 'A+'
    if percentage >= 75: return 'A'
    if percentage >= 60: return 'B+'
    if percentage >= 50: return 'B'
    if percentage >= 40: return 'C'
    if percentage >= 33: return 'D'
    return 'F'

def run_grading_async(submission_id, assignment_id, student_id, questions, answer_sheet_urls):
    """Run AI grading in background thread. answer_sheet_urls is list of {url, filename}."""
    try:
        engine = get_engine()

        # Generate SAS URLs for all images so GPT-4o can read private blobs
        sas_urls = []
        for item in answer_sheet_urls:
            sas_url = get_sas_url(item["url"], expiry_hours=1)
            sas_urls.append({"url": sas_url, "filename": item["filename"]})
        print(f"Grading {len(sas_urls)} image(s)", flush=True)

        # Grade with GPT-4o
        results = grade_submission(questions, sas_urls)

        total_awarded = 0
        total_max     = sum(q['max_marks'] for q in questions)

        with engine.begin() as conn:
            for i, r in enumerate(results):
                # Match by question_number OR by position
                q_match = next((q for q in questions
                    if str(q['question_number']) == str(r.get('question_number'))), None)
                # Fallback: match by index position
                if not q_match and i < len(questions):
                    q_match = questions[i]

                if not q_match:
                    print(f"No match for {r.get('question_number')} — skipping", flush=True)
                    continue

                question_id = q_match['question_id']
                marks = min(r.get('ai_marks_awarded', 0), q_match['max_marks'])
                total_awarded += marks

                # Safeguards for all fields
                strict_marks = min(max(int(r.get('ai_strict_marks', marks)), 0), q_match['max_marks'])
                confidence   = min(max(float(r.get('ai_confidence', 0.8)), 0.0), 1.0)
                irrelevant   = str(r.get('ai_irrelevant', 'false')).lower() in ('true', '1')
                flag         = bool(r.get('ai_flag_review', False)) or confidence < 0.85

                conn.execute(text("""
                    INSERT INTO submission_questions
                        (sq_id, submission_id, question_id, question_number,
                         max_marks, ai_marks_awarded, ai_step_breakdown,
                         ai_strength, ai_weakness, ai_model_solution,
                         ai_coaching_tip, ai_confidence, ai_flag_review,
                         ai_strict_marks, ai_strict_reason,
                         ai_concept, ai_formula, ai_calculation, ai_irrelevant)
                    VALUES
                        (NEWID(),
                         CAST(:sid AS UNIQUEIDENTIFIER),
                         CAST(:qid AS UNIQUEIDENTIFIER),
                         :qnum, :max_marks, :awarded,
                         :breakdown, :strength, :weakness,
                         :model_sol, :tip,
                         :confidence, :flag,
                         :strict_marks, :strict_reason,
                         :concept, :formula, :calculation, :irrelevant)
                """), {
                    "sid"          : str(submission_id),
                    "qid"          : str(question_id),
                    "qnum"         : r.get('question_number'),
                    "max_marks"    : q_match['max_marks'],
                    "awarded"      : marks,
                    "breakdown"    : r.get('ai_step_breakdown', ''),
                    "strength"     : r.get('ai_strength', ''),
                    "weakness"     : r.get('ai_weakness', ''),
                    "model_sol"    : wrap_latex(r.get('ai_model_solution', '')),
                    "tip"          : wrap_latex(r.get('ai_coaching_tip', '')),
                    "confidence"   : confidence,
                    "flag"         : flag,
                    "strict_marks" : strict_marks,
                    "strict_reason": r.get('ai_strict_reason', ''),
                    "concept"      : wrap_latex(r.get('ai_concept', '')),
                    "formula"      : wrap_latex(r.get('ai_formula', '')),
                    "calculation"  : wrap_latex(r.get('ai_calculation', '')),
                    "irrelevant"   : irrelevant
                })

            pct   = round((total_awarded / total_max * 100), 2) if total_max > 0 else 0
            grade = calculate_grade(pct)

            # Update submission
            conn.execute(text("""
                UPDATE submissions SET
                    total_awarded = :awarded,
                    total_max     = :total_max,
                    percentage    = :pct,
                    grade         = :grade,
                    graded_at     = GETUTCDATE()
                WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
            """), {
                "awarded"  : total_awarded,
                "total_max": total_max,
                "pct"      : pct,
                "grade"    : grade,
                "sid"      : str(submission_id)
            })

            # Update assignment status
            conn.execute(text("""
                UPDATE assignments SET status = 'graded'
                WHERE assignment_id = CAST(:aid AS UNIQUEIDENTIFIER)
            """), {"aid": str(assignment_id)})

        print(f"✅ Grading complete: submission {submission_id} — {total_awarded}/{total_max} ({pct}%)")

    except Exception as e:
        print(f"❌ Grading failed for submission {submission_id}: {e}")
        try:
            engine = get_engine()
            with engine.begin() as conn:
                # Mark as ai_failed — keeps status 'submitted' so teacher can review manually
                conn.execute(text("""
                    UPDATE submissions SET
                        ai_results = 'ai_failed'
                    WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
                """), {"sid": str(submission_id)})
        except:
            pass

# ── PASSWORD ──────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(password.encode()).hexdigest(),
        hashed
    )

# ── JWT ───────────────────────────────────────────────
def create_token(user_id: int, name: str, email: str, role: str) -> str:
    payload = {
        "sub"   : str(user_id),
        "name"  : name,
        "email" : email,
        "role"  : role,
        "exp"   : datetime.utcnow() + timedelta(days=JWT_EXPIRY)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None

def get_current_user(req) -> dict | None:
    token = req.cookies.get("skooltest_token")
    if not token:
        return None
    return decode_token(token)

# ── AUTH GUARD ────────────────────────────────────────
def require_role(*roles):
    def decorator(f):
        from functools import wraps
        @wraps(f)
        def wrapper(*args, **kwargs):
            user = get_current_user(request)
            if not user:
                return redirect(url_for('index'))
            if roles and user.get("role") not in roles:
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return wrapper
    return decorator

# ── PAGES ─────────────────────────────────────────────
@app.route("/")
def index():
    user = get_current_user(request)
    if user:
        role = user.get("role")
        if role == "teacher":
            return redirect(url_for('teacher'))
        elif role == "admin":
            return redirect(url_for('admin'))
        else:
            return redirect(url_for('student'))
    return render_template("login.html")

@app.route("/admin")
@require_role("admin")
def admin():
    user = get_current_user(request)
    return render_template("admin.html", user=user)

@app.route("/student")
@require_role("student")
def student():
    user = get_current_user(request)
    return render_template("student.html", user=user)

@app.route("/teacher")
@require_role("teacher")
def teacher():
    user = get_current_user(request)
    return render_template("teacher.html", user=user)

@app.route("/health-check")
def health_check():
    return render_template("index.html")

# ── AUTH ENDPOINTS ────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if not data.get("email") or not data.get("password"):
        return jsonify({"ok": False, "error": "Email and password required"})
    try:
        engine = get_engine()
        with engine.connect() as conn:
            user = conn.execute(text("""
                SELECT user_id, name, email, password_hash, role, is_active
                FROM users WHERE email = :email
            """), {"email": data["email"]}).fetchone()

            if not user:
                return jsonify({"ok": False, "error": "Email not found"})
            if not user.is_active:
                return jsonify({"ok": False, "error": "Account deactivated"})
            if not verify_password(data["password"], user.password_hash):
                return jsonify({"ok": False, "error": "Wrong password"})

            token = create_token(user.user_id, user.name, user.email, user.role)

            resp = make_response(jsonify({
                "ok": True,
                "message": f"Welcome {user.name}",
                "user": {
                    "name"  : user.name,
                    "email" : user.email,
                    "role"  : user.role
                }
            }))
            resp.set_cookie(
                "skooltest_token",
                token,
                max_age   = 60 * 60 * 24 * 7,
                httponly  = True,
                samesite  = "Lax",
                secure    = True
            )
            return resp

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})

@app.route("/logout")
def logout():
    resp = make_response(redirect(url_for('index')))
    resp.delete_cookie("skooltest_token")
    return resp

# ── REGISTER ──────────────────────────────────────────
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    try:
        if not all([data.get("name"), data.get("email"), data.get("password"), data.get("role")]):
            return jsonify({"ok": False, "error": "All fields required"})
        if data["role"] not in ["student", "teacher", "admin"]:
            return jsonify({"ok": False, "error": "Invalid role"})

        engine = get_engine()
        with engine.begin() as conn:
            existing = conn.execute(text(
                "SELECT user_id FROM users WHERE email = :email"
            ), {"email": data["email"]}).fetchone()
            if existing:
                return jsonify({"ok": False, "error": "Email already exists"})

            conn.execute(text("""
                INSERT INTO users (name, email, password_hash, role)
                VALUES (:name, :email, :hash, :role)
            """), {
                "name"  : data["name"],
                "email" : data["email"],
                "hash"  : hash_password(data["password"]),
                "role"  : data["role"]
            })
        return jsonify({"ok": True, "message": f"User {data['email']} created as {data['role']}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})

# ── API — CURRENT USER ─────────────────────────────────
@app.route("/api/me")
def me():
    user = get_current_user(request)
    if not user:
        return jsonify({"ok": False, "error": "Not authenticated"}), 401
    return jsonify({"ok": True, "user": user})

# ── HEALTH CHECK API ───────────────────────────────────
@app.route("/health")
def health():
    results = {}

    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=os.getenv("KEY_VAULT_URL"), credential=credential)
        client.get_secret("JWT-SECRET")
        results["keyvault"] = {"ok": True, "status": "Connected"}
    except Exception as e:
        results["keyvault"] = {"ok": False, "status": str(e)[:120]}

    try:
        engine = get_engine()
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM users")).fetchone()[0]
            results["database"] = {"ok": True, "status": f"Connected — {count} users in DB"}
    except Exception as e:
        results["database"] = {"ok": False, "status": str(e)[:120]}

    try:
        engine = get_engine()
        with engine.connect() as conn:
            tables = conn.execute(text("""
                SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME
            """)).fetchall()
            results["tables"] = {
                "ok": True,
                "status": f"{len(tables)} tables found",
                "list": [t[0] for t in tables]
            }
    except Exception as e:
        results["tables"] = {"ok": False, "status": str(e)[:120]}

    return jsonify(results)

# ── LIST USERS ────────────────────────────────────────
@app.route("/users")
def list_users():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            users = conn.execute(text("""
                SELECT name, email, role, is_active,
                       CONVERT(VARCHAR, created_at, 120) as created_at
                FROM users ORDER BY created_at DESC
            """)).fetchall()
            return jsonify({"ok": True, "users": [dict(u._mapping) for u in users]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})

@app.route("/admin/set-role", methods=["POST"])
def set_role():
    data = request.json
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE users SET role = :role WHERE email = :email
            """), {"role": data["role"], "email": data["email"]})
        return jsonify({"ok": True, "message": f"Updated {data['email']} to {data['role']}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ── TEACHER DASHBOARD API ─────────────────────────────
@app.route("/api/teacher/dashboard")
@require_role("teacher")
def teacher_dashboard():
    user = get_current_user(request)
    try:
        engine = get_engine()
        with engine.connect() as conn:

            pending = conn.execute(text("""
                SELECT COUNT(*) FROM assignments WHERE status = 'graded'
            """)).fetchone()[0]

            disputes = conn.execute(text("""
                SELECT COUNT(*) FROM disputes WHERE status = 'open'
            """)).fetchone()[0]

            urgent = conn.execute(text("""
                SELECT COUNT(*) FROM disputes
                WHERE status = 'open'
                AND DATEDIFF(hour, raised_at, GETUTCDATE()) >= 40
            """)).fetchone()[0]

            students = conn.execute(text("""
                SELECT COUNT(DISTINCT student_id) FROM assignments
                WHERE MONTH(created_at) = MONTH(GETUTCDATE())
                AND YEAR(created_at) = YEAR(GETUTCDATE())
            """)).fetchone()[0]

            active_exams = conn.execute(text("""
                SELECT COUNT(DISTINCT paper_id) FROM assignments
                WHERE due_date >= GETUTCDATE()
                AND status NOT IN ('released')
            """)).fetchone()[0]

            pending_subs = conn.execute(text("""
                SELECT TOP 10
                    u.name     AS student_name,
                    p.title    AS paper_title,
                    p.total_marks,
                    s.total_awarded AS ai_score,
                    s.submission_id
                FROM assignments a
                JOIN students st   ON a.student_id  = st.student_id
                JOIN users u       ON st.user_id    = u.user_id
                JOIN papers p      ON a.paper_id    = p.paper_id
                LEFT JOIN submissions s ON a.assignment_id = s.assignment_id
                WHERE a.status = 'graded'
                ORDER BY a.created_at ASC
            """)).fetchall()

            active_asgn = conn.execute(text("""
                SELECT
                    p.title,
                    p.total_marks,
                    MAX(a.due_date)                                       AS due_date,
                    COUNT(DISTINCT a.student_id)                          AS student_count,
                    COUNT(DISTINCT s.submission_id)                       AS submitted,
                    COUNT(DISTINCT CASE WHEN a.status IN ('graded','released')
                                   THEN a.assignment_id END)              AS graded
                FROM papers p
                JOIN assignments a   ON p.paper_id      = a.paper_id
                LEFT JOIN submissions s ON a.assignment_id = s.assignment_id
                WHERE a.status NOT IN ('released')
                GROUP BY p.paper_id, p.title, p.total_marks
                ORDER BY due_date DESC
            """)).fetchall()

        return jsonify({
            "ok": True,
            "dashboard": {
                "name"               : user["name"],
                "pending_review"     : pending,
                "open_disputes"      : disputes,
                "urgent_disputes"    : urgent,
                "total_students"     : students,
                "active_exams"       : active_exams,
                "pending_submissions": [dict(r._mapping) for r in pending_subs],
                "active_assignments" : [
                    {**dict(r._mapping), "due_date": str(r.due_date)}
                    for r in active_asgn
                ]
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})

# ── QUESTION BANK ─────────────────────────────────────
@app.route("/api/teacher/questions", methods=["GET"])
@require_role("teacher")
def get_questions():
    subject    = request.args.get("subject", "")
    chapter    = request.args.get("chapter", "")
    difficulty = request.args.get("difficulty", "")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            query = """
                SELECT
                    q.question_id,
                    q.latex_content,
                    q.subject,
                    q.chapter,
                    q.difficulty,
                    q.max_marks,
                    q.type,
                    q.model_solution,
                    q.source,
                    q.year,
                    q.approved,
                    CONVERT(VARCHAR, q.created_at, 120) as created_at
                FROM questions q
                WHERE 1=1
            """
            params = {}
            if subject:
                query += " AND q.subject = :subject"
                params["subject"] = subject
            if chapter:
                query += " AND q.chapter LIKE :chapter"
                params["chapter"] = f"%{chapter}%"
            if difficulty:
                query += " AND q.difficulty = :difficulty"
                params["difficulty"] = difficulty
            query += " ORDER BY q.created_at DESC"

            rows = conn.execute(text(query), params).fetchall()
            return jsonify({
                "ok": True,
                "questions": [dict(r._mapping) for r in rows],
                "total": len(rows)
            })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/teacher/questions", methods=["POST"])
@require_role("teacher")
def save_question():
    user = get_current_user(request)
    data = request.json
    try:
        required = ["latex_content", "subject", "chapter", "class_num", "difficulty", "type", "max_marks", "source"]
        if not all(data.get(f) for f in required):
            return jsonify({"ok": False, "error": "All fields required"})
        if data["difficulty"] not in ["easy", "medium", "hard", "very_hard"]:
            return jsonify({"ok": False, "error": "Invalid difficulty"})
        if data["source"] not in ["past_paper", "teacher"]:
            return jsonify({"ok": False, "error": "Invalid source"})

        engine = get_engine()

        # Step 1 — get user_id from users table using email from JWT
        with engine.connect() as conn:
            user_row = conn.execute(text(
                "SELECT user_id FROM users WHERE email = :email"
            ), {"email": user["email"]}).fetchone()

        if not user_row:
            return jsonify({"ok": False, "error": "User not found in DB"})

        user_uuid = user_row[0]

        # Step 2 — find teacher profile by user_id
        with engine.connect() as conn:
            teacher = conn.execute(text(
                "SELECT teacher_id FROM teachers WHERE user_id = :uid"
            ), {"uid": user_uuid}).fetchone()

        # Step 3 — auto-create teacher profile if missing
        if not teacher:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO teachers (user_id, subject)
                    VALUES (:uid, :subject)
                """), {"uid": user_uuid, "subject": data["subject"]})
            with engine.connect() as conn:
                teacher = conn.execute(text(
                    "SELECT teacher_id FROM teachers WHERE user_id = :uid"
                ), {"uid": user_uuid}).fetchone()

        # Step 4 — insert question
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO questions
                    (latex_content, subject, chapter, class, difficulty,
                     type, max_marks, source, year, created_by, approved,
                     model_solution)
                VALUES
                    (:content, :subject, :chapter, :class, :difficulty,
                     :type, :marks, :source, :year, :created_by, 1,
                     :model_solution)
            """), {
                "content"        : data["latex_content"],
                "subject"        : data["subject"],
                "chapter"        : data["chapter"],
                "class"          : int(data.get("class_num", 12)),
                "difficulty"     : data["difficulty"],
                "type"           : data["type"],
                "marks"          : int(data["max_marks"]),
                "source"         : data["source"],
                "year"           : data.get("year") or None,
                "created_by"     : str(teacher[0]),
                "model_solution" : data.get("model_solution") or None
            })
        return jsonify({"ok": True, "message": "Question saved"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})

# ── PAPERS API ────────────────────────────────────────
@app.route("/api/teacher/papers", methods=["GET"])
@require_role("teacher")
def get_papers():
    user = get_current_user(request)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Get teacher's user_id
            user_row = conn.execute(text(
                "SELECT user_id FROM users WHERE email = :email"
            ), {"email": user["email"]}).fetchone()
            if not user_row:
                return jsonify({"ok": False, "error": "User not found"})

            teacher_row = conn.execute(text(
                "SELECT teacher_id FROM teachers WHERE user_id = :uid"
            ), {"uid": str(user_row[0])}).fetchone()
            if not teacher_row:
                return jsonify({"ok": False, "error": "Teacher profile not found"})

            rows = conn.execute(text("""
                SELECT
                    p.paper_id, p.title, p.subject, p.class,
                    p.total_marks, p.duration_minutes, p.is_active,
                    CONVERT(VARCHAR, p.created_at, 120) as created_at,
                    COUNT(pq.question_id) as question_count
                FROM papers p
                LEFT JOIN paper_questions pq ON p.paper_id = pq.paper_id
                WHERE p.created_by = :tid
                GROUP BY p.paper_id, p.title, p.subject, p.class,
                         p.total_marks, p.duration_minutes, p.is_active, p.created_at
                ORDER BY p.created_at DESC
            """), {"tid": str(teacher_row[0])}).fetchall()

        return jsonify({"ok": True, "papers": [dict(r._mapping) for r in rows]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/teacher/papers/<paper_id>", methods=["GET"])
@require_role("teacher")
def get_paper(paper_id):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            paper = conn.execute(text("""
                SELECT paper_id, title, subject, class, total_marks,
                       duration_minutes, marking_scheme, is_active
                FROM papers WHERE paper_id = :pid
            """), {"pid": paper_id}).fetchone()

            if not paper:
                return jsonify({"ok": False, "error": "Paper not found"})

            questions = conn.execute(text("""
                SELECT
                    pq.question_id, pq.section, pq.order_num, pq.marks_override,
                    q.latex_content, q.subject, q.chapter, q.difficulty,
                    q.max_marks, q.type, q.model_solution
                FROM paper_questions pq
                JOIN questions q ON pq.question_id = q.question_id
                WHERE pq.paper_id = :pid
                ORDER BY pq.section, pq.order_num
            """), {"pid": paper_id}).fetchall()

        result = dict(paper._mapping)
        result["questions"] = [dict(q._mapping) for q in questions]
        return jsonify({"ok": True, "paper": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/teacher/papers", methods=["POST"])
@require_role("teacher")
def create_paper():
    user = get_current_user(request)
    data = request.json
    try:
        required = ["title", "subject", "class_num", "duration_minutes", "total_marks"]
        if not all(data.get(f) for f in required):
            return jsonify({"ok": False, "error": "Required fields missing"})

        engine = get_engine()

        with engine.connect() as conn:
            user_row = conn.execute(text(
                "SELECT user_id FROM users WHERE email = :email"
            ), {"email": user["email"]}).fetchone()
        if not user_row:
            return jsonify({"ok": False, "error": "User not found"})

        with engine.connect() as conn:
            teacher_row = conn.execute(text(
                "SELECT teacher_id FROM teachers WHERE user_id = :uid"
            ), {"uid": str(user_row[0])}).fetchone()
        if not teacher_row:
            return jsonify({"ok": False, "error": "Teacher profile not found"})

        with engine.begin() as conn:
            # Insert paper and get generated ID
            conn.execute(text("""
                INSERT INTO papers
                    (title, subject, class, total_marks, duration_minutes,
                     marking_scheme, created_by, is_active, is_locked)
                VALUES
                    (:title, :subject, :class, :marks, :duration,
                     :scheme, :tid, :active, 0)
            """), {
                "title"    : data["title"],
                "subject"  : data["subject"],
                "class"    : data["class_num"],
                "marks"    : data["total_marks"],
                "duration" : data["duration_minutes"],
                "scheme"   : data.get("instructions") or None,
                "tid"      : str(teacher_row[0]),
                "active"   : data.get("is_active", 0)
            })

            paper_id = conn.execute(text("""
                SELECT TOP 1 paper_id FROM papers
                WHERE created_by = :tid ORDER BY created_at DESC
            """), {"tid": str(teacher_row[0])}).fetchone()[0]

            # Insert paper_questions
            print("DEBUG questions:", data.get("questions", []), flush=True)
            for q in data.get("questions", []):
                conn.execute(text("""
                    INSERT INTO paper_questions
                        (paper_id, question_id, order_num, section)
                    VALUES (CAST(:pid AS UNIQUEIDENTIFIER), CAST(:qid AS UNIQUEIDENTIFIER), :order, :section)
                """), {
                    "pid"     : str(paper_id),
                    "qid"     : str(q["question_id"]),
                    "order"   : q["order_num"],
                    "section" : q["section"]
                })

        return jsonify({"ok": True, "paper_id": str(paper_id)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/teacher/papers/<paper_id>", methods=["PUT"])
@require_role("teacher")
def update_paper(paper_id):
    data = request.json
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE papers SET
                    title            = :title,
                    subject          = :subject,
                    class            = :class,
                    total_marks      = :marks,
                    duration_minutes = :duration,
                    marking_scheme   = :scheme,
                    is_active        = :active
                WHERE paper_id = :pid
            """), {
                "title"    : data["title"],
                "subject"  : data["subject"],
                "class"    : data["class_num"],
                "marks"    : data["total_marks"],
                "duration" : data["duration_minutes"],
                "scheme"   : data.get("instructions") or None,
                "active"   : data.get("is_active", 0),
                "pid"      : paper_id
            })

            # Replace all questions
            conn.execute(text("DELETE FROM paper_questions WHERE paper_id = CAST(:pid AS UNIQUEIDENTIFIER)"), {"pid": str(paper_id)})
            for q in data.get("questions", []):
                conn.execute(text("""
                    INSERT INTO paper_questions
                        (paper_id, question_id, order_num, section)
                    VALUES (CAST(:pid AS UNIQUEIDENTIFIER), CAST(:qid AS UNIQUEIDENTIFIER), :order, :section)
                """), {
                    "pid"     : str(paper_id),
                    "qid"     : str(q["question_id"]),
                    "order"   : q["order_num"],
                    "section" : q["section"]
                })

        return jsonify({"ok": True, "paper_id": paper_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/teacher/papers/<paper_id>", methods=["DELETE"])
@require_role("teacher")
def delete_paper(paper_id):
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM paper_questions WHERE paper_id = :pid"), {"pid": paper_id})
            conn.execute(text("DELETE FROM papers WHERE paper_id = :pid"), {"pid": paper_id})
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})

# ── DEBUG ─────────────────────────────────────────────
@app.route("/admin/papers-schema")
def papers_schema():
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'papers'
            ORDER BY ORDINAL_POSITION
        """)).fetchall()
    return jsonify([dict(r._mapping) for r in rows])

@app.route("/admin/teachers-schema")
def teachers_schema():
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'teachers'
        """)).fetchall()
    return jsonify([dict(r._mapping) for r in rows])

# ── UPDATE QUESTION ───────────────────────────────────
@app.route("/api/teacher/questions/<question_id>", methods=["PUT"])
@require_role("teacher")
def update_question(question_id):
    data = request.json
    try:
        required = ["latex_content", "subject", "chapter", "class_num", "difficulty", "type", "max_marks", "source"]
        if not all(data.get(f) for f in required):
            return jsonify({"ok": False, "error": "All fields required"})
        if data["difficulty"] not in ["easy", "medium", "hard", "very_hard"]:
            return jsonify({"ok": False, "error": "Invalid difficulty"})
        if data["source"] not in ["past_paper", "teacher"]:
            return jsonify({"ok": False, "error": "Invalid source"})

        engine = get_engine()
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE questions SET
                    latex_content  = :content,
                    subject        = :subject,
                    chapter        = :chapter,
                    class          = :class,
                    difficulty     = :difficulty,
                    type           = :type,
                    max_marks      = :marks,
                    source         = :source,
                    year           = :year,
                    model_solution = :model_solution
                WHERE question_id = :qid
            """), {
                "content"        : data["latex_content"],
                "subject"        : data["subject"],
                "chapter"        : data["chapter"],
                "class"          : int(data.get("class_num", 12)),
                "difficulty"     : data["difficulty"],
                "type"           : data["type"],
                "marks"          : int(data["max_marks"]),
                "source"         : data["source"],
                "year"           : data.get("year") or None,
                "model_solution" : data.get("model_solution") or None,
                "qid"            : question_id
            })
        return jsonify({"ok": True, "message": "Question updated"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})

# ── DELETE QUESTION ───────────────────────────────────
@app.route("/api/teacher/questions/<question_id>", methods=["DELETE"])
@require_role("teacher")
def delete_question(question_id):
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text(
                "DELETE FROM questions WHERE question_id = :qid"
            ), {"qid": question_id})
        return jsonify({"ok": True, "message": "Question deleted"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})

# ══════════════════════════════════════════════════════
# STUDENT API
# ══════════════════════════════════════════════════════

def generate_system_reg_number(conn):
    """Generate unique ever-incrementing registration number: SKT-YYYY-NNNN"""
    year = datetime.now().year
    row  = conn.execute(text("""
        SELECT MAX(CAST(SUBSTRING(system_reg_number, 10, 4) AS INT))
        FROM students
        WHERE system_reg_number LIKE 'SKT-____-____'
    """)).fetchone()
    last_seq = row[0] if row and row[0] else 0
    seq      = last_seq + 1
    return f"SKT-{year}-{seq:04d}"

def get_student_id(conn, user):
    """Resolve user email -> user_id -> student_id. Auto-creates student record if missing."""
    user_row = conn.execute(text(
        "SELECT user_id FROM users WHERE email = :email"
    ), {"email": user["email"]}).fetchone()
    if not user_row:
        return None, None

    user_id = str(user_row[0])

    student_row = conn.execute(text(
        "SELECT student_id FROM students WHERE user_id = CAST(:uid AS UNIQUEIDENTIFIER)"
    ), {"uid": user_id}).fetchone()

    if not student_row:
        # Auto-create student record using a new write connection
        engine = get_engine()
        with engine.begin() as write_conn:
            reg_num = generate_system_reg_number(write_conn)
            write_conn.execute(text("""
                INSERT INTO students (student_id, user_id, class, system_reg_number)
                VALUES (NEWID(), CAST(:uid AS UNIQUEIDENTIFIER), 12, :reg_num)
            """), {"uid": user_id, "reg_num": reg_num})
        student_row = conn.execute(text(
            "SELECT student_id FROM students WHERE user_id = CAST(:uid AS UNIQUEIDENTIFIER)"
        ), {"uid": user_id}).fetchone()

    return user_id, str(student_row[0]) if student_row else None


@app.route("/api/student/dashboard")
@require_role("student")
def student_dashboard():
    user = get_current_user(request)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            user_id, student_id = get_student_id(conn, user)
            if not student_id:
                return jsonify({"ok": False, "error": "Student profile not found"})

            # 1. Pending exams — assigned, not yet submitted, not overdue
            pending = conn.execute(text("""
                SELECT
                    a.assignment_id,
                    a.due_date,
                    a.status,
                    p.paper_id,
                    p.title,
                    p.subject,
                    p.class,
                    p.total_marks,
                    p.duration_minutes,
                    CONVERT(VARCHAR, a.due_date, 120) as due_date_str
                FROM assignments a
                JOIN papers p ON a.paper_id = p.paper_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND a.status = 'assigned'
                ORDER BY a.due_date ASC
            """), {"sid": student_id}).fetchall()

            # 2. Submitted — awaiting review/grading
            under_review = conn.execute(text("""
                SELECT
                    a.assignment_id,
                    a.status,
                    p.title,
                    p.subject,
                    p.total_marks,
                    s.submission_id,
                    CONVERT(VARCHAR, s.submitted_at, 120) as submitted_at
                FROM assignments a
                JOIN papers p ON a.paper_id = p.paper_id
                LEFT JOIN submissions s ON a.assignment_id = s.assignment_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND a.status NOT IN ('released')
                ORDER BY s.submitted_at DESC
            """), {"sid": student_id}).fetchall()

            # 3. Released results
            released = conn.execute(text("""
                SELECT
                    a.assignment_id,
                    p.title,
                    p.subject,
                    p.total_marks,
                    s.total_awarded,
                    s.percentage,
                    s.grade,
                    s.submission_id,
                    CONVERT(VARCHAR, s.released_at, 120) as released_at
                FROM assignments a
                JOIN papers p ON a.paper_id = p.paper_id
                JOIN submissions s ON a.assignment_id = s.assignment_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND a.status = 'released'
                AND s.final_released = 1
                ORDER BY s.released_at DESC
            """), {"sid": student_id}).fetchall()

            # 4. Stats
            total_assigned = conn.execute(text("""
                SELECT COUNT(*) FROM assignments
                WHERE student_id = CAST(:sid AS UNIQUEIDENTIFIER)
            """), {"sid": student_id}).fetchone()[0]

            total_submitted = conn.execute(text("""
                SELECT COUNT(*) FROM assignments
                WHERE student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND status IN ('submitted', 'graded', 'released')
            """), {"sid": student_id}).fetchone()[0]

            avg_score = conn.execute(text("""
                SELECT AVG(CAST(s.percentage AS FLOAT))
                FROM submissions s
                JOIN assignments a ON s.assignment_id = a.assignment_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND s.final_released = 1
            """), {"sid": student_id}).fetchone()[0]

            open_disputes = conn.execute(text("""
                SELECT COUNT(*) FROM disputes
                WHERE student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND status = 'open'
            """), {"sid": student_id}).fetchone()[0]

            # Practice attempts today
            from datetime import date as date_cls
            practice_today = conn.execute(text("""
                SELECT COUNT(*) FROM practice_attempts
                WHERE student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND CAST(attempted_at AS DATE) = :today
            """), {"sid": student_id, "today": date_cls.today()}).fetchone()[0]

        return jsonify({
            "ok": True,
            "dashboard": {
                "name"           : user["name"],
                "student_id"     : student_id,
                "pending_exams"  : [dict(r._mapping) for r in pending],
                "under_review"   : [dict(r._mapping) for r in under_review],
                "released"       : [dict(r._mapping) for r in released],
                "stats": {
                    "total_assigned"   : total_assigned,
                    "total_submitted"  : total_submitted,
                    "avg_score"        : round(float(avg_score), 1) if avg_score else None,
                    "open_disputes"    : open_disputes,
                    "practice_today"   : practice_today,
                    "practice_remaining": max(0, 3 - practice_today)
                }
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/student/paper/<paper_id>")
@require_role("student")
def get_student_paper(paper_id):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            paper = conn.execute(text("""
                SELECT paper_id, title, subject, class, total_marks,
                       duration_minutes, marking_scheme
                FROM papers WHERE paper_id = CAST(:pid AS UNIQUEIDENTIFIER)
                AND is_active = 1
            """), {"pid": paper_id}).fetchone()

            if not paper:
                return jsonify({"ok": False, "error": "Paper not found"})

            questions = conn.execute(text("""
                SELECT pq.order_num, pq.section,
                       q.question_id, q.latex_content, q.chapter,
                       q.difficulty, q.type, q.max_marks
                FROM paper_questions pq
                JOIN questions q ON pq.question_id = q.question_id
                WHERE pq.paper_id = CAST(:pid AS UNIQUEIDENTIFIER)
                ORDER BY pq.section, pq.order_num
            """), {"pid": paper_id}).fetchall()

        result = dict(paper._mapping)
        result["questions"] = [dict(q._mapping) for q in questions]
        return jsonify({"ok": True, "paper": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


# ── UPLOAD ANSWER SHEET ───────────────────────────────
@app.route("/api/student/upload", methods=["POST"])
@require_role("student")
def upload_answer():
    try:
        files = request.files.getlist('files') or request.files.getlist('files[]')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"ok": False, "error": "No files provided"})

        allowed = {'jpg', 'jpeg', 'png', 'pdf'}
        uploaded_urls = []
        total_size = 0

        # Sort files by filename so student-numbered files go in order
        files_sorted = sorted(files, key=lambda f: f.filename.lower())

        for file in files_sorted:
            if not file.filename:
                continue
            ext = file.filename.rsplit('.', 1)[-1].lower()
            if ext not in allowed:
                return jsonify({"ok": False, "error": f"File {file.filename}: only JPG, PNG, PDF accepted"})

            file_bytes = file.read()
            total_size += len(file_bytes)
            if total_size > 50 * 1024 * 1024:
                return jsonify({"ok": False, "error": "Total upload size too large — max 50MB"})

            if ext == 'pdf':
                # Convert PDF pages to images
                try:
                    from pdf2image import convert_from_bytes
                    pages = convert_from_bytes(file_bytes, dpi=200)
                    base_name = file.filename.rsplit('.', 1)[0]
                    for i, page in enumerate(pages):
                        import io
                        img_bytes = io.BytesIO()
                        page.save(img_bytes, format='JPEG', quality=85)
                        img_bytes = img_bytes.getvalue()
                        page_name = f"{base_name}_page{i+1:02d}.jpg"
                        url = upload_answer_sheet(img_bytes, page_name, 'image/jpeg')
                        uploaded_urls.append({"url": url, "filename": page_name})
                except ImportError:
                    return jsonify({"ok": False, "error": "PDF support unavailable. Please upload JPG images."})
            else:
                content_type = file.content_type or f"image/{ext}"
                url = upload_answer_sheet(file_bytes, file.filename, content_type)
                uploaded_urls.append({"url": url, "filename": file.filename})

        return jsonify({
            "ok": True,
            "urls": uploaded_urls,
            "count": len(uploaded_urls),
            "size_mb": round(total_size / 1024 / 1024, 2)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


# ── SUBMIT EXAM ───────────────────────────────────────
@app.route("/api/student/submit", methods=["POST"])
@require_role("student")
def submit_exam():
    user = get_current_user(request)
    data = request.json
    try:
        assignment_id     = data.get("assignment_id")
        answer_sheet_urls = data.get("answer_sheet_urls", [])  # array of {url, filename}

        if not assignment_id or not answer_sheet_urls:
            return jsonify({"ok": False, "error": "assignment_id and answer_sheet_urls required"})

        # Sort by filename to maintain student-defined sequence
        answer_sheet_urls = sorted(answer_sheet_urls, key=lambda x: x.get("filename","").lower())
        # Primary URL for storage (first image)
        answer_sheet_url = answer_sheet_urls[0]["url"] if answer_sheet_urls else ""

        engine = get_engine()
        with engine.connect() as conn:
            user_id, student_id = get_student_id(conn, user)
            if not student_id:
                return jsonify({"ok": False, "error": "Student profile not found"})

            # Verify assignment belongs to this student and is still pending
            assignment = conn.execute(text("""
                SELECT a.assignment_id, a.paper_id, a.status, a.due_date
                FROM assignments a
                WHERE a.assignment_id = CAST(:aid AS UNIQUEIDENTIFIER)
                AND a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
            """), {"aid": assignment_id, "sid": student_id}).fetchone()

            if not assignment:
                return jsonify({"ok": False, "error": "Assignment not found"})
            if assignment.status != 'assigned':
                return jsonify({"ok": False, "error": "Already submitted"})

            # Get questions for grading
            questions = conn.execute(text("""
                SELECT pq.order_num, pq.section,
                       q.question_id, q.latex_content, q.chapter,
                       q.max_marks, q.type, q.model_solution,
                       CONCAT('Q', pq.order_num) as question_number
                FROM paper_questions pq
                JOIN questions q ON pq.question_id = q.question_id
                WHERE pq.paper_id = CAST(:pid AS UNIQUEIDENTIFIER)
                ORDER BY pq.section, pq.order_num
            """), {"pid": str(assignment.paper_id)}).fetchall()

        questions_list = [dict(q._mapping) for q in questions]

        submission_id = str(uuid.uuid4())

        # Store all URLs as JSON array in answer_sheet_url column
        all_urls_json = json.dumps([item["url"] for item in answer_sheet_urls])

        with engine.begin() as conn:
            # Create submission
            conn.execute(text("""
                INSERT INTO submissions
                    (submission_id, assignment_id, student_id,
                     answer_sheet_url, submitted_at)
                VALUES
                    (CAST(:sub_id AS UNIQUEIDENTIFIER),
                     CAST(:aid AS UNIQUEIDENTIFIER),
                     CAST(:sid AS UNIQUEIDENTIFIER),
                     :url, GETUTCDATE())
            """), {
                "sub_id" : submission_id,
                "aid"    : assignment_id,
                "sid"    : student_id,
                "url"    : all_urls_json
            })

            # Update assignment status
            conn.execute(text("""
                UPDATE assignments SET status = 'submitted'
                WHERE assignment_id = CAST(:aid AS UNIQUEIDENTIFIER)
            """), {"aid": assignment_id})

        # Run AI grading in background thread
        thread = threading.Thread(
            target=run_grading_async,
            args=(submission_id, assignment_id, student_id,
                  questions_list, answer_sheet_urls),
            daemon=True
        )
        thread.start()

        return jsonify({
            "ok"           : True,
            "submission_id": submission_id,
            "message"      : "Submitted successfully. AI grading has started and will complete in 1–2 minutes."
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


# ── CHECK GRADING STATUS ──────────────────────────────
@app.route("/api/student/submission-status/<submission_id>")
@require_role("student")
def submission_status(submission_id):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            sub = conn.execute(text("""
                SELECT total_awarded, total_max, percentage, grade,
                       graded_at, ai_results
                FROM submissions
                WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
            """), {"sid": submission_id}).fetchone()
            if not sub:
                return jsonify({"ok": False, "error": "Not found"})
            result = dict(sub._mapping)
            result["graded"] = sub.graded_at is not None
        return jsonify({"ok": True, "submission": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/student/result/<submission_id>")
@require_role("student")
def get_student_result(submission_id):
    user = get_current_user(request)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            user_id, student_id = get_student_id(conn, user)
            if not student_id:
                return jsonify({"ok": False, "error": "Student profile not found"})

            # Get submission — must belong to this student and be released or graded
            sub = conn.execute(text("""
                SELECT s.submission_id, s.total_awarded, s.total_max,
                       s.percentage, s.grade, s.final_released,
                       s.answer_sheet_url,
                       CONVERT(VARCHAR, s.submitted_at, 120) as submitted_at,
                       CONVERT(VARCHAR, s.released_at, 120) as released_at,
                       p.title, p.subject, p.class, p.duration_minutes,
                       a.assignment_id, a.status as assignment_status
                FROM submissions s
                JOIN assignments a ON s.assignment_id = a.assignment_id
                JOIN papers p ON a.paper_id = p.paper_id
                WHERE s.submission_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND a.student_id = CAST(:stid AS UNIQUEIDENTIFIER)
            """), {"sid": submission_id, "stid": student_id}).fetchone()

            if not sub:
                return jsonify({"ok": False, "error": "Result not found"})

            # Get per-question breakdown — always return AI results
            questions = conn.execute(text("""
                SELECT sq.question_number, sq.max_marks,
                       sq.ai_marks_awarded, sq.teacher_marks, sq.final_marks,
                       sq.ai_step_breakdown, sq.ai_strength, sq.ai_weakness,
                       sq.ai_model_solution, sq.ai_coaching_tip,
                       sq.ai_confidence, sq.ai_flag_review,
                       sq.teacher_feedback,
                       sq.ai_strict_marks, sq.ai_strict_reason,
                       sq.ai_concept, sq.ai_formula, sq.ai_calculation, sq.ai_irrelevant,
                       q.latex_content, q.chapter, q.type,
                       pq.section
                FROM submission_questions sq
                JOIN questions q ON sq.question_id = q.question_id
                JOIN paper_questions pq ON q.question_id = pq.question_id
                    AND pq.paper_id = (
                        SELECT paper_id FROM assignments
                        WHERE assignment_id = (
                            SELECT assignment_id FROM submissions
                            WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
                        )
                    )
                WHERE sq.submission_id = CAST(:sid AS UNIQUEIDENTIFIER)
                ORDER BY sq.question_number
            """), {"sid": submission_id}).fetchall()

            # Check if AI failed
            ai_failed = conn.execute(text("""
                SELECT ai_results FROM submissions
                WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
            """), {"sid": submission_id}).fetchone()
            ai_failed = ai_failed and ai_failed[0] == 'ai_failed' 

            # Open disputes count
            disputes = conn.execute(text("""
                SELECT COUNT(*) FROM disputes
                WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
            """), {"sid": submission_id}).fetchone()[0]

            # Get annotations
            annot = conn.execute(text("""
                SELECT annotations FROM submissions
                WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
            """), {"sid": submission_id}).fetchone()

        result = dict(sub._mapping)
        result["questions"]      = [dict(q._mapping) for q in questions]
        result["disputes_count"] = disputes
        result["ai_failed"]      = bool(ai_failed)
        result["annotations"]    = annot[0] if annot and annot[0] else None
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


# ── REVIEW QUEUE ──────────────────────────────────────
@app.route("/api/teacher/review-queue")
@require_role("teacher")
def review_queue():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            submissions = conn.execute(text("""
                SELECT
                    s.submission_id,
                    s.total_awarded, s.total_max, s.percentage, s.grade,
                    s.answer_sheet_url,
                    CONVERT(VARCHAR, s.submitted_at, 120) as submitted_at,
                    p.title as paper_title, p.subject,
                    u.name as student_name,
                    a.assignment_id
                FROM submissions s
                JOIN assignments a ON s.assignment_id = a.assignment_id
                JOIN papers p ON a.paper_id = p.paper_id
                JOIN students st ON a.student_id = st.student_id
                JOIN users u ON st.user_id = u.user_id
                WHERE a.status = 'graded'
                AND s.final_released = 0
                ORDER BY s.submitted_at ASC
            """)).fetchall()

            result = []
            for sub in submissions:
                sub_dict = dict(sub._mapping)

                # Get per-question data
                questions = conn.execute(text("""
                    SELECT sq.sq_id, sq.question_number, sq.max_marks,
                           sq.ai_marks_awarded, sq.ai_step_breakdown,
                           sq.ai_strength, sq.ai_weakness,
                           sq.ai_model_solution, sq.ai_coaching_tip,
                           sq.ai_confidence, sq.ai_flag_review,
                           sq.ai_strict_marks, sq.ai_strict_reason,
                           sq.ai_concept, sq.ai_formula, sq.ai_calculation, sq.ai_irrelevant,
                           q.latex_content, q.chapter, pq.section
                    FROM submission_questions sq
                    JOIN questions q ON sq.question_id = q.question_id
                    JOIN paper_questions pq ON q.question_id = pq.question_id
                        AND pq.paper_id = (
                            SELECT paper_id FROM assignments
                            WHERE assignment_id = CAST(:aid AS UNIQUEIDENTIFIER)
                        )
                    WHERE sq.submission_id = CAST(:sid AS UNIQUEIDENTIFIER)
                    ORDER BY sq.question_number
                """), {
                    "sid": str(sub.submission_id),
                    "aid": str(sub.assignment_id)
                }).fetchall()

                sub_dict["questions"] = [dict(q._mapping) for q in questions]

                # Parse answer_sheet_url — may be single URL or JSON array
                url_field = sub.answer_sheet_url or ''
                if url_field.startswith('['):
                    try:
                        sub_dict["answer_sheet_urls"] = json.loads(url_field)
                    except:
                        sub_dict["answer_sheet_urls"] = [url_field] if url_field else []
                else:
                    sub_dict["answer_sheet_urls"] = [url_field] if url_field else []

                result.append(sub_dict)

        return jsonify({"ok": True, "submissions": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/teacher/approve-release/<submission_id>", methods=["POST"])
@require_role("teacher")
def approve_release(submission_id):
    data = request.json
    try:
        overrides   = data.get("overrides", [])   # [{sq_id, teacher_marks, teacher_feedback}]
        annotations = data.get("annotations", None) # fabric.js JSON string
        engine = get_engine()

        with engine.begin() as conn:
            total_awarded = 0
            total_max     = 0

            for o in overrides:
                teacher_marks = int(o.get("teacher_marks", 0))
                feedback      = o.get("teacher_feedback", "")

                # Get max_marks and ai_marks for this question
                sq = conn.execute(text("""
                    SELECT sq_id, max_marks, ai_marks_awarded
                    FROM submission_questions
                    WHERE sq_id = CAST(:sqid AS UNIQUEIDENTIFIER)
                """), {"sqid": o["sq_id"]}).fetchone()

                if not sq:
                    continue

                final_marks = min(teacher_marks, sq.max_marks)
                total_max  += sq.max_marks
                total_awarded += final_marks

                conn.execute(text("""
                    UPDATE submission_questions SET
                        teacher_marks    = :tm,
                        final_marks      = :fm,
                        teacher_feedback = :fb,
                        teacher_reviewed = 1
                    WHERE sq_id = CAST(:sqid AS UNIQUEIDENTIFIER)
                """), {
                    "tm"  : teacher_marks,
                    "fm"  : final_marks,
                    "fb"  : feedback,
                    "sqid": o["sq_id"]
                })

            # Recalculate grade
            pct   = round((total_awarded / total_max * 100), 2) if total_max > 0 else 0
            grade = calculate_grade(pct)

            # Update submission — release it with annotations
            conn.execute(text("""
                UPDATE submissions SET
                    total_awarded  = :awarded,
                    total_max      = :total_max,
                    percentage     = :pct,
                    grade          = :grade,
                    final_released = 1,
                    released_at    = GETUTCDATE(),
                    annotations    = :annotations
                WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
            """), {
                "awarded"    : total_awarded,
                "total_max"  : total_max,
                "pct"        : pct,
                "grade"      : grade,
                "sid"        : submission_id,
                "annotations": annotations
            })

            # Update assignment status
            conn.execute(text("""
                UPDATE assignments SET status = 'released'
                WHERE assignment_id = (
                    SELECT assignment_id FROM submissions
                    WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
                )
            """), {"sid": submission_id})

        return jsonify({
            "ok"          : True,
            "total_awarded": total_awarded,
            "total_max"   : total_max,
            "percentage"  : pct,
            "grade"       : grade
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


@app.route("/api/teacher/sas-url", methods=["POST"])
@require_role("teacher")
def get_sas_urls():
    data = request.json
    urls = data.get("urls", [])
    try:
        sas_urls = [get_sas_url(url, expiry_hours=2) for url in urls if url]
        return jsonify({"ok": True, "sas_urls": sas_urls})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200], "sas_urls": []})


@app.route("/api/student/sas-urls", methods=["POST"])
@require_role("student")
def student_sas_urls():
    data = request.json
    urls = data.get("urls", [])
    try:
        sas_urls = [get_sas_url(url, expiry_hours=2) for url in urls if url]
        return jsonify({"ok": True, "sas_urls": sas_urls})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200], "sas_urls": []})


@app.route("/api/student/profile", methods=["GET"])
@require_role("student")
def get_student_profile():
    user = get_current_user(request)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT u.name, u.email, s.class, s.system_reg_number, s.registration_number,
                       CAST(s.student_id AS NVARCHAR(36)) as student_id
                FROM users u
                JOIN students s ON s.user_id = u.user_id
                WHERE u.email = :email
            """), {"email": user["email"]}).fetchone()
            if not row:
                return jsonify({"ok": False, "error": "Profile not found"})
            profile = dict(row._mapping)

        # Auto-generate reg number if missing
        if not profile.get("system_reg_number"):
            with engine.begin() as conn:
                reg_num = generate_system_reg_number(conn)
                conn.execute(text("""
                    UPDATE students SET system_reg_number = :reg_num
                    WHERE student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                """), {"reg_num": reg_num, "sid": profile["student_id"]})
            profile["system_reg_number"] = reg_num

        return jsonify({"ok": True, "profile": profile})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/student/profile", methods=["PUT"])
@require_role("student")
def update_student_profile():
    user = get_current_user(request)
    data = request.json
    try:
        engine = get_engine()
        with engine.begin() as conn:
            # Update name
            if data.get("name"):
                conn.execute(text(
                    "UPDATE users SET name = :name WHERE email = :email"
                ), {"name": data["name"], "email": user["email"]})
            # Update class (6-12 only)
            if data.get("class"):
                cls = int(data["class"])
                if cls < 6 or cls > 12:
                    return jsonify({"ok": False, "error": "Class must be between 6 and 12"})
                conn.execute(text("""
                    UPDATE students SET class = :class
                    WHERE user_id = (SELECT user_id FROM users WHERE email = :email)
                """), {"class": cls, "email": user["email"]})
            # Change password
            if data.get("new_password"):
                if not data.get("current_password"):
                    return jsonify({"ok": False, "error": "Current password required"})
                user_row = conn.execute(text(
                    "SELECT password_hash FROM users WHERE email = :email"
                ), {"email": user["email"]}).fetchone()
                if not user_row or not verify_password(data["current_password"], user_row[0]):
                    return jsonify({"ok": False, "error": "Current password incorrect"})
                conn.execute(text(
                    "UPDATE users SET password_hash = :hash WHERE email = :email"
                ), {"hash": hash_password(data["new_password"]), "email": user["email"]})
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)


# ── PRACTICE API ──────────────────────────────────────
@app.route("/api/student/practice/questions", methods=["GET"])
@require_role("student")
def get_practice_questions():
    """Get tough/very tough questions for practice, filtered by class and chapter."""
    user = get_current_user(request)
    chapter = request.args.get("chapter", "")
    subject = request.args.get("subject", "")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Get student's class
            s_row = conn.execute(text("""
                SELECT s.class FROM students s
                JOIN users u ON s.user_id = u.user_id
                WHERE u.email = :email
            """), {"email": user["email"]}).fetchone()
            student_class = s_row[0] if s_row else 12

            query = """
                SELECT q.question_id, q.latex_content, q.subject, q.chapter,
                       q.difficulty, q.max_marks, q.type, q.class
                FROM questions q
                WHERE LOWER(q.difficulty) IN ('hard', 'very_hard', 'very hard')
                AND q.class = :cls
            """
            params = {"cls": student_class}
            if chapter:
                query += " AND q.chapter LIKE :chapter"
                params["chapter"] = f"%{chapter}%"
            if subject:
                query += " AND q.subject = :subject"
                params["subject"] = subject
            query += " ORDER BY q.difficulty DESC, NEWID()"

            rows = conn.execute(text(query), params).fetchall()

            # Get today's attempt count
            today = datetime.now().date()
            s_row2 = conn.execute(text("""
                SELECT s.student_id FROM students s
                JOIN users u ON s.user_id = u.user_id
                WHERE u.email = :email
            """), {"email": user["email"]}).fetchone()
            attempts_today = 0
            if s_row2:
                cnt = conn.execute(text("""
                    SELECT COUNT(*) FROM practice_attempts
                    WHERE student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                    AND CAST(attempted_at AS DATE) = :today
                """), {"sid": str(s_row2[0]), "today": today}).fetchone()
                attempts_today = cnt[0] if cnt else 0

            return jsonify({
                "ok"            : True,
                "questions"     : [dict(r._mapping) for r in rows],
                "attempts_today": attempts_today,
                "daily_limit"   : 3
            })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/student/practice/submit", methods=["POST"])
@require_role("student")
def submit_practice():
    """Submit a practice answer — single pass AI grading."""
    user    = get_current_user(request)
    data    = request.json
    q_id    = data.get("question_id")
    img_url = data.get("answer_sheet_url")  # already uploaded blob URL
    if not q_id or not img_url:
        return jsonify({"ok": False, "error": "question_id and answer_sheet_url required"})
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Check daily limit
            s_row = conn.execute(text("""
                SELECT s.student_id FROM students s
                JOIN users u ON s.user_id = u.user_id
                WHERE u.email = :email
            """), {"email": user["email"]}).fetchone()
            if not s_row:
                return jsonify({"ok": False, "error": "Student not found"})
            student_id = str(s_row[0])
            today = datetime.now().date()
            cnt = conn.execute(text("""
                SELECT COUNT(*) FROM practice_attempts
                WHERE student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND CAST(attempted_at AS DATE) = :today
            """), {"sid": student_id, "today": today}).fetchone()
            if cnt and cnt[0] >= 3:
                return jsonify({"ok": False, "error": "Daily limit of 3 practice attempts reached. Try again tomorrow."})

            # Get question
            q_row = conn.execute(text("""
                SELECT question_id, latex_content, chapter, subject,
                       max_marks, model_solution, difficulty, type
                FROM questions WHERE question_id = CAST(:qid AS UNIQUEIDENTIFIER)
            """), {"qid": q_id}).fetchone()
            if not q_row:
                return jsonify({"ok": False, "error": "Question not found"})
            q = dict(q_row._mapping)

        # Generate SAS URL for GPT-4o
        sas_url = get_sas_url(img_url, expiry_hours=1)

        # Single pass grading for practice
        client = get_openai_client()
        model_sol = (q.get("model_solution") or "").strip()
        model_sol_text = (
            f"Model Solution (one valid approach only — award full marks for any correct alternate method): {model_sol}"
            if model_sol else
            "No model solution provided — solve this question independently before grading. Award full marks for any correct method."
        )
        q_text = f"""
Q1. [{q['chapter']} · {q['max_marks']} marks]
Question: {q['latex_content']}
{model_sol_text}
---"""

        p1_response = client.chat.completions.create(
            model      = secrets["oai_deploy"],
            messages   = [
                {"role": "system", "content": """You are an expert CBSE Mathematics and Science examiner with deep pedagogical knowledge.
Grade this single practice question. Follow all CBSE marking rules. Award marks for correct steps.
If student used alternate valid method, award full marks.
LATEX RULES:
- ALL mathematical expressions, equations, symbols MUST be wrapped in $ delimiters.
- Inline math: $expression$ — example: $\\int x \\cdot e^x dx$
- NEVER write raw LaTeX without $ delimiters — example WRONG: \\int x dx  RIGHT: $\\int x dx$
- NEVER wrap English words in LaTeX.
RESPOND ONLY with a valid JSON array containing one object. No preamble, no markdown."""},
                {"role": "user", "content": [
                    {"type": "text", "text": f"""Grade this student's practice answer:
{q_text}
Return a JSON array with ONE object containing:
- question_number: "Q1"
- max_marks: integer
- ai_marks_awarded: integer
- ai_strict_marks: integer
- ai_strict_reason: string
- ai_irrelevant: boolean
- ai_concept: string
- ai_formula: string
- ai_calculation: string
- ai_model_solution: string
- ai_coaching_tip: string
- ai_confidence: float 0 to 1
- ai_flag_review: boolean
Return ONLY the JSON array."""},
                    {"type": "image_url", "image_url": {"url": sas_url, "detail": "high"}}
                ]}
            ],
            max_tokens  = 2000,
            temperature = 0.1
        )
        raw = p1_response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        results = json.loads(raw.strip())
        r = results[0] if isinstance(results, list) else results

        # Safeguards
        marks      = min(max(int(r.get("ai_marks_awarded", 0)), 0), q["max_marks"])
        confidence = min(max(float(r.get("ai_confidence", 0.8)), 0.0), 1.0)
        irrelevant = str(r.get("ai_irrelevant", "false")).lower() in ("true", "1")
        flag       = bool(r.get("ai_flag_review", False)) or confidence < 0.85
        strict_marks = min(max(int(r.get("ai_strict_marks", marks)), 0), q["max_marks"])
        pct        = round((marks / q["max_marks"] * 100), 2) if q["max_marks"] > 0 else 0

        # Store attempt
        with engine.begin() as conn:
            attempt_id = str(uuid.uuid4())
            conn.execute(text("""
                INSERT INTO practice_attempts
                    (attempt_id, student_id, question_id, marks_awarded, max_marks,
                     percentage, difficulty_used, answer_image_url,
                     ai_concept, ai_formula, ai_calculation, ai_model_solution,
                     ai_coaching_tip, ai_confidence, ai_strict_marks, ai_strict_reason,
                     ai_irrelevant, attempted_at, attempt_date,
                     allowed_attempt_sec, actual_attempt_sec, actual_upload_sec,
                     total_allowed_sec, total_taken_sec, time_delta_sec, submitted_on_time)
                VALUES
                    (CAST(:aid AS UNIQUEIDENTIFIER), CAST(:sid AS UNIQUEIDENTIFIER),
                     CAST(:qid AS UNIQUEIDENTIFIER), :marks, :max_marks,
                     :pct, :diff, :img_url,
                     :concept, :formula, :calc, :model_sol,
                     :tip, :conf, :strict_marks, :strict_reason,
                     :irrelevant, GETDATE(), CAST(GETDATE() AS DATE),
                     0, 0, 0, 0, 0, 0, 1)
            """), {
                "aid"          : attempt_id,
                "sid"          : student_id,
                "qid"          : q_id,
                "marks"        : marks,
                "max_marks"    : q["max_marks"],
                "pct"          : pct,
                "diff"         : q.get("difficulty", ""),
                "img_url"      : img_url,
                "concept"      : wrap_latex(r.get("ai_concept", "")),
                "formula"      : wrap_latex(r.get("ai_formula", "")),
                "calc"         : wrap_latex(r.get("ai_calculation", "")),
                "model_sol"    : wrap_latex(r.get("ai_model_solution", "")),
                "tip"          : wrap_latex(r.get("ai_coaching_tip", "")),
                "conf"         : confidence,
                "strict_marks" : strict_marks,
                "strict_reason": r.get("ai_strict_reason", ""),
                "irrelevant"   : irrelevant
            })

        return jsonify({
            "ok"          : True,
            "attempt_id"  : attempt_id,
            "marks"       : marks,
            "max_marks"   : q["max_marks"],
            "percentage"  : pct,
            "ai_concept"  : wrap_latex(r.get("ai_concept", "")),
            "ai_formula"  : wrap_latex(r.get("ai_formula", "")),
            "ai_calculation"  : wrap_latex(r.get("ai_calculation", "")),
            "ai_model_solution": wrap_latex(r.get("ai_model_solution", "")),
            "ai_coaching_tip" : wrap_latex(r.get("ai_coaching_tip", "")),
            "ai_strict_marks" : strict_marks,
            "ai_strict_reason": r.get("ai_strict_reason", ""),
            "ai_irrelevant"   : irrelevant,
            "ai_confidence"   : confidence,
            "ai_flag_review"  : flag,
            "chapter"     : q["chapter"]
        })
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e)[:300], "trace": traceback.format_exc()[-500:]})


@app.route("/api/student/practice/doubt", methods=["POST"])
@require_role("student")
def raise_doubt():
    """Raise a doubt after a practice attempt."""
    user = get_current_user(request)
    data = request.json
    attempt_id  = data.get("attempt_id")
    question_id = data.get("question_id")
    doubt_text  = data.get("doubt_text", "").strip()
    chapter     = data.get("chapter", "")
    if not attempt_id or not question_id or not doubt_text:
        return jsonify({"ok": False, "error": "attempt_id, question_id and doubt_text required"})
    try:
        engine = get_engine()
        with engine.begin() as conn:
            s_row = conn.execute(text("""
                SELECT s.student_id FROM students s
                JOIN users u ON s.user_id = u.user_id
                WHERE u.email = :email
            """), {"email": user["email"]}).fetchone()
            if not s_row:
                return jsonify({"ok": False, "error": "Student not found"})
            conn.execute(text("""
                INSERT INTO doubts (doubt_id, student_id, question_id, practice_attempt_id, doubt_text, chapter)
                VALUES (NEWID(), CAST(:sid AS UNIQUEIDENTIFIER), CAST(:qid AS UNIQUEIDENTIFIER),
                        CAST(:aid AS UNIQUEIDENTIFIER), :doubt_text, :chapter)
            """), {
                "sid"       : str(s_row[0]),
                "qid"       : question_id,
                "aid"       : attempt_id,
                "doubt_text": doubt_text,
                "chapter"   : chapter
            })
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


# ── GET STUDENTS LIST ─────────────────────────────────
@app.route("/api/teacher/students", methods=["GET"])
@require_role("teacher")
def get_students():
    try:
        engine = get_engine()
        # Auto-create missing student records for users with role='student'
        with engine.begin() as conn:
            missing = conn.execute(text("""
                SELECT u.user_id FROM users u
                WHERE u.role = 'student' AND u.is_active = 1
                AND NOT EXISTS (SELECT 1 FROM students s WHERE s.user_id = u.user_id)
            """)).fetchall()
            for row in missing:
                conn.execute(text("""
                    INSERT INTO students (student_id, user_id, class)
                    VALUES (NEWID(), CAST(:uid AS UNIQUEIDENTIFIER), 12)
                """), {"uid": str(row[0])})

        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT s.student_id, u.name, u.email,
                    -- Check if student has any unreleased assignment
                    (SELECT TOP 1 p.title
                     FROM assignments a
                     JOIN papers p ON a.paper_id = p.paper_id
                     WHERE a.student_id = s.student_id
                     AND a.status NOT IN ('released')
                    ) as blocked_by
                FROM users u
                JOIN students s ON s.user_id = u.user_id
                WHERE u.role = 'student' AND u.is_active = 1
                ORDER BY u.name
            """)).fetchall()

        # Separate assignable vs blocked
        assignable = []
        blocked    = []
        for r in rows:
            d = dict(r._mapping)
            if d.get('blocked_by'):
                blocked.append(d)
            else:
                assignable.append(d)

        return jsonify({
            "ok"        : True,
            "students"  : assignable,
            "blocked"   : blocked
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


# ── DEBUG STORAGE ────────────────────────────────────
@app.route("/admin/debug-storage")
def debug_storage():
    storage_val = secrets.get("storage", "NOT FOUND")
    if storage_val:
        # Mask the account key but show structure
        safe = storage_val[:80] + "..." if len(storage_val) > 80 else storage_val
        valid = storage_val.startswith("DefaultEndpointsProtocol")
    else:
        safe = "None or empty"
        valid = False
    return jsonify({
        "storage_secret_present": bool(storage_val),
        "storage_secret_preview": safe,
        "looks_valid": valid,
        "length": len(storage_val) if storage_val else 0
    })

# ── DEBUG SCHEMA ─────────────────────────────────────
@app.route("/admin/debug-schema")
def debug_schema():
    engine = get_engine()
    with engine.connect() as conn:
        tables = ['assignments','submissions','students','disputes','submission_questions']
        result = {}
        for t in tables:
            cols = conn.execute(text(
                "SELECT COLUMN_NAME, IS_NULLABLE, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME=:t ORDER BY ORDINAL_POSITION"
            ), {"t": t}).fetchall()
            result[t] = [dict(r._mapping) for r in cols]
    return jsonify(result)

# ── DEBUG STUDENTS ───────────────────────────────────
@app.route("/admin/debug-students")
def debug_students():
    engine = get_engine()
    with engine.connect() as conn:
        users = conn.execute(text("SELECT user_id, name, email, role, is_active FROM users WHERE role='student'")).fetchall()
        students = conn.execute(text("SELECT student_id, user_id FROM students")).fetchall()
        schema = conn.execute(text("SELECT COLUMN_NAME, IS_NULLABLE, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='students'")).fetchall()
    return jsonify({
        "student_users": [dict(r._mapping) for r in users],
        "students_table": [dict(r._mapping) for r in students],
        "students_schema": [dict(r._mapping) for r in schema]
    })

# ── GET PUBLISHED PAPERS FOR ASSIGN ───────────────────
@app.route("/api/teacher/published-papers", methods=["GET"])
@require_role("teacher")
def get_published_papers():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT p.paper_id, p.title, p.subject, p.class,
                       p.total_marks, p.duration_minutes,
                       COUNT(pq.question_id) as question_count
                FROM papers p
                LEFT JOIN paper_questions pq ON p.paper_id = pq.paper_id
                WHERE p.is_active = 1
                GROUP BY p.paper_id, p.title, p.subject, p.class,
                         p.total_marks, p.duration_minutes
                ORDER BY p.title
            """)).fetchall()
        return jsonify({"ok": True, "papers": [dict(r._mapping) for r in rows]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


# ── CREATE ASSIGNMENT ─────────────────────────────────
@app.route("/api/teacher/assignments", methods=["POST"])
@require_role("teacher")
def create_assignment():
    import uuid
    user = get_current_user(request)
    data = request.json
    try:
        paper_id    = data.get("paper_id")
        student_ids = data.get("student_ids", [])
        due_date    = data.get("due_date")

        if not paper_id or not student_ids or not due_date:
            return jsonify({"ok": False, "error": "paper_id, student_ids and due_date are required"})

        engine = get_engine()
        with engine.connect() as conn:
            teacher_user = conn.execute(text(
                "SELECT user_id FROM users WHERE email = :email"
            ), {"email": user["email"]}).fetchone()
        if not teacher_user:
            return jsonify({"ok": False, "error": "Teacher not found"})

        with engine.connect() as conn:
            teacher_row = conn.execute(text(
                "SELECT teacher_id FROM teachers WHERE user_id = CAST(:uid AS UNIQUEIDENTIFIER)"
            ), {"uid": str(teacher_user[0])}).fetchone()
        if not teacher_row:
            return jsonify({"ok": False, "error": "Teacher profile not found"})

        assigned_count = 0
        skipped_count  = 0
        blocked_students = []
        with engine.begin() as conn:
            for sid in student_ids:
                # Block if student has any submitted or graded (unreleased) assignment
                unreleased = conn.execute(text("""
                    SELECT a.assignment_id, p.title
                    FROM assignments a
                    JOIN papers p ON a.paper_id = p.paper_id
                    WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                    AND a.status NOT IN ('released')
                """), {"sid": str(sid)}).fetchone()

                if unreleased:
                    blocked_students.append({
                        "student_id": str(sid),
                        "blocked_by": unreleased.title
                    })
                    skipped_count += 1
                    continue

                existing = conn.execute(text("""
                    SELECT assignment_id FROM assignments
                    WHERE paper_id = CAST(:pid AS UNIQUEIDENTIFIER)
                    AND student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                """), {"pid": str(paper_id), "sid": str(sid)}).fetchone()

                if existing:
                    skipped_count += 1
                    continue

                conn.execute(text("""
                    INSERT INTO assignments
                        (assignment_id, paper_id, student_id, assigned_by,
                         due_date, watermark_token, status)
                    VALUES
                        (NEWID(),
                         CAST(:pid AS UNIQUEIDENTIFIER),
                         CAST(:sid AS UNIQUEIDENTIFIER),
                         CAST(:assigned_by AS UNIQUEIDENTIFIER),
                         :due_date, :token, 'assigned')
                """), {
                    "pid"         : str(paper_id),
                    "sid"         : str(sid),
                    "assigned_by" : str(teacher_row[0]),
                    "due_date"    : due_date,
                    "token"       : str(uuid.uuid4())
                })
                assigned_count += 1

        msg = f"Assigned to {assigned_count} student(s)."
        if blocked_students:
            blocked_titles = set(b["blocked_by"] for b in blocked_students)
            msg += f" {len(blocked_students)} student(s) skipped — they have unreleased grades for: {', '.join(blocked_titles)}. Release those grades first."
        elif skipped_count:
            msg += f" {skipped_count} already assigned (skipped)."
        return jsonify({"ok": True, "message": msg, "assigned": assigned_count,
                        "skipped": skipped_count, "blocked": blocked_students})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


# ── TEACHER DOUBTS API ────────────────────────────────
@app.route("/api/teacher/doubts", methods=["GET"])
@require_role("teacher")
def get_doubts():
    """Get doubts grouped by question, sorted by most recent, with status filter."""
    status_filter = request.args.get("status", "open")  # open or addressed
    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    CAST(d.doubt_id AS NVARCHAR(36))     as doubt_id,
                    CAST(d.question_id AS NVARCHAR(36))  as question_id,
                    d.doubt_text, d.chapter, d.raised_at,
                    ISNULL(d.status, 'open')             as status,
                    u.name                               as student_name,
                    q.latex_content, q.max_marks,
                    pa.marks_awarded,
                    pa.max_marks                         as q_max_marks
                FROM doubts d
                JOIN students s   ON d.student_id   = s.student_id
                JOIN users u      ON s.user_id       = u.user_id
                JOIN questions q  ON d.question_id   = q.question_id
                LEFT JOIN practice_attempts pa ON d.practice_attempt_id = pa.attempt_id
                WHERE ISNULL(d.status, 'open') = :status
                ORDER BY d.raised_at DESC
            """), {"status": status_filter}).fetchall()

            doubts = [dict(r._mapping) for r in rows]

            # Group by question_id
            groups = {}
            for d in doubts:
                qid = d["question_id"]
                if qid not in groups:
                    groups[qid] = {
                        "question_id"   : qid,
                        "latex_content" : d["latex_content"],
                        "chapter"       : d.get("chapter") or "Unknown",
                        "max_marks"     : d["max_marks"],
                        "doubt_count"   : 0,
                        "student_count" : 0,
                        "latest_date"   : d["raised_at"],
                        "students"      : set(),
                        "doubts"        : []
                    }
                g = groups[qid]
                g["doubt_count"] += 1
                g["students"].add(d["student_name"])
                # Track most recent
                if d["raised_at"] and (not g["latest_date"] or d["raised_at"] > g["latest_date"]):
                    g["latest_date"] = d["raised_at"]
                g["doubts"].append({
                    "doubt_id"     : d["doubt_id"],
                    "student_name" : d["student_name"],
                    "doubt_text"   : d["doubt_text"],
                    "raised_at"    : d["raised_at"].isoformat() if d["raised_at"] else None,
                    "marks_awarded": d["marks_awarded"],
                    "q_max_marks"  : d["q_max_marks"],
                    "status"       : d["status"]
                })

            # Convert to list, sort by latest_date DESC
            grouped = []
            for g in groups.values():
                g["student_count"] = len(g["students"])
                g["latest_date"]   = g["latest_date"].isoformat() if g["latest_date"] else None
                del g["students"]
                grouped.append(g)

            grouped.sort(key=lambda x: x["latest_date"] or "", reverse=True)

            return jsonify({"ok": True, "groups": grouped, "total": len(doubts)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/teacher/doubts/address", methods=["POST"])
@require_role("teacher")
def address_doubts():
    """Mark doubts as addressed — by question_id (all for that question) or single doubt_id."""
    data        = request.json
    question_id = data.get("question_id")
    doubt_id    = data.get("doubt_id")
    try:
        engine = get_engine()
        with engine.begin() as conn:
            if question_id:
                conn.execute(text("""
                    UPDATE doubts SET status = 'addressed', addressed_at = GETDATE()
                    WHERE question_id = CAST(:qid AS UNIQUEIDENTIFIER)
                    AND ISNULL(status, 'open') = 'open'
                """), {"qid": question_id})
            elif doubt_id:
                conn.execute(text("""
                    UPDATE doubts SET status = 'addressed', addressed_at = GETDATE()
                    WHERE doubt_id = CAST(:did AS UNIQUEIDENTIFIER)
                """), {"did": doubt_id})
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


# ── TEACHER PERFORMANCE DASHBOARD API ────────────────
@app.route("/api/teacher/performance", methods=["GET"])
@require_role("teacher")
def get_performance():
    """Get performance data for all students or a specific student."""
    student_id = request.args.get("student_id", "")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Get all students
            students = conn.execute(text("""
                SELECT CAST(s.student_id AS NVARCHAR(36)) as student_id,
                       u.name, s.class, s.system_reg_number
                FROM students s
                JOIN users u ON s.user_id = u.user_id
                WHERE u.role = 'student' AND u.is_active = 1
                ORDER BY u.name
            """)).fetchall()

            if not student_id:
                return jsonify({"ok": True, "students": [dict(r._mapping) for r in students]})

            # Per-student performance
            submissions = conn.execute(text("""
                SELECT sub.submission_id, sub.total_awarded, sub.total_max,
                       sub.percentage, sub.submitted_at, sub.final_released,
                       p.title as paper_title, p.subject
                FROM submissions sub
                JOIN assignments a ON sub.assignment_id = a.assignment_id
                JOIN papers p      ON a.paper_id = p.paper_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND sub.graded_at IS NOT NULL
                ORDER BY sub.submitted_at DESC
            """), {"sid": student_id}).fetchall()

            # Per-question performance
            questions = conn.execute(text("""
                SELECT sq.question_number, sq.max_marks, sq.final_marks, sq.ai_marks_awarded,
                       sq.ai_concept, sq.ai_coaching_tip, q.chapter, q.subject, q.type,
                       p.title as paper_title, sub.submitted_at
                FROM submission_questions sq
                JOIN submissions sub ON sq.submission_id = sub.submission_id
                JOIN questions q     ON sq.question_id = q.question_id
                JOIN assignments a   ON sub.assignment_id = a.assignment_id
                JOIN papers p        ON a.paper_id = p.paper_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND sub.graded_at IS NOT NULL
                ORDER BY sub.submitted_at DESC
            """), {"sid": student_id}).fetchall()

            # Practice performance
            practice = conn.execute(text("""
                SELECT pa.marks_awarded, pa.max_marks, pa.percentage,
                       pa.difficulty_used, pa.attempted_at,
                       q.chapter, q.subject, q.latex_content
                FROM practice_attempts pa
                JOIN questions q ON pa.question_id = q.question_id
                WHERE pa.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                ORDER BY pa.attempted_at DESC
            """), {"sid": student_id}).fetchall()

            # Mark-wise performance aggregation
            mark_wise = {}
            for q in questions:
                mk = q.max_marks
                if mk not in mark_wise:
                    mark_wise[mk] = {"total": 0, "awarded": 0, "count": 0}
                final = q.final_marks if q.final_marks is not None else q.ai_marks_awarded
                if final is not None:
                    mark_wise[mk]["total"]   += mk
                    mark_wise[mk]["awarded"] += final
                    mark_wise[mk]["count"]   += 1

            # Chapter-wise performance
            chapter_wise = {}
            for q in questions:
                ch = q.chapter or "Unknown"
                if ch not in chapter_wise:
                    chapter_wise[ch] = {"total": 0, "awarded": 0, "count": 0}
                final = q.final_marks if q.final_marks is not None else q.ai_marks_awarded
                if final is not None:
                    chapter_wise[ch]["total"]   += q.max_marks
                    chapter_wise[ch]["awarded"] += final
                    chapter_wise[ch]["count"]   += 1

            # Student name, reg number, class
            s_data = next((dict(r._mapping) for r in students
                          if str(dict(r._mapping)["student_id"]) == student_id), {})
            s_name   = s_data.get("name", "Student")
            s_reg    = s_data.get("system_reg_number", "")
            s_class  = s_data.get("class", "")

            return jsonify({
                "ok"              : True,
                "student_name"    : s_name,
                "system_reg_number": s_reg,
                "student_class"   : s_class,
                "submissions"     : [dict(r._mapping) for r in submissions],
                "questions"       : [dict(r._mapping) for r in questions],
                "practice"        : [dict(r._mapping) for r in practice],
                "mark_wise"       : [
                    {"marks": k, "avg_pct": round(v["awarded"]/v["total"]*100, 1) if v["total"] > 0 else 0,
                     "count": v["count"]}
                    for k, v in sorted(mark_wise.items())
                ],
                "chapter_wise"    : [
                    {"chapter": k, "avg_pct": round(v["awarded"]/v["total"]*100, 1) if v["total"] > 0 else 0,
                     "count": v["count"], "total": v["total"], "awarded": v["awarded"]}
                    for k, v in sorted(chapter_wise.items(), key=lambda x: -x[1]["awarded"]/x[1]["total"] if x[1]["total"] > 0 else 0)
                ]
            })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/teacher/performance/narrative", methods=["POST"])
@require_role("teacher")
def get_performance_narrative():
    """Generate AI narrative summary for a student's performance."""
    data       = request.json
    student_id = data.get("student_id")
    if not student_id:
        return jsonify({"ok": False, "error": "student_id required"})
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Get student name
            s_row = conn.execute(text("""
                SELECT u.name, s.class FROM students s
                JOIN users u ON s.user_id = u.user_id
                WHERE CAST(s.student_id AS NVARCHAR(36)) = :sid
            """), {"sid": student_id}).fetchone()
            s_name  = s_row[0] if s_row else "Student"
            s_class = s_row[1] if s_row else 12

            # Get chapter-wise performance
            rows = conn.execute(text("""
                SELECT q.chapter, q.max_marks,
                       ISNULL(sq.final_marks, sq.ai_marks_awarded) as awarded
                FROM submission_questions sq
                JOIN questions q ON sq.question_id = q.question_id
                JOIN submissions sub ON sq.submission_id = sub.submission_id
                JOIN assignments a ON sub.assignment_id = a.assignment_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND sub.graded_at IS NOT NULL
            """), {"sid": student_id}).fetchall()

            # Aggregate
            chapters = {}
            for r in rows:
                ch = r[0] or "Unknown"
                if ch not in chapters:
                    chapters[ch] = {"total": 0, "awarded": 0}
                chapters[ch]["total"]   += r[1] or 0
                chapters[ch]["awarded"] += r[2] or 0

            chapter_summary = "\n".join([
                f"- {ch}: {round(v['awarded']/v['total']*100,1)}% ({v['awarded']}/{v['total']})"
                for ch, v in sorted(chapters.items(), key=lambda x: -x[1]["awarded"]/x[1]["total"] if x[1]["total"] > 0 else 0)
            ]) or "No exam data available yet."

            # Practice summary
            practice_count = conn.execute(text("""
                SELECT COUNT(*) FROM practice_attempts
                WHERE student_id = CAST(:sid AS UNIQUEIDENTIFIER)
            """), {"sid": student_id}).fetchone()[0]

        client = get_openai_client()
        prompt = f"""You are an expert academic counselor writing a brief performance summary for a teacher.

Student: {s_name}, Class {s_class}
Chapter-wise exam performance:
{chapter_summary}
Practice attempts: {practice_count}

Write a 3-4 sentence performance summary that:
1. Identifies the student's strongest topics
2. Identifies areas needing improvement
3. Mentions practice engagement
4. Gives one specific actionable recommendation for the teacher

Write in third person, professional tone. Be specific and constructive, not generic."""

        response = client.chat.completions.create(
            model      = secrets["oai_deploy"],
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 300,
            temperature = 0.7
        )
        narrative = response.choices[0].message.content.strip()
        return jsonify({"ok": True, "narrative": narrative})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

# ── STUDENT REPORTS API ───────────────────────────────
@app.route("/api/teacher/student-submissions", methods=["GET"])
@require_role("teacher")
def get_student_submissions():
    """Get all submissions for a student — for Student Reports page."""
    student_id = request.args.get("student_id", "")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            if not student_id:
                # Return student list
                rows = conn.execute(text("""
                    SELECT CAST(s.student_id AS NVARCHAR(36)) as student_id,
                           u.name, s.class, s.system_reg_number
                    FROM students s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE u.role = 'student' AND u.is_active = 1
                    ORDER BY u.name
                """)).fetchall()
                return jsonify({"ok": True, "students": [dict(r._mapping) for r in rows]})

            # Get submissions for this student
            rows = conn.execute(text("""
                SELECT CAST(sub.submission_id AS NVARCHAR(36)) as submission_id,
                       p.title as paper_title, p.subject,
                       sub.total_awarded, sub.total_max, sub.percentage,
                       sub.submitted_at, sub.final_released,
                       sub.graded_at
                FROM submissions sub
                JOIN assignments a ON sub.assignment_id = a.assignment_id
                JOIN papers p      ON a.paper_id = p.paper_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND sub.graded_at IS NOT NULL
                ORDER BY sub.submitted_at DESC
            """), {"sid": student_id}).fetchall()
            return jsonify({"ok": True, "submissions": [dict(r._mapping) for r in rows]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/teacher/student-report/<submission_id>", methods=["GET"])
@require_role("teacher")
def get_student_report(submission_id):
    """Get full Q-by-Q report for a submission — for Student Reports page."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Submission info
            sub = conn.execute(text("""
                SELECT CAST(sub.submission_id AS NVARCHAR(36)) as submission_id,
                       p.title as paper_title, p.subject,
                       sub.total_awarded, sub.total_max, sub.percentage,
                       sub.submitted_at, sub.final_released, sub.annotations,
                       sub.answer_sheet_url,
                       u.name as student_name,
                       s.system_reg_number, s.class
                FROM submissions sub
                JOIN assignments a ON sub.assignment_id = a.assignment_id
                JOIN papers p      ON a.paper_id = p.paper_id
                JOIN students s    ON a.student_id = s.student_id
                JOIN users u       ON s.user_id = u.user_id
                WHERE CAST(sub.submission_id AS NVARCHAR(36)) = :sid
            """), {"sid": submission_id}).fetchone()
            if not sub:
                return jsonify({"ok": False, "error": "Submission not found"})

            # Questions
            questions = conn.execute(text("""
                SELECT sq.question_number, sq.max_marks,
                       sq.ai_marks_awarded, sq.teacher_marks, sq.final_marks,
                       sq.ai_concept, sq.ai_formula, sq.ai_calculation,
                       sq.ai_model_solution, sq.ai_coaching_tip,
                       sq.ai_strict_marks, sq.ai_strict_reason,
                       sq.ai_confidence, sq.ai_flag_review, sq.ai_irrelevant,
                       sq.teacher_feedback,
                       q.latex_content, q.chapter, q.max_marks as q_max,
                       pq.section
                FROM submission_questions sq
                JOIN questions q ON sq.question_id = q.question_id
                JOIN paper_questions pq ON q.question_id = pq.question_id
                    AND pq.paper_id = (
                        SELECT paper_id FROM assignments
                        WHERE assignment_id = (SELECT assignment_id FROM submissions WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER))
                    )
                WHERE sq.submission_id = CAST(:sid AS UNIQUEIDENTIFIER)
                ORDER BY sq.question_number
            """), {"sid": submission_id}).fetchall()

            return jsonify({
                "ok"        : True,
                "submission": dict(sub._mapping),
                "questions" : [dict(r._mapping) for r in questions]
            })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


# ── ERROR ANALYSIS API ────────────────────────────────
@app.route("/api/teacher/error-analysis", methods=["GET"])
@require_role("teacher")
def get_error_analysis():
    """Get cached error analysis for this teacher."""
    user = get_current_user(request)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            teacher = conn.execute(text("""
                SELECT t.teacher_id FROM teachers t
                JOIN users u ON t.user_id = u.user_id
                WHERE u.email = :email
            """), {"email": user["email"]}).fetchone()
            if not teacher:
                return jsonify({"ok": False, "error": "Teacher not found"})
            teacher_id = str(teacher[0])

            row = conn.execute(text("""
                SELECT analysis, generated_at
                FROM error_analysis_cache
                WHERE CAST(teacher_id AS NVARCHAR(36)) = :tid
                ORDER BY generated_at DESC
            """), {"tid": teacher_id}).fetchone()

            if not row or not row[0]:
                return jsonify({"ok": True, "analysis": None, "generated_at": None})

            return jsonify({
                "ok"          : True,
                "analysis"    : json.loads(row[0]),
                "generated_at": row[1].isoformat() if row[1] else None
            })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/teacher/error-analysis/generate", methods=["POST"])
@require_role("teacher")
def generate_error_analysis():
    """Generate AI error analysis across all student submissions for this teacher."""
    user = get_current_user(request)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            teacher = conn.execute(text("""
                SELECT t.teacher_id FROM teachers t
                JOIN users u ON t.user_id = u.user_id
                WHERE u.email = :email
            """), {"email": user["email"]}).fetchone()
            if not teacher:
                return jsonify({"ok": False, "error": "Teacher not found"})
            teacher_id = str(teacher[0])

            # Get all feedback from submitted questions for papers by this teacher
            rows = conn.execute(text("""
                SELECT q.chapter, q.subject,
                       sq.ai_concept, sq.ai_formula, sq.ai_calculation,
                       sq.ai_strict_reason, sq.teacher_feedback,
                       sq.ai_marks_awarded, sq.max_marks,
                       sq.ai_irrelevant
                FROM submission_questions sq
                JOIN questions q ON sq.question_id = q.question_id
                JOIN submissions sub ON sq.submission_id = sub.submission_id
                JOIN assignments a ON sub.assignment_id = a.assignment_id
                JOIN papers p ON a.paper_id = p.paper_id
                WHERE p.created_by = CAST(:tid AS UNIQUEIDENTIFIER)
                AND sub.graded_at IS NOT NULL
                AND sq.ai_marks_awarded < sq.max_marks
            """), {"tid": teacher_id}).fetchall()

        if not rows:
            return jsonify({"ok": False, "error": "No graded submissions found to analyse."})

        # Build feedback text grouped by chapter
        chapter_data = {}
        for r in rows:
            ch = r.chapter or "Unknown"
            if ch not in chapter_data:
                chapter_data[ch] = []
            feedback_parts = []
            if r.ai_concept:    feedback_parts.append(f"Concept: {r.ai_concept}")
            if r.ai_formula:    feedback_parts.append(f"Formula: {r.ai_formula}")
            if r.ai_calculation: feedback_parts.append(f"Calculation: {r.ai_calculation}")
            if r.ai_strict_reason and r.ai_strict_reason != "Strict and neutral agree":
                feedback_parts.append(f"Strict note: {r.ai_strict_reason}")
            if r.teacher_feedback: feedback_parts.append(f"Teacher: {r.teacher_feedback}")
            if feedback_parts:
                chapter_data[ch].append(" | ".join(feedback_parts))

        # Build prompt
        feedback_text = ""
        for ch, feedbacks in chapter_data.items():
            feedback_text += f"\n\n{ch}:\n" + "\n".join(f"- {f}" for f in feedbacks[:20])

        client = get_openai_client()
        prompt = f"""You are an expert academic analyst. Below is AI and teacher feedback from student exam submissions, grouped by chapter.

{feedback_text}

Analyse these feedbacks and identify RECURRING error patterns (errors appearing in multiple submissions, not one-off mistakes).

Return a JSON object with exactly this structure:
{{
  "by_chapter": {{
    "Chapter Name": ["bullet point error pattern", "another pattern"],
    "Another Chapter": ["pattern"]
  }},
  "by_error_type": {{
    "Conceptual Errors": ["pattern — Chapter", "pattern — Chapter"],
    "Calculation Errors": ["pattern — Chapter"],
    "Formula Errors": ["pattern — Chapter"],
    "Incomplete Solutions": ["pattern — Chapter"]
  }},
  "summary": "2-3 sentence overall summary of the most critical patterns"
}}

Only include error types that actually appear in the data. Only include recurring patterns (2+ occurrences). Be specific and actionable. Return ONLY valid JSON."""

        response = client.chat.completions.create(
            model       = secrets["oai_deploy"],
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 1500,
            temperature = 0.3
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        analysis = json.loads(raw.strip())

        # Save to cache
        with engine.begin() as conn:
            # Delete old cache for this teacher
            conn.execute(text("""
                DELETE FROM error_analysis_cache
                WHERE CAST(teacher_id AS NVARCHAR(36)) = :tid
            """), {"tid": teacher_id})
            # Insert new
            conn.execute(text("""
                INSERT INTO error_analysis_cache (cache_id, teacher_id, analysis, generated_at)
                VALUES (NEWID(), CAST(:tid AS UNIQUEIDENTIFIER), :analysis, GETDATE())
            """), {"tid": teacher_id, "analysis": json.dumps(analysis)})

        return jsonify({
            "ok"          : True,
            "analysis"    : analysis,
            "generated_at": datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e)[:300]})
