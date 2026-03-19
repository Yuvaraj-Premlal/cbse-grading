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
    blob_service  = get_blob_client()
    container     = blob_service.get_container_client(BLOB_CONTAINER)
    blob_name     = f"{uuid.uuid4()}_{filename}"
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
    # Parse blob name from URL
    # URL format: https://<account>.blob.core.windows.net/<container>/<blob_name>
    parts     = blob_url.split(f".blob.core.windows.net/{BLOB_CONTAINER}/")
    blob_name = parts[1] if len(parts) > 1 else blob_url
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

def grade_submission(questions, answer_sheet_url):
    """
    Grade student answer sheet using GPT-4o vision.
    Returns list of per-question results.
    """
    client = get_openai_client()

    # Build question list for prompt
    q_text = ""
    for i, q in enumerate(questions, 1):
        q_text += f"""
Q{i}. [{q['chapter']} · {q['max_marks']} marks]
{q['latex_content']}
Model Solution: {q.get('model_solution') or 'Grade based on standard CBSE marking scheme'}
---"""

    system_prompt = """You are an expert CBSE Mathematics and Science examiner.
You will be given a question paper and a student's handwritten answer sheet image.
Grade each question strictly according to CBSE marking scheme — award marks for correct steps even if final answer is wrong.
Respond ONLY with a valid JSON array. No preamble, no markdown, no backticks."""

    user_prompt = f"""Grade this student's answer sheet for the following questions:

{q_text}

For each question return a JSON object with these exact fields:
- question_number: "Q1", "Q2" etc
- max_marks: integer
- ai_marks_awarded: integer (0 to max_marks)
- ai_step_breakdown: string explaining marks per step
- ai_strength: string (what student did well)
- ai_weakness: string (what student missed or got wrong)
- ai_model_solution: string (brief ideal solution)
- ai_coaching_tip: string (one actionable tip for improvement)
- ai_confidence: float between 0 and 1 (your confidence in this grade)
- ai_flag_review: boolean (true if teacher should review)

Return ONLY the JSON array, nothing else."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": answer_sheet_url, "detail": "high"}}
        ]}
    ]

    response = client.chat.completions.create(
        model       = secrets["oai_deploy"],
        messages    = messages,
        max_tokens  = 4000,
        temperature = 0.1
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def calculate_grade(percentage):
    if percentage >= 90: return 'A+'
    if percentage >= 75: return 'A'
    if percentage >= 60: return 'B+'
    if percentage >= 50: return 'B'
    if percentage >= 40: return 'C'
    if percentage >= 33: return 'D'
    return 'F'

def run_grading_async(submission_id, assignment_id, student_id, questions, answer_sheet_url):
    """Run AI grading in background thread."""
    try:
        engine = get_engine()

        # Generate SAS URL so GPT-4o can read the private blob
        sas_url = get_sas_url(answer_sheet_url, expiry_hours=1)
        print(f"Grading with SAS URL: {sas_url[:80]}...", flush=True)

        # Grade with GPT-4o
        results = grade_submission(questions, sas_url)

        total_awarded = 0
        total_max     = sum(q['max_marks'] for q in questions)

        with engine.begin() as conn:
            for r in results:
                # Find matching question
                q_match = next((q for q in questions
                    if q['question_number'] == r.get('question_number')), None)
                question_id = q_match['question_id'] if q_match else None

                marks = min(r.get('ai_marks_awarded', 0), r.get('max_marks', 0))
                total_awarded += marks

                conn.execute(text("""
                    INSERT INTO submission_questions
                        (sq_id, submission_id, question_id, question_number,
                         max_marks, ai_marks_awarded, ai_step_breakdown,
                         ai_strength, ai_weakness, ai_model_solution,
                         ai_coaching_tip, ai_confidence, ai_flag_review)
                    VALUES
                        (NEWID(),
                         CAST(:sid AS UNIQUEIDENTIFIER),
                         CAST(:qid AS UNIQUEIDENTIFIER),
                         :qnum, :max_marks, :awarded,
                         :breakdown, :strength, :weakness,
                         :model_sol, :tip,
                         :confidence, :flag)
                """), {
                    "sid"       : str(submission_id),
                    "qid"       : str(question_id) if question_id else str(uuid.uuid4()),
                    "qnum"      : r.get('question_number'),
                    "max_marks" : r.get('max_marks', 0),
                    "awarded"   : marks,
                    "breakdown" : r.get('ai_step_breakdown', ''),
                    "strength"  : r.get('ai_strength', ''),
                    "weakness"  : r.get('ai_weakness', ''),
                    "model_sol" : r.get('ai_model_solution', ''),
                    "tip"       : r.get('ai_coaching_tip', ''),
                    "confidence": float(r.get('ai_confidence', 0.8)),
                    "flag"      : bool(r.get('ai_flag_review', False))
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
                WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER)
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
                conn.execute(text("""
                    UPDATE submissions SET
                        ai_results = :err
                    WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER)
                """), {"err": f"Grading failed: {str(e)[:200]}", "sid": str(submission_id)})
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
                JOIN users u       ON a.student_id  = u.user_id
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
        if data["difficulty"] not in ["easy", "medium", "hard"]:
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
        if data["difficulty"] not in ["easy", "medium", "hard"]:
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

def get_student_id(conn, user):
    """Resolve user email -> user_id -> student_id. Auto-creates student record if missing."""
    user_row = conn.execute(text(
        "SELECT user_id FROM users WHERE email = :email"
    ), {"email": user["email"]}).fetchone()
    if not user_row:
        return None, None

    student_row = conn.execute(text(
        "SELECT student_id FROM students WHERE user_id = CAST(:uid AS UNIQUEIDENTIFIER)"
    ), {"uid": str(user_row[0])}).fetchone()

    return str(user_row[0]), str(student_row[0]) if student_row else None


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
                    CONVERT(VARCHAR, s.submitted_at, 120) as submitted_at
                FROM assignments a
                JOIN papers p ON a.paper_id = p.paper_id
                LEFT JOIN submissions s ON a.assignment_id = s.assignment_id
                WHERE a.student_id = CAST(:sid AS UNIQUEIDENTIFIER)
                AND a.status IN ('submitted', 'graded')
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

        return jsonify({
            "ok": True,
            "dashboard": {
                "name"           : user["name"],
                "student_id"     : student_id,
                "pending_exams"  : [dict(r._mapping) for r in pending],
                "under_review"   : [dict(r._mapping) for r in under_review],
                "released"       : [dict(r._mapping) for r in released],
                "stats": {
                    "total_assigned"  : total_assigned,
                    "total_submitted" : total_submitted,
                    "avg_score"       : round(float(avg_score), 1) if avg_score else None,
                    "open_disputes"   : open_disputes
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
    user = get_current_user(request)
    try:
        if 'file' not in request.files:
            return jsonify({"ok": False, "error": "No file provided"})

        file = request.files['file']
        if not file.filename:
            return jsonify({"ok": False, "error": "No file selected"})

        # Validate file type
        allowed = {'pdf', 'jpg', 'jpeg', 'png'}
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in allowed:
            return jsonify({"ok": False, "error": "Only PDF, JPG, PNG accepted"})

        # Validate file size (20MB max)
        file_bytes = file.read()
        if len(file_bytes) > 20 * 1024 * 1024:
            return jsonify({"ok": False, "error": "File too large — max 20MB"})

        content_type = file.content_type or f"image/{ext}"
        url = upload_answer_sheet(file_bytes, file.filename, content_type)

        return jsonify({"ok": True, "url": url, "filename": file.filename,
                        "size_mb": round(len(file_bytes) / 1024 / 1024, 2)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


# ── SUBMIT EXAM ───────────────────────────────────────
@app.route("/api/student/submit", methods=["POST"])
@require_role("student")
def submit_exam():
    user = get_current_user(request)
    data = request.json
    try:
        assignment_id    = data.get("assignment_id")
        answer_sheet_url = data.get("answer_sheet_url")

        if not assignment_id or not answer_sheet_url:
            return jsonify({"ok": False, "error": "assignment_id and answer_sheet_url required"})

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
                "url"    : answer_sheet_url
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
                  questions_list, answer_sheet_url),
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
                WHERE submission_id = CAST(:sid AS UNIQUEIDENTIFIER)
            """), {"sid": submission_id}).fetchone()
            if not sub:
                return jsonify({"ok": False, "error": "Not found"})
            result = dict(sub._mapping)
            result["graded"] = sub.graded_at is not None
        return jsonify({"ok": True, "submission": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)


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
                SELECT s.student_id, u.name, u.email
                FROM users u
                JOIN students s ON s.user_id = u.user_id
                WHERE u.role = 'student' AND u.is_active = 1
                ORDER BY u.name
            """)).fetchall()
        return jsonify({"ok": True, "students": [dict(r._mapping) for r in rows]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})


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
        with engine.begin() as conn:
            for sid in student_ids:
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
        if skipped_count:
            msg += f" {skipped_count} already assigned (skipped)."
        return jsonify({"ok": True, "message": msg, "assigned": assigned_count, "skipped": skipped_count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:300]})
