from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
from sqlalchemy import create_engine, text
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from datetime import datetime, timedelta
import urllib, os, hashlib, hmac
from jose import jwt, JWTError

load_dotenv()
app = Flask(__name__)

# ── SECRETS ───────────────────────────────────────────
def get_secrets():
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(
            vault_url=os.getenv("KEY_VAULT_URL"),
            credential=credential
        )
        return {
            "db":  client.get_secret("DB-CONNECTION-STRING").value,
            "jwt": client.get_secret("JWT-SECRET").value,
        }
    except Exception as e:
        print(f"Key Vault fallback: {e}")
        return {
            "db":  os.getenv("DB_CONNECTION_STRING"),
            "jwt": os.getenv("JWT_SECRET"),
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
    """Decorator — checks JWT cookie, redirects to login if missing or wrong role."""
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

            # Set JWT as httpOnly cookie — 7 days
            resp.set_cookie(
                "skooltest_token",
                token,
                max_age   = 60 * 60 * 24 * 7,
                httponly  = True,
                samesite  = "Lax",
                secure    = True   # HTTPS only in production
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

            # Pending review — submitted but not released
            pending = conn.execute(text("""
                SELECT COUNT(*) FROM assignments
                WHERE status = 'graded'
            """)).fetchone()[0]

            # Open disputes
            disputes = conn.execute(text("""
                SELECT COUNT(*) FROM disputes WHERE status = 'open'
            """)).fetchone()[0]

            # Urgent disputes — raised more than 40h ago
            urgent = conn.execute(text("""
                SELECT COUNT(*) FROM disputes
                WHERE status = 'open'
                AND DATEDIFF(hour, raised_at, GETUTCDATE()) >= 40
            """)).fetchone()[0]

            # Total students assigned this month
            students = conn.execute(text("""
                SELECT COUNT(DISTINCT student_id) FROM assignments
                WHERE MONTH(created_at) = MONTH(GETUTCDATE())
                AND YEAR(created_at) = YEAR(GETUTCDATE())
            """)).fetchone()[0]

            # Active exams — due date in future
            active_exams = conn.execute(text("""
                SELECT COUNT(DISTINCT paper_id) FROM assignments
                WHERE due_date >= GETUTCDATE()
                AND status NOT IN ('released')
            """)).fetchone()[0]

            # Pending submissions detail
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

            # Active assignments with counts
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
