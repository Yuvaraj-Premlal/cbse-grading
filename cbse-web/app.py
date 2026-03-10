from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from passlib.context import CryptContext
from dotenv import load_dotenv
import urllib, os

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("JWT_SECRET", "dev-secret-key")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

# ── DATABASE ──────────────────────────────────────────
def get_engine():
    params = urllib.parse.quote_plus(secrets["db"])
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        pool_pre_ping=True
    )

# ── ROUTES ────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    results = {}

    # Key Vault check
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(
            vault_url=os.getenv("KEY_VAULT_URL"),
            credential=credential
        )
        client.get_secret("JWT-SECRET")
        results["keyvault"] = {"ok": True, "status": "Connected"}
    except Exception as e:
        results["keyvault"] = {"ok": False, "status": str(e)[:120]}

    # Database check
    try:
        engine = get_engine()
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM users")
            ).fetchone()[0]
            results["database"] = {
                "ok": True,
                "status": f"Connected — {count} users in DB"
            }
    except Exception as e:
        results["database"] = {"ok": False, "status": str(e)[:120]}

    # Tables check
    try:
        engine = get_engine()
        with engine.connect() as conn:
            tables = conn.execute(text("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)).fetchall()
            results["tables"] = {
                "ok": True,
                "status": f"{len(tables)} tables found",
                "list": [t[0] for t in tables]
            }
    except Exception as e:
        results["tables"] = {"ok": False, "status": str(e)[:120]}

    return jsonify(results)

# ── REGISTER ──────────────────────────────────────────
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    try:
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
                "name":  data["name"],
                "email": data["email"],
                "hash":  pwd_context.hash(data["password"]),
                "role":  data["role"]
            })

        return jsonify({
            "ok": True,
            "message": f"✅ User {data['email']} created as {data['role']}"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})

# ── LOGIN ─────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    data = request.json
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
            if not pwd_context.verify(data["password"], user.password_hash):
                return jsonify({"ok": False, "error": "Wrong password"})

            return jsonify({
                "ok": True,
                "message": f"✅ Welcome {user.name}",
                "user": {
                    "name":  user.name,
                    "email": user.email,
                    "role":  user.role
                }
            })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
