from flask import Flask, render_template_string, request, redirect, url_for, flash
from datetime import date, datetime
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json

# ---- Load environment variables from .env file (for local development) ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use system environment variables

# ---- OpenAI SDK ----
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

APP_DB = Path("finance.sqlite3")

app = Flask(__name__)
app.secret_key = os.getenv("OPENAI_API_KEY", "dev-key")

# In-memory chat log (ephemeral)
CHAT_LOG = []  # list[{"q": str, "a": str}]

# ---------------------- SQLite helpers ----------------------
def get_db():
    conn = sqlite3.connect(APP_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,        -- YYYY-MM-DD
            amount REAL NOT NULL,      -- positive
            category TEXT NOT NULL,
            description TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cur.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('starting_balance', '0.0')")
    conn.commit()
    conn.close()

def get_starting_balance():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key='starting_balance'")
    row = cur.fetchone()
    conn.close()
    return round(float(row["value"]) if row and row["value"] else 0.0, 2)

def set_starting_balance(v: float):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE settings SET value=? WHERE key='starting_balance'", (str(round(v, 2)),))
    conn.commit()
    conn.close()

def insert_purchase(d):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO purchases(date, amount, category, description) VALUES(?,?,?,?)",
        (d["date"], d["amount"], d["category"], d["description"]),
    )
    conn.commit()
    conn.close()

def fetch_purchases():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, date, amount, category, description FROM purchases ORDER BY date DESC, id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def totals():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(SUM(amount), 0) AS spent FROM purchases")
    spent = float(cur.fetchone()["spent"] or 0.0)
    conn.close()
    total_spent = round(spent, 2)
    current_balance = round(get_starting_balance() - total_spent, 2)
    return total_spent, current_balance

# ---- NEW: build a compact context JSON for the model ----
def get_finance_context(limit: int = 200) -> dict:
    """
    Returns a small JSON-serializable dict with:
      - starting_balance, total_spent, current_balance
      - by_category: {category: sum}
      - by_date: [{date: 'YYYY-MM-DD', total: float}, ...] ascending by date
      - recent_purchases: last N purchases (most recent first)
    """
    rows = fetch_purchases()
    total_spent, current_balance = totals()
    starting_balance = get_starting_balance()

    # by_category
    cat_totals = {}
    for r in rows:
        cat_totals[r["category"]] = cat_totals.get(r["category"], 0.0) + float(r["amount"])

    # by_date
    date_totals = {}
    for r in rows:
        d = r["date"]
        date_totals[d] = date_totals.get(d, 0.0) + float(r["amount"])
    by_date = [{"date": d, "total": round(date_totals[d], 2)} for d in sorted(date_totals.keys())]

    # recent N purchases
    recent = []
    for r in rows[:max(0, limit)]:
        recent.append({
            "id": r["id"],
            "date": r["date"],
            "amount": float(r["amount"]),
            "category": r["category"],
            "description": r["description"] or ""
        })

    return {
        "starting_balance": starting_balance,
        "total_spent": total_spent,
        "current_balance": current_balance,
        "by_category": {k: round(v, 2) for k, v in cat_totals.items()},
        "by_date": by_date,
        "recent_purchases": recent,
        "row_count": len(rows),
        "recent_included": len(recent)
    }

# ---- Call OpenAI with optional context ----
def ask_openai(question: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Calls OpenAI and returns the model's text. If context is provided, it is
    included as JSON so the model can analyze the user's finance data.
    """
    if not _OPENAI_AVAILABLE:
        return "OpenAI SDK not installed. Run: pip install openai"
    if not os.environ.get("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY is not set. "
            "Set it via: export OPENAI_API_KEY='sk-...' (local) "
            "or add it as a GitHub Secret for CI/CD workflows."
        )

    client = OpenAI()  # reads OPENAI_API_KEY
    try:
        # Prepare a single textual input that includes context + question
        context_block = ""
        if context:
            # Keep it compact; stringify JSON
            context_block = (
                "Here is the user's finance data as JSON.\n"
                "Use it to answer the question. If something is missing, say so.\n"
                f"FINANCE_JSON_BEGIN\n{json.dumps(context, ensure_ascii=False)}\nFINANCE_JSON_END\n\n"
            )

        prompt = (
            "You are a helpful finance analyst. Be concrete and cite numbers from the data.\n"
            "If you propose budgets or savings, show a short table or bullet list.\n\n"
            f"{context_block}"
            f"User question: {question}\n"
        )

        resp = client.responses.create(
            model="gpt-4o-mini",
            instructions="Answer clearly and concisely, using the provided JSON if present.",
            input=prompt
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()
        return " ".join(
            getattr(item, "text", "")
            for item in getattr(resp, "output", []) if getattr(item, "type", "") == "output_text"
        ).strip() or "[No text in response]"
    except Exception as e:
        return f"[OpenAI error] {e}"

# ---------------------- Template (Finance + Chat) ----------------------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Personal Finance (SQLite + Chat)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
    body { margin: 0; background:#f6f7fb; color:#111; }
    .container { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
    h1 { margin: 0 0 1rem; }
    .grid { display: grid; gap: 1rem; grid-template-columns: 1fr; }
    @media (min-width: 1050px) { .grid { grid-template-columns: 1fr 1fr; } }
    .card { background: #fff; border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,.08); padding: 1rem; }
    label { display:block; font-weight:600; margin:.5rem 0 .25rem; }
    input, select, textarea { width: 100%; padding:.6rem .7rem; border:1px solid #cfd4dc; border-radius:10px; background:#fff; box-sizing:border-box; }
    button { appearance:none; border:0; background:#111; color:#fff; padding:.7rem 1rem; border-radius:10px; cursor:pointer; font-weight:600; }
    .secondary { background:#e5e7eb; color:#111; }
    .row { display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; }
    .pill { background:#111; color:#fff; padding:.35rem .6rem; border-radius:999px; font-variant-numeric: tabular-nums; }
    .muted { color:#6b7280; }
    table { width:100%; border-collapse: collapse; }
    th, td { padding:.6rem .5rem; border-bottom:1px solid #eef0f4; text-align:left; }
    th { font-size:.9rem; color:#6b7280; font-weight:600; }
    .right { text-align:right; }
    .flash { background:#fef3c7; border:1px solid #fde68a; padding:.6rem .8rem; border-radius:10px; margin-bottom:.75rem; }
    .chat-log { display:flex; flex-direction:column; gap:.6rem; max-height: 320px; overflow:auto; padding-right:.25rem; }
    .bubble { padding:.6rem .8rem; border-radius:14px; }
    .user { background:#eef2ff; align-self:flex-end; }
    .assistant { background:#ecfeff; align-self:flex-start; }
    .small { font-size:.85rem; }
    .inline { display:inline-flex; align-items:center; gap:.5rem; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Personal Financial Management — SQLite + OpenAI</h1>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for m in messages %}<div class="flash">{{ m }}</div>{% endfor %}
      {% endif %}
    {% endwith %}

    <div class="grid">
      <!-- Finance: account + add form -->
      <div class="card">
        <h2 style="margin-top:0;">Account Balance</h2>
        <form method="post" action="{{ url_for('set_balance') }}" class="row" style="gap:.6rem;">
          <label for="starting_balance">Starting balance</label>
          <div class="row" style="width:100%; gap:.6rem;">
            <input type="number" id="starting_balance" name="starting_balance" step="0.01" min="0" value="{{ starting_balance }}">
            <button type="submit">Set</button>
            <button class="secondary" type="button" onclick="location.href='{{ url_for('reset') }}'">Reset All</button>
          </div>
        </form>
        <div class="row" style="gap:1rem; margin-top:1rem; flex-wrap:wrap;">
          <div><span class="muted">Total Spent:</span> <span class="pill">${{ '%.2f' % total_spent }}</span></div>
          <div><span class="muted">Current Balance:</span> <span class="pill">${{ '%.2f' % current_balance }}</span></div>
          <div><span class="muted"># Purchases:</span> <span class="pill">{{ purchases|length }}</span></div>
        </div>
      </div>

      <div class="card">
        <h2 style="margin-top:0;">Add Purchase</h2>
        <form method="post" action="{{ url_for('index') }}">
          <label for="date">Date</label>
          <input type="date" id="date" name="date" value="{{ today }}" required>

          <label for="amount">Amount</label>
          <input type="number" id="amount" name="amount" step="0.01" min="0.01" required>

          <label for="category">Category</label>
          <select id="category" name="category" required>
            <option value="" disabled selected>Select a category</option>
            {% for c in ['Groceries','Dining','Transport','Housing','Utilities','Health','Entertainment','Education','Other'] %}
              <option>{{ c }}</option>
            {% endfor %}
          </select>

          <label for="description">Description (optional)</label>
          <input type="text" id="description" name="description" placeholder="e.g., Trader Joe's, Lyft to airport">

          <div class="row" style="margin-top:.8rem;">
            <button type="submit">Save Purchase</button>
            <button type="reset" class="secondary">Clear</button>
          </div>
        </form>
      </div>
    </div>

    <!-- Purchases -->
    <div class="card" style="margin-top:1rem;">
      <h2 style="margin-top:0;">Past Purchases</h2>
      {% if purchases %}
        <table>
          <thead><tr><th>Date</th><th>Category</th><th>Description</th><th class="right">Amount</th></tr></thead>
          <tbody>
            {% for p in purchases %}
              <tr>
                <td>{{ p.date }}</td>
                <td>{{ p.category }}</td>
                <td>{{ p.description or '-' }}</td>
                <td class="right">${{ '%.2f' % p.amount }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <p class="muted">No purchases yet. Add your first one using the form above.</p>
      {% endif %}
      <p class="muted small" style="margin-top:.5rem;">SQLite file: <code>{{ db_path }}</code></p>
    </div>

    <!-- Conversation box -->
    <div class="card" style="margin-top:1rem;">
      <h2 style="margin-top:0;">Conversation (OpenAI)</h2>
      <form method="post" action="{{ url_for('ask') }}">
        <label for="question">Ask a question</label>
        <textarea id="question" name="question" rows="3" placeholder="e.g., Analyze my spending and suggest a monthly budget." required></textarea>
        <label class="inline" style="margin-top:.5rem;">
          <input type="checkbox" name="include_data" checked>
          Include my finance data in this question
        </label>
        <div class="row" style="margin-top:.8rem;">
          <button type="submit">Ask</button>
          <button type="reset" class="secondary">Clear</button>
        </div>
      </form>

      <div class="chat-log" style="margin-top:1rem;">
        {% if chat_log %}
          {% for item in chat_log %}
            <div class="bubble user"><strong>You:</strong> {{ item.q }}</div>
            <div class="bubble assistant"><strong>AI:</strong> {{ item.a|safe }}</div>
          {% endfor %}
        {% else %}
          <p class="muted">No messages yet.</p>
        {% endif %}
      </div>
      <p class="muted small">Note: chat history here is in-memory; it clears when the server restarts.</p>
    </div>

  </div>
</body>
</html>
"""

# ---------------------- Routes ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            raw_date = request.form.get("date") or date.today().isoformat()
            datetime.strptime(raw_date, "%Y-%m-%d")
            amount = float((request.form.get("amount") or "").strip())
            if amount <= 0:
                raise ValueError("Amount must be positive.")
            category = (request.form.get("category") or "").strip()
            if not category:
                raise ValueError("Category is required.")
            description = (request.form.get("description") or "").strip()
            insert_purchase({"date": raw_date, "amount": amount, "category": category, "description": description})
            flash("Purchase saved.")
        except ValueError as e:
            flash(f"Error: {e}")
        except Exception:
            flash("Something went wrong while saving the purchase.")
        return redirect(url_for("index"))

    rows = fetch_purchases()
    total_spent, current_balance = totals()

    class Row:
        def __init__(self, r): self.__dict__.update(dict(r))

    return render_template_string(
        TEMPLATE,
        purchases=[Row(r) for r in rows],
        total_spent=total_spent,
        current_balance=current_balance,
        starting_balance=get_starting_balance(),
        today=date.today().isoformat(),
        db_path=str(APP_DB.resolve()),
        chat_log=CHAT_LOG
    )

@app.post("/ask")
def ask():
    question = (request.form.get("question") or "").strip()
    include_data = request.form.get("include_data") == "on"
    if not question:
        flash("Please enter a question.")
        return redirect(url_for("index"))

    context = get_finance_context(limit=200) if include_data else None
    answer = ask_openai(question, context)

    CHAT_LOG.append({"q": question, "a": answer})
    del CHAT_LOG[:-10]  # keep last 10
    return redirect(url_for("index"))

@app.post("/set-balance")
def set_balance():
    try:
        sb = float(request.form.get("starting_balance", "0").strip())
        if sb < 0:
            raise ValueError("Starting balance cannot be negative.")
        set_starting_balance(sb)
        flash("Starting balance updated.")
    except ValueError as e:
        flash(f"Error: {e}")
    except Exception:
        flash("Unable to set starting balance.")
    return redirect(url_for("index"))

@app.get("/reset")
def reset():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM purchases")
        cur.execute("UPDATE settings SET value='0.0' WHERE key='starting_balance'")
        conn.commit()
        conn.close()
        CHAT_LOG.clear()
        flash("All data cleared.")
    except Exception:
        flash("Reset failed.")
    return redirect(url_for("index"))

# ---------------------- Bootstrap & run ----------------------
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)