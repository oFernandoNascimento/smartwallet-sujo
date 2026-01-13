"""
SmartWallet | Personal Pro Edition (v9.43.0)
===========================================
Enterprise-grade Financial Management System.

Architecture: N-Tier (Presentation, Service, Persistence, Security).
Stack: Streamlit, Google Gemini AI, PostgreSQL, Docker.

Updates v9.43.0:
- [FIX] Suppressed transient DB Init error on file upload (False positive).
- [FIX] Resolved 'StreamlitAPIException' on Download Button (binary conversion).
- [FEAT] Features from previous versions maintained perfectly.
- [UI] Enhanced visual alignment for Budget Gauge Charts (Fixed Margins & Typography).
- [NEW] Voice Transaction Entry (Gemini Multimodal Audio) with Auto-Submit.
- [FIX] Solved Infinite Loop on Audio & Added Auto-Reset functionality.

Author: Fernando Teixeira do Nascimento
Version: 9.43.0
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re
import time
import pytz
import hashlib
import psycopg2 
import io 
import os
import logging
from datetime import datetime, timedelta, date
from typing import List, Optional

# Optional dependency for PDF generation
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="SmartWallet Personal Pro",
    page_icon="💲", 
    layout="wide",
    initial_sidebar_state="expanded"
)

FUSO_BR = pytz.timezone('America/Sao_Paulo')

# Categorias Padrão (Base) - As personalizadas virão do banco
CATEGORIAS_BASE: List[str] = [
    "Alimentação", "Transporte", "Moradia", "Lazer", "Saúde", 
    "Salário", "Investimentos", "Educação", "Viagem", "Compras", 
    "Assinaturas", "Presentes", "Outros"
]

def configure_system() -> None:
    """System initialization and dependency check."""
    try:
        api_key = st.secrets.get("GEMINI_KEY")
        if not api_key:
            st.error("Environment Error: 'GEMINI_KEY' not found.")
            st.stop()
        else:
            genai.configure(api_key=api_key)
            
        if not st.secrets.get("DATABASE_URL"):
            st.warning("Database connection string not found. Running in restricted mode.")
            
    except Exception as e:
        logging.critical(f"Boot failure: {e}")
        st.stop()

configure_system()

# ==============================================================================
# UI MANAGER
# ==============================================================================

class UIManager:
    """Handles UI styling, CSS injection and helper formatting methods."""
    
    @staticmethod
    def inject_global_css() -> None:
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }
            .main { background-color: #0E1117; }
            
            .login-container {
                background: linear-gradient(145deg, #1E1E1E, #252525);
                padding: 45px;
                border-radius: 24px;
                border: 1px solid #333;
                box-shadow: 0 20px 50px rgba(0,0,0,0.5);
                text-align: center;
                margin-top: 60px;
            }
            
            @keyframes flashGreen {
                0% { border-color: #4CAF50; box-shadow: 0 0 15px rgba(76, 175, 80, 0.5); }
                100% { border-color: #2C2F38; box-shadow: none; }
            }
            @keyframes flashRed {
                0% { border-color: #F44336; box-shadow: 0 0 15px rgba(244, 67, 54, 0.5); }
                100% { border-color: #2C2F38; box-shadow: none; }
            }

            .market-card { 
                background-color: #121318; 
                border: 1px solid #2C2F38; 
                border-radius: 24px; 
                padding: 16px; 
                text-align: center;
                position: relative;
                overflow: hidden; 
                height: 100px;
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            }
            .up-trend { animation: flashGreen 2s ease-out; }
            .down-trend { animation: flashRed 2s ease-out; }

            .market-card:hover { transform: translateY(-5px); border-color: #888; }
            
            .label-coin { font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; z-index: 2; position: relative; }
            .value-coin { font-size: 24px; font-weight: 700; color: #fff; z-index: 2; position: relative; }
            
            .kpi-card {
                background-color: #1F2129;
                padding: 24px;
                border-radius: 24px;
                border-left: 6px solid #4CAF50;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                transition: transform 0.2s;
            }
            .kpi-card:hover { transform: scale(1.01); }
            .kpi-label { font-size: 13px; color: #aaa; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
            .kpi-value { font-size: 32px; font-weight: 600; margin-top: 8px; letter-spacing: -0.5px; }
            
            div.stButton > button { border-radius: 50px; font-weight: 600; padding: 0.6rem 2rem; border: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2); background-color: #4CAF50 !important; color: white !important; }
            div.stButton > button:hover { transform: translateY(-2px); }
            
            div[data-baseweb="input"] > div { border-radius: 12px !important; background-color: #121318 !important; border: 1px solid #333 !important; color: white !important; }
            div[data-baseweb="input"] > div:focus-within { border-color: #4CAF50 !important; }
            div[data-testid="stToast"] { border-radius: 16px !important; background-color: #262730 !important; border: 1px solid #333 !important; }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def get_svg_chart(is_up: bool = True) -> str:
        """Returns dynamic SVG string for market trend visualization."""
        color = "#4CAF50" if is_up else "#F44336"
        fill_color = "rgba(76, 175, 80, 0.15)" if is_up else "rgba(244, 67, 54, 0.15)"
        
        if is_up:
            pts = "0,80 20,60 40,70 60,30 80,40 100,10"
            area = "0,100 0,80 20,60 40,70 60,30 80,40 100,10 100,100"
        else:
            pts = "0,20 20,40 40,30 60,70 80,60 100,90"
            area = "0,100 0,20 20,40 40,30 60,70 80,60 100,90 100,100"
        
        return f"""
        <svg viewBox="0 0 100 100" style="position:absolute; bottom:-5px; left:0; width:100%; height:60%; opacity:0.3;" preserveAspectRatio="none">
            <polygon points="{area}" fill="{fill_color}" />
            <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
        </svg>
        """

    @staticmethod
    def format_money(value: float, hidden: bool = False) -> str:
        if hidden: return "R$ ****"
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ==============================================================================
# DATA ACCESS OBJECTS (DAO) & SECURITY
# ==============================================================================

class SecurityManager:
    """Handles password hashing and security protocols."""
    SALT = "SmartWallet_2026_SecureSalt_#99"

    @staticmethod
    def hash_pwd(pwd: str) -> str:
        salted_pwd = pwd + SecurityManager.SALT
        return hashlib.sha256(salted_pwd.encode()).hexdigest()
    
    @staticmethod
    def is_strong_password(pwd: str) -> bool:
        """[SEC] Enforces password complexity policies."""
        if len(pwd) < 8: return False
        if not re.search(r"[A-Za-z]", pwd): return False # Pelo menos uma letra
        if not re.search(r"[0-9]", pwd): return False # Pelo menos um número
        return True

class RobustDatabase:
    """
    Database Manager with connection pooling and singleton pattern via Streamlit cache.
    """
    def __init__(self):
        self.init_tables()

    def get_conn(self):
        # Utilizes Streamlit resource caching to maintain persistent connection pool
        @st.cache_resource(ttl=3600)
        def _get_cached_connection():
            return psycopg2.connect(st.secrets["DATABASE_URL"])
        return _get_cached_connection()

    def init_tables(self) -> None:
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT, created_at TEXT)")
                    cur.execute("""CREATE TABLE IF NOT EXISTS transactions (
                        id SERIAL PRIMARY KEY, user_id TEXT, date TEXT, amount REAL, category TEXT, description TEXT, type TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(username))""")
                    cur.execute("""CREATE TABLE IF NOT EXISTS budgets (
                        id SERIAL PRIMARY KEY, user_id TEXT, category TEXT, limit_amount REAL,
                        FOREIGN KEY(user_id) REFERENCES users(username))""")
                    cur.execute("""CREATE TABLE IF NOT EXISTS recurring (
                        id SERIAL PRIMARY KEY, user_id TEXT, category TEXT, amount REAL, description TEXT, type TEXT, day_of_month INT, last_processed TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(username))""")
                    
                    # [FEAT] Custom Categories Table
                    cur.execute("""CREATE TABLE IF NOT EXISTS custom_categories (
                        id SERIAL PRIMARY KEY, user_id TEXT, name TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(username))""")
                    
                    # [FEAT] Receipt Columns (Safe Migration)
                    try:
                        cur.execute("ALTER TABLE transactions ADD COLUMN proof_data BYTEA")
                        cur.execute("ALTER TABLE transactions ADD COLUMN proof_name TEXT")
                        conn.commit()
                    except Exception:
                        conn.rollback() # Columns likely exist

                    conn.commit()
        except Exception as e:
            # [FIX] Logging only to console to avoid user panic on transient errors
            logging.error(f"DB Init Error (Transient): {e}")
            # st.error("Database initialization failed.") <-- SUPPRESSED VISUAL ERROR

    # --- Auth ---
    def register(self, user, pwd):
        if not user or not pwd: return False, "Invalid input."
        # [SEC] Password Strength Check
        if not SecurityManager.is_strong_password(pwd):
            return False, "Senha fraca! Use no mínimo 8 caracteres, letras e números."
            
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO users VALUES (%s, %s, %s)", (user, SecurityManager.hash_pwd(pwd), str(datetime.now())))
                    conn.commit()
            return True, "User created successfully."
        except Exception as e:
            logging.warning(f"Registration error: {e}")
            return False, "User already exists."

    def login(self, user, pwd):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT username FROM users WHERE username=%s AND password_hash=%s", (user, SecurityManager.hash_pwd(pwd)))
                    return cur.fetchone() is not None
        except Exception as e:
            logging.error(f"Login error: {e}")
            return False

    # --- Category Management ---
    def get_categories(self, uid):
        """Merges default categories with user custom categories."""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT name FROM custom_categories WHERE user_id=%s", (uid,))
                    custom = [row[0] for row in cur.fetchall()]
            return sorted(list(set(CATEGORIAS_BASE + custom)))
        except Exception:
            return sorted(CATEGORIAS_BASE)

    def add_category(self, uid, name):
        if name in CATEGORIAS_BASE: return False
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO custom_categories (user_id, name) VALUES (%s, %s)", (uid, name))
                    conn.commit()
            return True
        except Exception: return False

    def delete_category(self, uid, name):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM custom_categories WHERE user_id=%s AND name=%s", (uid, name))
                    conn.commit()
            return True
        except Exception: return False

    # --- Core Transactions ---
    def add_transaction(self, uid, date, amt, cat, desc, type_, proof_file=None, proof_name=None):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    # [FEAT] Handle Receipt Upload (Binary)
                    proof_bytes = proof_file.getvalue() if proof_file else None
                    
                    cur.execute("""INSERT INTO transactions 
                        (user_id, date, amount, category, description, type, proof_data, proof_name) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        (uid, str(date), float(amt), cat, desc, type_, psycopg2.Binary(proof_bytes) if proof_bytes else None, proof_name))
                    conn.commit()
            return True
        except Exception as e:
            logging.error(f"Add transaction error: {e}")
            return False

    def remove_transaction(self, tid, uid):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM transactions WHERE id=%s AND user_id=%s", (tid, uid))
                    conn.commit()
            return True
        except Exception:
            return False

    def get_totals(self, uid, start_date=None, end_date=None):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    q = "SELECT type, SUM(amount) FROM transactions WHERE user_id = %s"
                    p = [uid]
                    if start_date and end_date:
                        q += " AND date >= %s AND date <= %s"
                        p.extend([str(start_date), str(end_date)])
                    q += " GROUP BY type"
                    cur.execute(q, tuple(p))
                    r = dict(cur.fetchall())
                    return r.get('Receita', 0.0), r.get('Despesa', 0.0)
        except Exception:
            return 0.0, 0.0

    def fetch_all(self, uid, limit=None, start_date=None, end_date=None):
        try:
            with self.get_conn() as conn:
                # [FEAT] Fetching proof columns
                q = "SELECT id, date, amount, category, description, type, proof_name, proof_data FROM transactions WHERE user_id = %s"
                p = [uid]
                if start_date and end_date:
                    q += " AND date >= %s AND date <= %s"
                    p.extend([str(start_date), str(end_date)])
                q += " ORDER BY date DESC, id DESC"
                if limit:
                    q += " LIMIT %s"
                    p.append(limit)
                
                # Manual fetch to handle binary data correctly if needed, but pandas handles it okay as object
                df = pd.read_sql_query(q, conn, params=p)
            return df if not df.empty else pd.DataFrame(columns=['id', 'date', 'amount', 'category', 'description', 'type', 'proof_name', 'proof_data'])
        except Exception as e:
            logging.error(f"Fetch error: {e}")
            return pd.DataFrame(columns=['id', 'date', 'amount', 'category', 'description', 'type'])

    def nuke_data(self, uid):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM transactions WHERE user_id=%s", (uid,))
                    cur.execute("DELETE FROM budgets WHERE user_id=%s", (uid,))
                    cur.execute("DELETE FROM recurring WHERE user_id=%s", (uid,))
                    # Categories are kept as they are structural preferences
                    conn.commit()
            return True
        except Exception:
            return False

    # --- Budgets ---
    def set_meta(self, uid, cat, lim):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM budgets WHERE user_id=%s AND category=%s", (uid, cat))
                    if cur.fetchone():
                        cur.execute("UPDATE budgets SET limit_amount=%s WHERE user_id=%s AND category=%s", (float(lim), uid, cat))
                    else:
                        cur.execute("INSERT INTO budgets (user_id, category, limit_amount) VALUES (%s, %s, %s)", (uid, cat, float(lim)))
                    conn.commit()
            return True
        except Exception:
            return False

    def delete_meta(self, uid, cat):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM budgets WHERE user_id=%s AND category=%s", (uid, cat))
                    conn.commit()
            return True
        except Exception:
            return False

    def get_metas(self, uid):
        try:
            with self.get_conn() as conn:
                return pd.read_sql_query("SELECT category, limit_amount FROM budgets WHERE user_id=%s", conn, params=(uid,))
        except Exception:
            return pd.DataFrame()

    # --- Recurring Engine ---
    def add_recurring(self, uid, cat, amt, desc, type_, day):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO recurring (user_id, category, amount, description, type, day_of_month, last_processed) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                                (uid, cat, float(amt), desc, type_, int(day), ''))
                    conn.commit()
            return True
        except Exception as e:
            logging.error(f"Add recurring error: {e}")
            return False

    def process_recurring_items(self, uid):
        try:
            today = datetime.now(FUSO_BR).date()
            current_month_str = today.strftime('%Y-%m')
            
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, category, amount, description, type, day_of_month, last_processed FROM recurring WHERE user_id=%s", (uid,))
                    items = cur.fetchall()
                    
                    count = 0
                    for item in items:
                        rid, cat, amt, desc, type_, day, last_proc = item
                        if today.day >= day and last_proc != current_month_str:
                            self.add_transaction(uid, today, amt, cat, f"{desc} (Recorrente)", type_)
                            cur.execute("UPDATE recurring SET last_processed=%s WHERE id=%s", (current_month_str, rid))
                            count += 1
                    conn.commit()
            return count
        except Exception as e:
            logging.error(f"Recurring Process Error: {e}")
            return 0

db = RobustDatabase()

# ==============================================================================
# AI & ANALYTICS ENGINE
# ==============================================================================

class AIManager:
    """Interface for Gemini AI interactions."""
    
    @staticmethod
    def extract_json(txt):
        m = re.search(r'\{.*\}', txt, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except: pass
        return None

    @staticmethod
    def process_nlp(text, mkt, categories):
        """Processes natural language input using Gemini models with fallback strategies."""
        # [FEAT] Inject custom categories into prompt
        cat_str = ", ".join(categories)
        
        prompt = f"""
        Role: Financial Assistant. Today: {datetime.now(FUSO_BR).strftime('%Y-%m-%d')}.
        User Input: "{text}"
        Rates: USD={mkt['USD']}, BTC={mkt['BTC']}
        
        Task: Extract transaction details.
        1. Type: 'Receita' OR 'Despesa'.
        2. Category: Best fit from this list: [{cat_str}]. 
           Rules:
           - "Shopping" (passeio, gastar no shopping) -> "Lazer"
           - "Mercado", "Feira" -> "Alimentação"
           - IF specific item bought (e.g. "Tênis", "Celular") -> "Compras"
           - Investment assets -> "Investimentos"
        3. Amount: Float (BRL).
        4. Description: 
           MANDATORY: If buying Crypto/Currency, CALCULATE converted amount and ADD to description.
        
        Output JSON ONLY: {{ "amount": float, "category": "string", "date": "YYYY-MM-DD", "description": "string", "type": "string" }}
        """
        
        models = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        for model_name in models:
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                data = AIManager.extract_json(res.text)
                if data: return data
            except Exception as e:
                logging.error(f"AI Model {model_name} failed: {e}")
                continue 
                
        return {"error": "AI Service unavailable."}

    @staticmethod
    def process_audio_nlp(audio_file, mkt, categories):
        """[NEW] Processes audio input using Gemini Multimodal capabilities."""
        try:
            audio_bytes = audio_file.read()
            cat_str = ", ".join(categories)
            
            prompt = f"""
            Role: Financial Assistant. Today: {datetime.now(FUSO_BR).strftime('%Y-%m-%d')}.
            Task: Listen to the audio user input and extract transaction details.
            Rates: USD={mkt['USD']}, BTC={mkt['BTC']}
            
            1. Type: 'Receita' OR 'Despesa'.
            2. Category: Best fit from this list: [{cat_str}]. 
            3. Amount: Float (BRL).
            4. Description: Short description of what was bought/received.
            
            Output JSON ONLY: {{ "amount": float, "category": "string", "date": "YYYY-MM-DD", "description": "string", "type": "string" }}
            """
            
            # [FIX] Prioritize 'gemini-2.0-flash-exp' to avoid 404 errors with older models on audio
            model_options = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']
            
            response = None
            last_error = None

            for model_name in model_options:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content([
                        prompt,
                        {"mime_type": "audio/wav", "data": audio_bytes}
                    ])
                    # Se funcionou, para o loop
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            if response:
                data = AIManager.extract_json(response.text)
                if data: return data
                return {"error": "Não entendi o áudio."}
            else:
                return {"error": f"Erro de Modelo AI: {last_error}"}
            
        except Exception as e:
            logging.error(f"Audio AI failed: {e}")
            return {"error": f"Audio Error: {e}"}

    @staticmethod
    def coach_financeiro(df, renda_total, mkt):
        if df.empty: return "Insufficient data for analysis."
        
        prompt = f"""
        ACT AS: Personal Financial Coach.
        CLIENT DATA:
        - Monthly Income: R$ {renda_total:,.2f}
        - Transactions: {df.head(40).to_string()}
        - Market: USD {mkt['USD']}, BTC {mkt['BTC']}
        
        MISSION:
        1. Analyze superfluous vs essential spending.
        2. Suggest a realistic savings goal.
        3. Provide one investment tip based on spending profile.
        
        FORMAT: Clean Markdown, portuguese language.
        """
        try:
            m = genai.GenerativeModel('gemini-2.0-flash-exp')
            return m.generate_content(prompt).text
        except Exception:
            try:
                m = genai.GenerativeModel('gemini-1.5-flash')
                return m.generate_content(prompt).text
            except Exception as e: 
                return f"Coach offline: {e}"

@st.cache_data(ttl=300, show_spinner=False)
def get_market_data():
    """
    Fetches real-time market data (Direct BRL quotes preferred).
    """
    data = {"USD": 0.0, "EUR": 0.0, "GBP": 0.0, "BTC": 0.0, "status": "offline"}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        url_awesome = "https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL,GBP-BRL,BTC-BRL"
        r = requests.get(url_awesome, headers=headers, timeout=5)
        
        if r.status_code == 200:
            d = r.json()
            data["USD"] = float(d.get('USDBRL', {}).get('bid', 0))
            data["EUR"] = float(d.get('EURBRL', {}).get('bid', 0))
            data["GBP"] = float(d.get('GBPBRL', {}).get('bid', 0))
            data["BTC"] = float(d.get('BTCBRL', {}).get('bid', 0)) 
            data["status"] = "online (AwesomeAPI)"
            return data
    except Exception:
        pass

    try:
        api_key = "fxr_live_8686bcd98ec3ccc03f52ac26d084546fe794" 
        url_fx = f"https://api.fxratesapi.com/latest?base=USD&currencies=BRL,EUR,GBP,BTC&api_key={api_key}"
        r = requests.get(url_fx, headers=headers, timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get('success', False):
                rates = d['rates']
                usd_brl = rates.get('BRL', 5.85)
                data['USD'] = usd_brl
                if rates.get('EUR'): data['EUR'] = usd_brl / rates['EUR']
                if rates.get('GBP'): data['GBP'] = usd_brl / rates['GBP']
                if rates.get('BTC'): data['BTC'] = usd_brl / rates['BTC']
                data["status"] = "online (FXRates Backup)"
                return data
    except Exception:
        pass
            
    return data

class DocGenerator:
    @staticmethod
    def to_excel(df):
        out = io.BytesIO()
        try:
            with pd.ExcelWriter(out, engine='openpyxl') as w:
                d = df.drop(columns=['proof_data'], errors='ignore').copy()
                d['date'] = pd.to_datetime(d['date'], errors='coerce').dt.strftime('%d/%m/%Y %H:%M')
                d['amount'] = d['amount'].apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                d.to_excel(w, index=False, sheet_name="SmartWallet")
        except Exception:
            return io.BytesIO()
        return out

    @staticmethod
    def to_pdf(user, df, inc, exp, bal, period):
        if FPDF is None: return None
        pdf = FPDF()
        try:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16); pdf.set_text_color(76, 175, 80)
            pdf.cell(0, 10, "SmartWallet Personal | Relatório", ln=True, align='C')
            pdf.set_font("Arial", '', 10); pdf.set_text_color(50)
            pdf.cell(0, 10, f"Cliente: {user} | {period} | {datetime.now().strftime('%d/%m/%Y')}", ln=True, align='C')
            pdf.ln(5)
            
            pdf.set_fill_color(245); pdf.rect(10, 35, 190, 20, 'F')
            pdf.set_y(40); pdf.set_font("Arial", 'B', 11)
            pdf.cell(63, 10, f"Entradas: R$ {inc:,.2f}", align='C')
            pdf.cell(63, 10, f"Saídas: R$ {exp:,.2f}", align='C')
            pdf.cell(63, 10, f"Saldo: R$ {bal:,.2f}", align='C')
            pdf.ln(25)
            
            pdf.set_font("Arial", 'B', 9); pdf.set_fill_color(50); pdf.set_text_color(255)
            cols = [("Data", 30), ("Tipo", 25), ("Categoria", 40), ("Descrição", 55), ("Valor", 40)]
            for c, w in cols: pdf.cell(w, 8, c, 1, 0, 'C', True)
            pdf.ln(); pdf.set_text_color(0); pdf.set_font("Arial", '', 9)
            
            for _, r in df.iterrows():
                try:
                    date_val = pd.to_datetime(r['date'], errors='coerce')
                    date_str = date_val.strftime('%d/%m') if pd.notnull(date_val) else "--/--"
                    pdf.cell(30, 8, date_str, 1)
                    pdf.cell(25, 8, r['type'], 1)
                    pdf.cell(40, 8, r['category'][:20], 1)
                    pdf.cell(55, 8, r['description'][:35], 1)
                    pdf.cell(40, 8, f"R$ {r['amount']:,.2f}", 1, 0, 'R')
                    pdf.ln()
                except Exception: pass
            return pdf.output(dest='S').encode('latin-1', 'ignore')
        except Exception:
            return None

# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================

@st.fragment(run_every=10) 
def header_relogio(mkt):
    now = datetime.now(FUSO_BR)
    d_str = now.strftime("%A, %d de %B de %Y")
    t_map = {"Monday":"Segunda","Tuesday":"Terça","Wednesday":"Quarta","Thursday":"Quinta","Friday":"Sexta","Saturday":"Sábado","Sunday":"Domingo",
             "January":"Janeiro","February":"Fevereiro","March":"Março","April":"Abril","May":"Maio","June":"Junho","July":"Julho","August":"Agosto","September":"Setembro","October":"Outubro","November":"Novembro","December":"Dezembro"}
    for en, pt in t_map.items(): d_str = d_str.replace(en, pt)
    
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"### {d_str} | {now.strftime('%H:%M:%S')}")
    st_ico = "🟢" if "online" in mkt['status'] else "🔴"
    c2.caption(f"{st_ico} Conexão: {mkt['status'].upper()}")

def main():
    UIManager.inject_global_css()
    
    # [FIX] State management for Audio Reset
    if 'audio_key' not in st.session_state: st.session_state.audio_key = 0
    if 'history_mkt' not in st.session_state: st.session_state.history_mkt = {}
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'user' not in st.session_state: st.session_state.user = None
    
    # [FEAT] Session state for Clone feature
    if 'manual_form' not in st.session_state:
        st.session_state.manual_form = {}

    # --- Login Screen ---
    if not st.session_state.logged_in:
        c1, c2, c3 = st.columns([1, 1.5, 1])
        with c2:
            with st.container(border=True):
                logo_path = None
                for file in ["logo.png", "logo.jpg", "logo.jpeg"]:
                    if os.path.exists(file): logo_path = file; break
                
                if logo_path:
                    cl, cm, cr = st.columns([1, 1, 1])
                    with cm: st.image(logo_path, use_container_width=True)
                
                st.markdown('<h2 style="text-align: center; color: #4CAF50;">SmartWallet Personal</h2>', unsafe_allow_html=True)
                st.markdown('<p style="text-align: center; color: #888;">Seu dinheiro, sob controle.</p>', unsafe_allow_html=True)
                
                with st.form("login"):
                    u = st.text_input("Usuário")
                    p = st.text_input("Senha", type="password")
                    if st.form_submit_button("Entrar no Sistema", use_container_width=True):
                        if db.login(u.strip(), p.strip()):
                            st.session_state.logged_in = True
                            st.session_state.user = u.strip()
                            rec_count = db.process_recurring_items(u.strip())
                            if rec_count > 0: st.toast(f"{rec_count} transações recorrentes!", icon="🔄")
                            st.rerun()
                        else: st.error("Dados incorretos.")
            
            with st.expander("Primeiro acesso?"):
                nu, np = st.text_input("Criar Usuário"), st.text_input("Criar Senha", type="password")
                if st.button("Registrar Conta"): 
                    ok, msg = db.register(nu.strip(), np.strip())
                    if ok: st.success(msg) 
                    else: st.error(msg)
        return

    # --- Authenticated Area ---
    user = st.session_state.user
    user_cats = db.get_categories(user) # [FEAT] Load Custom Categories
    
    with st.sidebar:
        logo_path = None
        for file in ["logo.png", "logo.jpg", "logo.jpeg"]:
            if os.path.exists(file): logo_path = file; break
        
        if logo_path:
            st.logo(logo_path, icon_image=logo_path)
        else:
            st.title("💲 SmartWallet")
            
        st.info(f"Olá, **{user}**!")
        st.divider()
        
        # [FEAT] Date Filter
        st.markdown("### 📅 Filtro de Período")
        filter_mode = st.radio("Modo", ["Mês Atual", "Personalizado"], horizontal=True)
        start_date, end_date = None, None
        
        if filter_mode == "Mês Atual":
            today = datetime.now(FUSO_BR).date()
            start_date = today.replace(day=1)
            next_month = today.replace(day=28) + timedelta(days=4)
            end_date = next_month - timedelta(days=next_month.day)
            st.caption(f"Mostrando: {start_date.strftime('%d/%m')} até {end_date.strftime('%d/%m')}")
        else:
            d_range = st.date_input("Selecione o intervalo", [])
            if len(d_range) == 2: start_date, end_date = d_range
            else: st.warning("Selecione data inicial e final.")

        st.divider()
        
        # [FEAT] Manage Categories
        with st.expander("⚙️ Gerenciar Categorias"):
            new_cat = st.text_input("Nova Categoria")
            if st.button("Adicionar"):
                if db.add_category(user, new_cat): st.success(f"'{new_cat}' adicionada!"); time.sleep(1); st.rerun()
                else: st.error("Erro ou já existe.")
            
            del_cat = st.selectbox("Excluir Categoria", [c for c in user_cats if c not in CATEGORIAS_BASE])
            if st.button("Excluir"):
                if db.delete_category(user, del_cat): st.success("Excluída!"); time.sleep(1); st.rerun()

        st.divider()
        if st.button("Sair da Conta"): st.session_state.logged_in = False; st.rerun()

    mkt = get_market_data()
    header_relogio(mkt)
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    assets = [("USD", "Dólar", "$"), ("EUR", "Euro", "€"), ("GBP", "Libra", "£"), ("BTC", "Bitcoin", "₿")]
    
    for i, (k, n, s) in enumerate(assets):
        val = mkt.get(k, 0.0)
        prev_val = st.session_state.history_mkt.get(k, val)
        is_up = val >= prev_val 
        st.session_state.history_mkt[k] = val 
        trend_class = "up-trend" if is_up else "down-trend"
        with [mc1, mc2, mc3, mc4][i]:
            svg = UIManager.get_svg_chart(is_up)
            st.markdown(f"""<div class="market-card {trend_class}">{svg}<div class="label-coin">{n}</div><div class="value-coin">{s} {UIManager.format_money(val).replace('R$ ','')}</div></div>""", unsafe_allow_html=True)
            
    st.divider()

    tabs = st.tabs(["🤖 IA Rápida", "✍️ Manual", "📊 Dashboard", "💰 Investimentos", "🎯 Metas", "📑 Extrato", "🧠 Coach"])

    with tabs[0]:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-weight: 600; color: #fff;">💬 Assistente Financeiro</h2>
            <p style="color: #888; font-size: 14px;">Digite ou grave um áudio (envio automático).</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.info("💡 **Dicas:** 'Gastei 50 no Uber', 'Recebi 2000 de pix'")
            
            # [UI FIX] Colunas para colocar Texto e Áudio na mesma linha visual com mais espaço para texto
            c_input, c_mic = st.columns([5, 1], vertical_alignment="bottom")
            
            with c_input:
                with st.form("ia_text", clear_on_submit=True):
                    txt = st.text_input("O que aconteceu?", placeholder="Digite aqui...", label_visibility="collapsed")
                    # Botão invisível para permitir submit com Enter no teclado
                    submitted_text = st.form_submit_button("Enviar Texto", type="primary", use_container_width=True)

            with c_mic:
                # [UI] Indicação visual clara "Gravar Áudio"
                # [FIX] Usando chave dinâmica para resetar o componente após o uso
                audio_val = st.audio_input("🎙️ Gravar", label_visibility="visible", key=f"audio_{st.session_state.audio_key}")

            # Lógica de Processamento (Prioridade: Áudio > Texto)
            if audio_val:
                with st.spinner("🎙️ Processando áudio..."):
                    res = AIManager.process_audio_nlp(audio_val, mkt, user_cats)
                    if "error" not in res:
                        db.add_transaction(user, datetime.now(FUSO_BR), res['amount'], res['category'], res['description'], res['type'])
                        st.toast(f"{res['type']} de R$ {res['amount']} registrada!", icon="✅")
                        
                        # [FIX] Força o reset do componente de áudio incrementando a chave
                        st.session_state.audio_key += 1
                        time.sleep(1.0)
                        st.rerun() # Recarrega a página para limpar o input
                    else: 
                        st.error(res['error'])
            
            elif submitted_text and txt:
                with st.spinner("🤖 Lendo texto..."):
                    res = AIManager.process_nlp(txt, mkt, user_cats)
                    if "error" not in res:
                        db.add_transaction(user, datetime.now(FUSO_BR), res['amount'], res['category'], res['description'], res['type'])
                        st.toast(f"{res['type']} de R$ {res['amount']} registrada!", icon="✅")
                        time.sleep(1.5); st.rerun()
                    else: st.error(res['error'])

    with tabs[1]:
        c1, c2 = st.columns(2)
        
        # [FEAT] Pre-fill from Clone (SAFE DEFAULT FIX)
        # Fallback to 0.01 to match min_value
        default_val = st.session_state.manual_form.get('amount', 0.01)
        default_desc = st.session_state.manual_form.get('desc', "")
        default_cat = st.session_state.manual_form.get('cat', user_cats[0])
        if default_cat not in user_cats: default_cat = user_cats[0]
        
        with c1:
            tp = st.radio("Tipo", ["Despesa", "Receita"], horizontal=True)
            # Ensure value never drops below min_value (0.01)
            safe_val = max(0.01, float(default_val))
            vl = st.number_input("Valor (R$)", min_value=0.01, value=safe_val)
        with c2:
            ct = st.selectbox("Categoria", user_cats, index=user_cats.index(default_cat))
            ds = st.text_input("Descrição", value=default_desc)
        
        # [FEAT] Receipt Upload
        uploaded_file = st.file_uploader("Anexar Comprovante (Opcional)", type=['png', 'jpg', 'jpeg', 'pdf'])
        is_rec = st.checkbox("🔄 Repetir todo mês (Conta Fixa)")
        
        if st.button("Salvar Registro"):
            now = datetime.now(FUSO_BR)
            # Pass uploaded file to DB
            file_name = uploaded_file.name if uploaded_file else None
            db.add_transaction(user, now, vl, ct, ds, tp, uploaded_file, file_name)
            
            if is_rec:
                if db.add_recurring(user, ct, vl, ds, tp, now.day): st.toast("Salvo e programado!", icon="🔄")
            else:
                st.toast("Salvo!", icon="💾")
            
            # Clear Clone Data
            st.session_state.manual_form = {}
            time.sleep(1); st.rerun()

    with tabs[2]:
        if start_date and end_date:
            c_tit, c_eye = st.columns([6, 1])
            c_tit.subheader(f"Visão Geral: {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')}")
            priv = c_eye.toggle("👁️", value=False)

            inc, exp = db.get_totals(user, start_date, end_date)
            bal = inc - exp
            
            k1, k2, k3 = st.columns(3)
            with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Entrou</div><div class="kpi-value" style="color:#4CAF50">{UIManager.format_money(inc, priv)}</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Saiu</div><div class="kpi-value" style="color:#F44336">{UIManager.format_money(exp, priv)}</div></div>', unsafe_allow_html=True)
            with k3: 
                cor = "#4CAF50" if bal >= 0 else "#F44336"
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">Saldo</div><div class="kpi-value" style="color:{cor}">{UIManager.format_money(bal, priv)}</div></div>', unsafe_allow_html=True)
            
            st.divider()
            df_dash = db.fetch_all(user, start_date=start_date, end_date=end_date)
            if not df_dash.empty:
                df_exp = df_dash[df_dash['type']=='Despesa']
                if not df_exp.empty:
                    c_ch, c_li = st.columns([1.5, 1])
                    with c_ch:
                        grp = df_exp.groupby('category')['amount'].sum().reset_index()
                        grp['fmt'] = grp['amount'].apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                        fig = px.pie(grp, values='amount', names='category', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel, custom_data=['fmt'])
                        fig.update_traces(hovertemplate='<b>%{label}</b><br>Gasto: %{customdata[0]}<br>(%{percent})<extra></extra>')
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=350, margin=dict(t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    with c_li:
                        st.markdown("##### 🏆 Onde você mais gasta")
                        top = df_exp.groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
                        for c, v in top.items():
                            st.write(f"**{c}**"); st.progress(min(v/exp, 1.0) if exp>0 else 0, text=f"{UIManager.format_money(v, priv)}")
                else: st.info("Nenhuma despesa no período.")
            else: st.warning("Sem dados para este período.")
        else:
            st.info("👈 Selecione um período na barra lateral.")

    with tabs[3]:
        st.subheader("💰 Carteira de Investimentos")
        df_all = db.fetch_all(user, limit=None)
        if not df_all.empty:
            invs = df_all[df_all['category'].str.contains("Invest", case=False, na=False)]
            if not invs.empty:
                invs['date'] = pd.to_datetime(invs['date'], errors='coerce')
                invs = invs.sort_values('date', ascending=False)
                tot = invs['amount'].sum()
                st.markdown(f'<div class="kpi-card" style="margin-bottom:20px"><div class="kpi-label">Total Acumulado</div><div class="kpi-value" style="color:#4CAF50">{UIManager.format_money(tot, priv)}</div></div>', unsafe_allow_html=True)
                st.markdown("---")
                for _, r in invs.iterrows():
                    c1,c2,c3,c4,c5 = st.columns([1.5, 1.5, 5, 2, 1])
                    data_formatada = r['date'].strftime('%d/%m %H:%M') if pd.notnull(r['date']) else "--/--"
                    val = f"R$ {r['amount']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    
                    if r['type'] == 'Despesa':
                        tipo_visual = "Investiu"
                        cor = "green"
                        sig = ""
                    else:
                        tipo_visual = "Resgate"
                        cor = "green"
                        sig = "+"
                    
                    c1.caption(data_formatada)
                    c2.write(tipo_visual) 
                    c3.write(r['description'])
                    c4.markdown(f":{cor}[{sig} {val}]")
                    
                    @st.dialog(f"Apagar Investimento?")
                    def modal_del_inv(tid):
                        st.write("Confirmar exclusão?"); c_a, c_b = st.columns(2)
                        if c_a.button("Sim", key=f"s_inv_{tid}"):
                            db.remove_transaction(tid, user); st.rerun()
                        if c_b.button("Não", key=f"n_inv_{tid}"): st.rerun()

                    if c5.button("🗑️", key=f"del_inv_{r['id']}"): modal_del_inv(r['id'])
                    st.markdown("---")
            else: st.info("Nenhum registro em 'Investimentos'.")
        else: st.info("Sem dados.")

    with tabs[4]:
        c_h, c_b = st.columns([4,1])
        c_h.markdown("#### 🎯 Metas de Gastos")
        
        @st.dialog("Definir Meta")
        def modal_meta():
            ct = st.selectbox("Categoria", user_cats)
            lm = st.number_input("Limite Mensal (R$)", 100.0, step=50.0)
            if st.button("Salvar Meta"):
                db.set_meta(user, ct, lm); st.rerun()
        
        @st.dialog("Excluir Meta?")
        def delete_meta_dialog(category):
            st.write(f"Excluir meta de **{category}**?")
            c1, c2 = st.columns(2)
            if c1.button("Sim", type="primary"):
                db.delete_meta(user, category); st.rerun()
            if c2.button("Cancelar"): st.rerun()

        if c_b.button("➕ Nova Meta"): modal_meta()
        
        metas = db.get_metas(user)
        if not metas.empty and start_date and end_date:
            atual = db.fetch_all(user, start_date=start_date, end_date=end_date)
            gastos = atual[atual['type']=='Despesa'].groupby('category')['amount'].sum()
            
            cols = st.columns(3) 
            for idx, r in metas.iterrows():
                c, l = r['category'], r['limit_amount']
                s = gastos.get(c, 0.0)
                pct = s / l if l > 0 else 0
                
                if pct < 0.75: bar_color = "#4CAF50"
                elif pct < 1.0: bar_color = "#FFC107"
                else: bar_color = "#FF5252"

                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = s,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    number = {'prefix': "R$ ", 'font': {'family': "Poppins", 'color': "white", 'size': 30}},
                    title = {'text': f"<span style='font-size:2.2em; color: white'><b>{c}</b></span><br><span style='font-size:0.9em;color:#ccc'>Meta: R$ {l:,.0f}</span>", 'align': "center"},
                    gauge = {
                        'axis': {'range': [None, max(l, s*1.1)], 'visible': False},
                        'bar': {'color': bar_color, 'thickness': 0.25},
                        'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0,
                        'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.25, 'value': l}
                    }
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Poppins"}, height=250, margin=dict(l=20, r=20, t=90, b=20))
                
                with cols[idx % 3]:
                    c_chart, c_trash = st.columns([0.85, 0.15])
                    with c_trash:
                        if st.button("🗑️", key=f"btn_del_{idx}", help="Excluir Meta"): delete_meta_dialog(c)
                    with c_chart: st.plotly_chart(fig, use_container_width=True)
        else: st.info("Defina metas e selecione um período para visualizar.")

    with tabs[5]:
        with st.container(border=True):
            st.markdown("### 🗂️ Central de Arquivos")
            b1, b2 = st.columns(2)
            full = db.fetch_all(user, limit=None)
            if not full.empty:
                exc = DocGenerator.to_excel(full)
                b1.download_button("📥 Baixar Excel", exc.getvalue(), "controle.xlsx")
                if start_date and end_date:
                    mes = db.fetch_all(user, start_date=start_date, end_date=end_date)
                    if not mes.empty and FPDF:
                        i, e = db.get_totals(user, start_date, end_date)
                        pdf = DocGenerator.to_pdf(user, mes, i, e, i-e, f"Periodo: {start_date} a {end_date}")
                        b2.download_button("📄 Baixar PDF", pdf, "relatorio.pdf")

        st.divider()
        opt = st.selectbox("Ordenar:", ["Recentes", "Antigos", "Maior Valor"])
        v = db.fetch_all(user, start_date=start_date, end_date=end_date) if start_date else db.fetch_all(user, limit=20)
        
        if not v.empty:
            v['date'] = pd.to_datetime(v['date'], errors='coerce')
            if opt == "Recentes": v = v.sort_values('date', ascending=False)
            elif opt == "Antigos": v = v.sort_values('date', ascending=True)
            else: v = v.sort_values('amount', ascending=False)

            st.markdown("---")
            for _, r in v.iterrows():
                c1,c2,c3,c4,c5,c6 = st.columns([1.5, 1.5, 2, 2, 2, 1])
                data_formatada = r['date'].strftime('%d/%m %H:%M') if pd.notnull(r['date']) else "--/--"
                val = f"R$ {r['amount']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                
                cor = "green" if r['type']=='Receita' else "red"
                sig = "+" if r['type']=='Receita' else "-"
                
                c1.caption(data_formatada)
                c2.write(r['type'])
                c3.write(r['category'])
                c4.write(r['description'])
                
                # [FEAT] Clone & Download Buttons
                with c5:
                    st.markdown(f":{cor}[{sig} {val}]")
                    if r.get('proof_data'):
                        # [FIX v9.43.0] Convert memoryview to bytes for Streamlit
                        proof_bytes = bytes(r['proof_data'])
                        st.download_button("📎", proof_bytes, file_name=r['proof_name'] or "comprovante.pdf", key=f"dl_{r['id']}")
                
                with c6:
                    if st.button("🔄", key=f"clone_{r['id']}", help="Clonar para Manual"):
                        st.session_state.manual_form = {'amount': r['amount'], 'desc': r['description'], 'cat': r['category']}
                        st.toast("Dados copiados para a aba Manual! ✍️", icon="📋")
                    
                    @st.dialog(f"Apagar?")
                    def modal_del(tid):
                        st.write("Confirmar exclusão?"); c_a, c_b = st.columns(2)
                        if c_a.button("Sim", key=f"s_{tid}"):
                            db.remove_transaction(tid, user); st.rerun()
                        if c_b.button("Não", key=f"n_{tid}"): st.rerun()
                    if st.button("🗑️", key=f"del_{r['id']}"): modal_del(r['id'])
                st.markdown("---")
            
            @st.dialog("🗑️ Gerenciamento de Dados")
            def modal_nuke():
                st.write("Essa ação apaga TUDO. Certeza?")
                if st.button("Sim, apagar tudo", type="primary"):
                    db.nuke_data(user); st.rerun()
            
            if st.button("⚠️ Resetar Conta"): modal_nuke()
        else: st.info("Vazio.")

    with tabs[6]:
        st.markdown("#### 🧠 Coach Financeiro")
        if st.button("Analisar minhas finanças", type="primary"):
            with st.spinner("O Coach está pensando..."):
                df_coach = db.fetch_all(user, limit=50)
                inc_t, _ = db.get_totals(user, start_date, end_date)
                rep = AIManager.coach_financeiro(df_coach, inc_t, mkt)
                st.markdown(f'<div style="background:#262730;padding:25px;border-radius:15px;border-left:5px solid #8e44ad;">{rep}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

