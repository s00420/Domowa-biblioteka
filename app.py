import streamlit as st
import openai
import base64
import requests
import json
import os
import pandas as pd
import time
from authlib.integrations.requests_client import OAuth2Session
from urllib.parse import urlencode

# --- Konfiguracja klienta OpenAI ---
client = openai.OpenAI(api_key=os.environ["openai_api_key"])

st.set_page_config(page_title="Domowa Biblioteka", layout="wide")

# --- Logowanie przez Google OAuth ---
GOOGLE_CLIENT_ID = st.secrets["google_client_id"]
GOOGLE_CLIENT_SECRET = st.secrets["google_client_secret"]
REDIRECT_URI = st.secrets["redirect_uri"]

def get_google_auth_url():
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    base_url = "https://accounts.google.com/o/oauth2/v2/auth"
    auth_url = f"{base_url}?{urlencode(params)}"
    return auth_url

def get_google_user_info(code):
    token_url = "https://oauth2.googleapis.com/token"
    userinfo_url = "https://openidconnect.googleapis.com/v1/userinfo"

    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    token_response = requests.post(token_url, data=urlencode(data), headers=headers)

    if token_response.status_code != 200:
        st.error(f"Błąd pobierania tokenu: {token_response.text}")
        st.stop()

    tokens = token_response.json()

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    userinfo_response = requests.get(userinfo_url, headers=headers)
    userinfo_response.raise_for_status()

    return userinfo_response.json()

# --- Dodanie przycisku wylogowania ---
if st.sidebar.button("Wyloguj"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.query_params.clear()
    st.rerun()

if "user_email" not in st.session_state:
    params = st.query_params
    if "code" in params:
        code = params["code"] if isinstance(params["code"], str) else params["code"][0]
        user_info = get_google_user_info(code)
        st.session_state.user_email = user_info.get("email")
        st.rerun()
    else:
        auth_url = get_google_auth_url()
        st.markdown(f"[Kliknij tutaj, aby się zalogować przez Google]({auth_url})")
        st.stop()

user_email = st.session_state.user_email
st.sidebar.success(f"Zalogowano jako: {user_email}")


# --- Wydobywanie tytułów książek z obrazu ---
@st.cache_data(show_spinner=False)
def extract_text_lines_from_image(file_bytes):
    base64_image = base64.b64encode(file_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Na podstawie tekstu widocznego na zdjęciu rozpoznaj tytuły książek oraz ich autorów. "
                            "Wszystkie tytuły są w języku polskim"
                            "Na zdjęciu mogą być okładki, ale większość stanowią grzbiety książek, ułożone poziomo lub pionowo"
                            "Wypisz listę w formacie: Tytuł - Autor. Jeśli autor jest nieczytelny, zostaw puste. "
                            "Każdą pozycję wypisz w nowej linii. Nie dodawaj żadnych komentarzy, tylko listę."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=5000
    )
    lines_raw = response.choices[0].message.content
    return [line.strip() for line in lines_raw.split("\n") if line.strip()]

# --- Pobieranie danych książki z Open Library ---
@st.cache_data
def fetch_book_data(query):
    url = f"https://openlibrary.org/search.json?q={requests.utils.quote(query)}"
    response = requests.get(url)
    if response.status_code == 200:
        docs = response.json().get("docs", [])
        if docs:
            book = docs[0]
            title = book.get("title", "")
            author = ", ".join(book.get("author_name", ["nieznany"]))
            year = book.get("first_publish_year", "brak danych")
            subject = ", ".join(book.get("subject", [])[:1]) if book.get("subject") else "brak danych"
            return {
                "title": title,
                "author": author,
                "year": year,
                "label": f"{title} - {author} ({year})"
            }
    return None

# --- Streszczenie książki przez GPT ---
@st.cache_data(show_spinner=False)
def summarize_book(title):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Napisz krótkie streszczenie książki w maksymalnie 5 zdaniach."},
            {"role": "user", "content": f"Stwórz streszczenie książki '{title}'."}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

# --- Operacje na półce użytkownika ---
def save_user_books(user_email, books):
    os.makedirs("user_shelves", exist_ok=True)
    user_file = f"user_shelves/{user_email.replace('@', '_at_')}.json"
    if os.path.exists(user_file):
        with open(user_file, "r", encoding="utf-8") as f:
            existing_books = json.load(f)
    else:
        existing_books = []
    new_books = [b for b in books if b not in existing_books]
    if new_books:
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(existing_books + new_books, f, ensure_ascii=False, indent=2)

def load_user_books(user_email):
    user_file = f"user_shelves/{user_email.replace('@', '_at_')}.json"
    if os.path.exists(user_file):
        with open(user_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def delete_book_from_shelf(user_email, label_to_delete):
    user_file = f"user_shelves/{user_email.replace('@', '_at_')}.json"
    if os.path.exists(user_file):
        with open(user_file, "r", encoding="utf-8") as f:
            books = json.load(f)
        updated_books = [b for b in books if b.get("label") != label_to_delete]
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(updated_books, f, ensure_ascii=False, indent=2)

# --- UI: półka użytkownika ---
st.title("📚 Domowa Biblioteka")
user_email = st.session_state.user_email

st.header("Moja półka")
user_books = load_user_books(user_email)

available_authors = sorted(set(b["author"] for b in user_books))
available_titles = sorted(set(b["title"] for b in user_books))

st.sidebar.markdown("### Filtry")
author_filter = st.sidebar.selectbox("Filtruj wg autora", [""] + available_authors)
title_filter = st.sidebar.selectbox("Filtruj wg tytułu", [""] + available_titles)

filtered_books = user_books
if author_filter:
    filtered_books = [b for b in filtered_books if author_filter.lower() in b["author"].lower()]
if title_filter:
    filtered_books = [b for b in filtered_books if title_filter.lower() in b["title"].lower()]

if filtered_books:
    df_books = pd.DataFrame(filtered_books)
    selected_label = st.selectbox("Wybierz książkę z półki", ["(brak)"] + df_books["label"].tolist())
    st.dataframe(df_books, use_container_width=True)

    col1, col2 = st.columns(2)

    if selected_label != "(brak)":
        if col1.button("📄 Wygeneruj streszczenie"):
            title_for_summary = selected_label.split("-")[0].strip()
            summary = summarize_book(title_for_summary)

            st.success("Streszczenie:")
            st.markdown(
                f"<div style='padding: 1rem; font-size: 1.1rem; border-radius: 8px;'>{summary}</div>",
                unsafe_allow_html=True
            )

        if col2.button("🗑️ Usuń książkę"):
            delete_book_from_shelf(user_email, selected_label)
            st.success("Książka została usunięta z półki.")
            time.sleep(2)
            st.rerun()
else:
    st.info("Brak książek spełniających kryteria.")
    time.sleep(2)

# --- Rozpoznawanie książek ze zdjęcia ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Rozpoznaj książki ze zdjęcia")

uploaded_file = st.sidebar.file_uploader("Załaduj zdjęcie swojej półki", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Załadowane zdjęcie", use_container_width=True)
    file_bytes = uploaded_file.read()

    if st.sidebar.button("🔍 Rozpoznaj książki"):
        with st.spinner("Wydobywam tekst i szukam książek..."):
            lines = extract_text_lines_from_image(file_bytes)
            enriched = []
            for line in lines:
                if "–" in line:
                    title, author = map(str.strip, line.split("–", 1))
                    query = f"{title} {author}"
                else:
                    title = line.strip()
                    query = title
                result = fetch_book_data(query)
                if result:
                    enriched.append(result)
            if enriched:
                st.session_state["recognized_books"] = enriched
            else:
                st.session_state["recognized_books"] = []

recognized_books = st.session_state.get("recognized_books", [])
if recognized_books:
    selected_labels = st.sidebar.multiselect(
        "Wybierz książki do dodania:",
        [book["label"] for book in recognized_books],
        default=[], key="book_select_labels"
    )
    selected_books = [book for book in recognized_books if book["label"] in selected_labels]
    if st.sidebar.button("✅ Dodaj wybrane książki"):
        existing = load_user_books(user_email)
        existing_labels = {b["label"] for b in existing}
        new_books = [b for b in selected_books if b["label"] not in existing_labels]

        if new_books:
            save_user_books(user_email, new_books)
            st.success(f"Dodano {len(new_books)} nowych książek.")
        else:
            st.info("Wszystkie wybrane książki już znajdują się na Twojej półce.")
        time.sleep(2)
        del st.session_state["recognized_books"]
        st.rerun()

# --- Dodawanie książki ręcznie ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Dodaj książkę ręcznie")

manual_form = st.sidebar.form(key="manual_form")
with manual_form:
    title = st.text_input("Tytuł")
    author = st.text_input("Autor")
    year = st.text_input("Rok wydania")
    submitted = st.form_submit_button("➕ Dodaj książkę")
    if submitted:
        if title and author and year:
            new_book = {
                "title": title,
                "author": author,
                "year": year,
                "label": f"{title} - {author} ({year})"
            }
            save_user_books(user_email, [new_book])
            st.success("Książka została dodana do Twojej półki.")
            time.sleep(2)
            st.rerun()
        else:
            st.warning("Wszystkie pola muszą być uzupełnione.")
            time.sleep(2)
