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
from langfuse import Langfuse

def normalize_text(text):
    """
    Normalizuje tekst do porównań - usuwa znaki specjalne, sprowadza do małych liter,
    usuwa polskie znaki diakrytyczne.
    """
    if not text:
        return ""
    
    # Zamień polskie znaki na ich odpowiedniki bez diakrytyków
    polish_chars = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    }
    
    # Usuń znaki specjalne i sprowadź do małych liter
    text = text.lower()
    for char, replacement in polish_chars.items():
        text = text.replace(char, replacement)
    
    # Usuń znaki specjalne i spacje
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text.strip()

def calculate_confidence(title, author, result):
    """
    Oblicza poziom pewności dopasowania wyniku wyszukiwania.
    """
    confidence = 0
    
    # Normalizuj teksty do porównania
    result_title = normalize_text(result.get("title", ""))
    search_title = normalize_text(title)
    
    # Porównanie tytułów
    if search_title == result_title:
        confidence += 0.6  # Dokładne dopasowanie
    elif search_title in result_title or result_title in search_title:
        confidence += 0.4  # Jeden zawiera się w drugim
    else:
        # Sprawdź podobieństwo słów
        search_words = set(search_title.split())
        result_words = set(result_title.split())
        common_words = search_words.intersection(result_words)
        if common_words:
            confidence += 0.2 * (len(common_words) / len(search_words))
    
    # Porównanie autorów
    if author:
        result_authors = [normalize_text(a) for a in result.get("authors", [])]
        search_author = normalize_text(author)
        
        # Sprawdź dokładne dopasowanie autora
        if search_author in result_authors:
            confidence += 0.4
        else:
            # Sprawdź częściowe dopasowanie
            for result_author in result_authors:
                if search_author in result_author or result_author in search_author:
                    confidence += 0.3
                    break
                else:
                    # Sprawdź podobieństwo słów w nazwisku
                    search_author_words = set(search_author.split())
                    result_author_words = set(result_author.split())
                    common_words = search_author_words.intersection(result_author_words)
                    if common_words:
                        confidence += 0.2 * (len(common_words) / len(search_author_words))
    
    return confidence

def search_book(title, author=None):
    """
    Wyszukuje książkę w różnych źródłach.
    Zwraca najlepiej dopasowany wynik.
    """
    results = []
    
    # Przygotuj zapytanie
    if author:
        query = f"{title} {author}"
    else:
        query = title
    
    # Usuń znaki specjalne i popraw formatowanie
    query = query.replace("(", "").replace(")", "").strip()
    
    # Wyszukiwanie w Google Books API
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q={requests.utils.quote(query)}&langRestrict=pl"
        result = requests.get(url)
        if result.status_code == 200:
            items = result.json().get("items", [])
            for item in items[:5]:  # Sprawdź 5 najlepszych wyników
                volume_info = item.get("volumeInfo", {})
                if volume_info:
                    results.append({
                        "source": "google",
                        "title": volume_info.get("title", ""),
                        "author": ", ".join(volume_info.get("authors", ["nieznany"])),
                        "year": volume_info.get("publishedDate", "brak danych"),
                        "confidence": calculate_confidence(title, author, volume_info)
                    })
    except Exception as e:
        st.error(f"Błąd podczas wyszukiwania w Google Books: {str(e)}")
    
    # Wyszukiwanie w Open Library API
    try:
        url = f"https://openlibrary.org/search.json?q={requests.utils.quote(query)}&language=pol"
        result = requests.get(url)
        if result.status_code == 200:
            docs = result.json().get("docs", [])
            for doc in docs[:5]:  # Sprawdź 5 najlepszych wyników
                results.append({
                    "source": "openlibrary",
                    "title": doc.get("title", ""),
                    "author": ", ".join(doc.get("author_name", ["nieznany"])),
                    "year": doc.get("first_publish_year", "brak danych"),
                    "confidence": calculate_confidence(title, author, doc)
                })
    except Exception as e:
        st.error(f"Błąd podczas wyszukiwania w Open Library: {str(e)}")
    
    # Wyszukiwanie w WorldCat API (jeśli dostępny)
    try:
        url = f"http://www.worldcat.org/webservices/catalog/search/worldcat/opensearch?q={requests.utils.quote(query)}&wskey=YOUR_WORLDCAT_KEY"
        result = requests.get(url)
        if result.status_code == 200:
            # Przetwarzanie wyników WorldCat
            # (tutaj dodaj kod do przetwarzania XML z WorldCat)
            pass
    except Exception as e:
        st.error(f"Błąd podczas wyszukiwania w WorldCat: {str(e)}")
    
    # Wybierz najlepszy wynik
    if results:
        best_result = max(results, key=lambda x: x["confidence"])
        if best_result["confidence"] >= 0.3:  # Minimalny próg pewności
            return {
                "title": best_result["title"],
                "author": best_result["author"],
                "year": best_result["year"],
                "label": f"{best_result['title']} - {best_result['author']} ({best_result['year']})",
                "confidence": best_result["confidence"]
            }
    
    # Jeśli nie znaleziono dobrego dopasowania, użyj oryginalnego rozpoznania
    return {
        "title": title,
        "author": author if author else "nieznany",
        "year": "brak danych",
        "label": f"{title} - {author if author else 'nieznany'} (brak danych)",
        "confidence": 0
    }

# Inicjalizacja Langfuse z Streamlit secrets
try:
    lf = Langfuse(
        public_key=os.environ["langfuse_public_key"],
        secret_key=os.environ["langfuse_secret_key"],
        host=os.environ["langfuse_host"]
    )
    st.session_state["langfuse_initialized"] = True
except Exception as e:
    st.error(f"Błąd inicjalizacji Langfuse: {str(e)}")
    st.session_state["langfuse_initialized"] = False

# --- Konfiguracja klienta OpenAI ---
client = openai.OpenAI(api_key=os.environ["openai_api_key"])

st.set_page_config(page_title="Domowa Biblioteka", layout="wide")

# --- Logowanie przez Google OAuth ---
GOOGLE_CLIENT_ID = os.environ["google_client_id"]
GOOGLE_CLIENT_SECRET = os.environ["google_client_secret"]
REDIRECT_URI = os.environ["redirect_uri"]

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
        if st.button("🔒 Zaloguj się przez Google"):
            st.query_params.clear()
            st.markdown(f"<meta http-equiv='refresh' content='0; url={auth_url}'>", unsafe_allow_html=True)
            st.stop()
        st.stop()

user_email = st.session_state.user_email
st.sidebar.success(f"Zalogowano jako: {user_email}")

# --- Główna aplikacja po zalogowaniu ---

st.title("📚 Domowa Biblioteka")

st.header("Moja półka")

# Funkcje pomocnicze
@st.cache_data
def load_user_books(user_email):
    user_file = f"user_shelves/{user_email.replace('@', '_at_')}.json"
    if os.path.exists(user_file):
        with open(user_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_books(user_email, books):
    os.makedirs("user_shelves", exist_ok=True)
    user_file = f"user_shelves/{user_email.replace('@', '_at_')}.json"
    try:
        if os.path.exists(user_file):
            with open(user_file, "r", encoding="utf-8") as f:
                existing_books = json.load(f)
        else:
            existing_books = []
        
        # Sprawdź, które książki są nowe (tylko tytuł i autor)
        existing_books_set = {(b["title"].lower(), b["author"].lower()) for b in existing_books}
        new_books = []
        for b in books:
            book_key = (b["title"].lower(), b["author"].lower())
            if book_key not in existing_books_set:
                new_books.append({
                    "title": b["title"],
                    "author": b["author"]
                })
        
        if new_books:
            # Dodaj nowe książki do istniejących
            updated_books = existing_books + new_books
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(updated_books, f, ensure_ascii=False, indent=2)
            st.success(f"Dodano {len(new_books)} nowych książek.")
            time.sleep(2)
            # Odśwież dane użytkownika
            st.session_state["user_books"] = updated_books
        else:
            st.info("Wszystkie wybrane książki już znajdują się na Twojej półce.")
            time.sleep(2)
    except Exception as e:
        st.error(f"Błąd podczas zapisywania książek: {str(e)}")

def delete_book_from_shelf(user_email, title, author):
    user_file = f"user_shelves/{user_email.replace('@', '_at_')}.json"
    if os.path.exists(user_file):
        with open(user_file, "r", encoding="utf-8") as f:
            books = json.load(f)
        updated_books = [b for b in books if not (b["title"].lower() == title.lower() and b["author"].lower() == author.lower())]
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(updated_books, f, ensure_ascii=False, indent=2)

# Załaduj książki użytkownika
if "user_books" not in st.session_state:
    st.session_state["user_books"] = load_user_books(user_email)

user_books = st.session_state["user_books"]

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
    # Usuń kolumny year i label
    df_books = df_books[["title", "author"]]
    selected_book = st.selectbox("Wybierz książkę z półki", ["(brak)"] + [f"{b['title']} - {b['author']}" for b in filtered_books])
    st.dataframe(df_books, use_container_width=True)

    col1, col2 = st.columns(2)

    if selected_book != "(brak)":
        if col1.button("📄 Wygeneruj streszczenie"):
            # Rozdziel tytuł i autora
            parts = selected_book.split(" - ")
            title_for_summary = parts[0].strip()
            author_for_summary = parts[1].strip()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "Twoim zadaniem jest przygotowanie krótkiego streszczenia książki w maksymalnie 5 zdaniach."
                        "Szukaj w sieci dostępnych informacji, ale jeśli nie znajdziesz wystarczających informacji, napisz: "
                        "'Nie mam wystarczającej wiedzy o tej książce, aby przygotować streszczenie.'"
                    )},
                    {"role": "user", "content": (
                        f"Stwórz streszczenie książki pod tytułem '{title_for_summary}', napisanej przez '{author_for_summary}'."
                    )}
                ],
                max_tokens=300
            )
            summary = response.choices[0].message.content
            st.success("Streszczenie:")
            st.markdown(
                f"<div style='padding: 1rem; font-size: 1.1rem; border-radius: 8px;'>{summary}</div>",
                unsafe_allow_html=True
            )

        if col2.button("🗑️ Usuń książkę"):
            parts = selected_book.split(" - ")
            title_to_delete = parts[0].strip()
            author_to_delete = parts[1].strip()
            delete_book_from_shelf(user_email, title_to_delete, author_to_delete)
            st.success("Książka została usunięta z półki.")
            time.sleep(2)
            st.rerun()
else:
    st.info("Brak książek spełniających kryteria.")

# --- Upload i rozpoznawanie książek ze zdjęcia ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Rozpoznaj książki ze zdjęcia")

uploaded_file = st.sidebar.file_uploader("Załaduj zdjęcie swojej półki", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Załadowane zdjęcie", use_container_width=True)
    file_bytes = uploaded_file.read()

    if st.sidebar.button("🔍 Rozpoznaj książki"):
        base64_image = base64.b64encode(file_bytes).decode("utf-8")

        # --- START Langfuse śledzenie ---
        if st.session_state.get("langfuse_initialized", False):
            try:
                trace = lf.trace(
                    name="book_recognition",
                    user_id=user_email,
                    metadata={"total_books": len(user_books)},
                    model="gpt-4-turbo"
                )
                generation = trace.generation(
                    name="image_to_books",
                    input={"image_base64_length": len(base64_image) if "base64_image" in locals() else 0},
                    model="gpt-4-turbo"
                )
            except Exception as e:
                st.error(f"Błąd podczas inicjalizacji śledzenia Langfuse: {str(e)}")
        # --- END START ---

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                """Na podstawie załączonym zdjęciu rozpoznaj wszystkie możliwe teksty. 
                                Wśród tych tekstów rozpoznaj tytuły książek oraz ich autorów.
                                Załóż że większość tekstów jest w języku polskim, ale mogą się trafić również w innym języku. 
                                Książek może być więcej niż jedna, a grzbiety mogą być w układzie zarówno pionowym jak i poziomym"""
                                "Wypisz listę w formacie: Tytuł - Autor. Jeśli autor jest nieczytelny, zostaw puste. Nie dodawaj żadnych innych komentarzy"
                                "Każdą pozycję wypisz w nowej linii."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=4000
        )
        lines_raw = response.choices[0].message.content
        lines = [line.strip() for line in lines_raw.split("\n") if line.strip()]
        enriched = []
        for line in lines:
            if "-" in line:
                title, author = map(str.strip, line.split("-", 1))
            else:
                title = line.strip()
                author = None
            
            # Wyszukaj książkę
            book_info = search_book(title, author)
            if book_info:
                enriched.append(book_info)
            else:
                # Jeśli nie znaleziono dopasowania, użyj oryginalnego rozpoznania
                enriched.append({
                    "title": title,
                    "author": author if author else "nieznany",
                    "year": "brak danych",
                    "label": f"{title} - {author if author else 'nieznany'} (brak danych)"
                })

        if enriched:
            st.session_state["recognized_books"] = enriched
            st.success(f"Rozpoznano {len(enriched)} książek!")
        else:
            st.session_state["recognized_books"] = []
            st.warning("Nie rozpoznano żadnych książek na zdjęciu.")

        # --- DOMKNIJ Langfuse generation ---
        if st.session_state.get("langfuse_initialized", False):
            try:
                # Pobierz informacje o tokenach z odpowiedzi OpenAI
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                generation.end(
                    output=response.choices[0].message.content,
                    usage=token_usage,
                    metadata={
                        "recognized_books_count": len(enriched),
                        "image_size": len(base64_image),
                        "model": response.model,
                        "token_usage": token_usage
                    }
                )
                trace.update(
                    status="COMPLETED",
                    metadata={
                        "total_books_recognized": len(enriched),
                        "token_usage": token_usage
                    }
                )
            except Exception as e:
                st.error(f"Błąd podczas zamykania śledzenia Langfuse: {str(e)}")

recognized_books = st.session_state.get("recognized_books", [])
if recognized_books:
    st.sidebar.markdown("### Rozpoznane książki")
    selected_books = st.sidebar.multiselect(
        "Wybierz książki do dodania:",
        [f"{book['title']} - {book['author']}" for book in recognized_books],
        default=[], 
        key="book_select"
    )
    if st.sidebar.button("✅ Dodaj wybrane książki"):
        existing = load_user_books(user_email)
        existing_books_set = {(b["title"].lower(), b["author"].lower()) for b in existing}
        new_books = []
        
        for book_str in selected_books:
            parts = book_str.split(" - ")
            title = parts[0].strip()
            author = parts[1].strip()
            if (title.lower(), author.lower()) not in existing_books_set:
                new_books.append({
                    "title": title,
                    "author": author
                })
        
        if new_books:
            save_user_books(user_email, new_books)
            # Wyczyść stan rozpoznanych książek przed odświeżeniem
            if "recognized_books" in st.session_state:
                del st.session_state["recognized_books"]
            st.rerun()  # Odśwież stronę po dodaniu książek
        else:
            st.info("Wszystkie wybrane książki już znajdują się na Twojej półce.")
            time.sleep(2)
            # Wyczyść stan rozpoznanych książek
            if "recognized_books" in st.session_state:
                del st.session_state["recognized_books"]
            st.rerun()

# --- Dodawanie książki ręcznie ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Dodaj książkę ręcznie")

manual_form = st.sidebar.form(key="manual_form")
with manual_form:
    title = st.text_input("Tytuł")
    author = st.text_input("Autor")
    submitted = st.form_submit_button("➕ Dodaj książkę")
    if submitted:
        if title and author:
            # Sprawdź, czy książka już istnieje
            existing = load_user_books(user_email)
            if any(b["title"].lower() == title.lower() and b["author"].lower() == author.lower() for b in existing):
                st.warning("Ta książka już znajduje się na Twojej półce.")
                time.sleep(2)
            else:
                new_book = {
                    "title": title,
                    "author": author
                }
                save_user_books(user_email, [new_book])
                st.success("Książka została dodana do Twojej półki.")
                time.sleep(2)
                st.rerun()
        else:
            st.warning("Tytuł i autor muszą być uzupełnione.")
            time.sleep(2)
            