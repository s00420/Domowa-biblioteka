# System Oceny Streszczeń

## Opis Systemu

System oceny streszczeń to komponent aplikacji Domowa Biblioteka, który wykorzystuje uczenie maszynowe do przewidywania jakości streszczeń książek. System zbiera oceny użytkowników, trenuje model regresyjny i wykorzystuje go do poprawy jakości generowanych streszczeń. System integruje się z Langfuse do monitorowania i analizy procesów.

## Architektura

System składa się z następujących komponentów:

1. **Moduł oceny streszczeń** (`summary_evaluation.py`)
   - Klasa `SummaryEvaluator` zarządzająca całym procesem
   - Integracja z Digital Ocean Spaces
   - Ekstrakcja cech z tekstu przy użyciu LLM (GPT-3.5-turbo)
   - Trenowanie i ewaluacja modelu
   - Integracja z Langfuse do monitorowania

2. **Integracja z aplikacją główną** (`app.py`)
   - Interfejs użytkownika do oceny streszczeń
   - Zapisywanie danych do Digital Ocean Spaces
   - Wyświetlanie wyników ewaluacji
   - System buforowania wyników

## Struktura Danych

### Digital Ocean Spaces

```
bucket_name/
├── summaries/           # Streszczenia w formacie JSON
│   └── {summary_id}.json
├── ratings/            # Oceny w formacie JSON
│   └── {summary_id}.json
├── models/             # Wytrenowane modele
│   └── summary_evaluator.pkl
└── logs/              # Logi ewaluacji
    └── evaluation_logs.json
```

### Format Danych

1. **Streszczenia** (JSON):
```python
{
    'summary_id': str,
    'summary_text': str,
    'book_title': str,
    'book_author': str,
    'timestamp': str (ISO format)
}
```

2. **Oceny** (JSON):
```python
{
    'summary_id': str,
    'rating': float (1.0-5.0),
    'timestamp': str (ISO format)
}
```

3. **Logi Ewaluacji** (JSON):
```python
{
    'mae': float,
    'mse': float,
    'r2': float,
    'timestamp': str (ISO format)
}
```

## Proces Trenowania

1. **Zbieranie danych**
   - Użytkownik ocenia streszczenie (1.0-5.0)
   - Dane są zapisywane w Digital Ocean Spaces
   - System buforuje wyniki dla optymalizacji

2. **Ekstrakcja cech**
   - Zwięzłość (1-5)
   - Spójność (1-5)
   - Kompletność (1-5)
   - Język (1-5)
   - Oryginalność (1-5)
   - Ekstrakcja cech przez LLM (GPT-3.5-turbo)

3. **Trenowanie modelu**
   - Random Forest Regressor
   - Podział danych: 80% trening, 20% test
   - Metryki: MSE (Mean Squared Error), R2 Score
   - System buforowania cech i przewidywań

## Konfiguracja Środowiska

### Wymagane zmienne środowiskowe

```bash
DO_SPACES_ENDPOINT=your_spaces_endpoint
DO_SPACES_KEY=your_spaces_key
DO_SPACES_SECRET=your_spaces_secret
DO_SPACES_BUCKET=your_bucket_name
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=your_langfuse_host
```

### Zależności

Wszystkie wymagane zależności znajdują się w pliku `requirements.txt`:
- boto3>=1.38.5
- openai>=1.0.0
- langfuse>=2.0.0
- scikit-learn>=1.0.0
- numpy>=1.20.0
- pandas>=1.3.0

## Obsługa Błędów

### Typowe błędy i rozwiązania

1. **Problem z połączeniem do Digital Ocean Spaces**
   - Sprawdź poprawność zmiennych środowiskowych
   - Zweryfikuj uprawnienia do bucketa
   - Sprawdź dostępność endpointu

2. **Brak danych do trenowania**
   - System wymaga minimum kilku ocen do rozpoczęcia trenowania
   - Sprawdź poprawność formatu danych w Spaces

3. **Problemy z pamięcią**
   - Model sentence-transformers może wymagać dużo pamięci
   - Rozważ użycie mniejszego modelu
   - Sprawdź limity pamięci w Digital Ocean Apps

## Monitorowanie i Utrzymanie

1. **Metryki wydajności**
   - Monitoruj MAE i MSE w czasie
   - Sprawdź logi ewaluacji w Digital Ocean Spaces

2. **Backup danych**
   - Regularnie tworz kopie zapasowe modeli i logów
   - Rozważ automatyczny backup danych z Spaces

3. **Aktualizacje**
   - Regularnie aktualizuj zależności
   - Monitoruj wydajność nowych wersji modeli

## Rozwój i Rozszerzenia

### Potencjalne ulepszenia

1. **Dodatkowe cechy**
   - Analiza sentymentu
   - Złożoność tekstu
   - Tematyka książki

2. **Inne modele**
   - Gradient Boosting
   - Neural Networks
   - Ensemble methods

3. **Automatyzacja**
   - Harmonogram trenowania
   - Automatyczne alerty
   - A/B testing modeli

## Kontakt i Wsparcie

W przypadku problemów lub pytań, prosimy o kontakt z zespołem developerskim. 