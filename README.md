# System Oceny Streszczeń

## Opis Systemu

System oceny streszczeń to komponent aplikacji Domowa Biblioteka, który wykorzystuje uczenie maszynowe do przewidywania jakości streszczeń książek. System zbiera oceny użytkowników, trenuje model regresyjny i wykorzystuje go do poprawy jakości generowanych streszczeń.

## Architektura

System składa się z następujących komponentów:

1. **Moduł oceny streszczeń** (`summary_evaluation.py`)
   - Klasa `SummaryEvaluator` zarządzająca całym procesem
   - Integracja z Digital Ocean Spaces
   - Ekstrakcja cech z tekstu
   - Trenowanie i ewaluacja modelu

2. **Integracja z aplikacją główną** (`app.py`)
   - Interfejs użytkownika do oceny streszczeń
   - Zapisywanie danych do Digital Ocean Spaces
   - Wyświetlanie wyników ewaluacji

## Struktura Danych

### Digital Ocean Spaces

```
bucket_name/
├── summaries/           # Streszczenia w formacie Parquet
│   └── {summary_id}.parquet
├── ratings/            # Oceny w formacie CSV
│   └── {summary_id}.csv
├── models/             # Wytrenowane modele
│   └── summary_evaluator.pkl
└── logs/              # Logi ewaluacji
    └── evaluation_logs.json
```

### Format Danych

1. **Streszczenia** (Parquet):
```python
{
    'summary_id': str,
    'summary_text': str,
    'timestamp': str (ISO format)
}
```

2. **Oceny** (CSV):
```python
{
    'summary_id': str,
    'rating': int (1-5),
    'timestamp': str (ISO format)
}
```

3. **Logi Ewaluacji** (JSON):
```python
{
    'mae': float,
    'mse': float,
    'timestamp': str (ISO format)
}
```

## Proces Trenowania

1. **Zbieranie danych**
   - Użytkownik ocenia streszczenie (1-5)
   - Dane są zapisywane w Digital Ocean Spaces

2. **Ekstrakcja cech**
   - Długość tekstu
   - Liczba słów
   - Liczba zdań
   - Średnia długość słów
   - Średnia długość zdań
   - Embedding semantyczny (paraphrase-multilingual-MiniLM-L12-v2)

3. **Trenowanie modelu**
   - Random Forest Regressor
   - Podział danych: 80% trening, 20% test
   - Metryki: MAE (Mean Absolute Error), MSE (Mean Squared Error)

## Konfiguracja Środowiska

### Wymagane zmienne środowiskowe

```bash
DO_SPACES_ENDPOINT=your_spaces_endpoint
DO_SPACES_KEY=your_spaces_key
DO_SPACES_SECRET=your_spaces_secret
DO_SPACES_BUCKET=your_bucket_name
```

### Zależności

Wszystkie wymagane zależności znajdują się w pliku `requirements.txt`:
- boto3>=1.38.5
- scikit-learn>=1.4.2
- sentence-transformers>=4.1.0
- joblib>=1.2.0

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