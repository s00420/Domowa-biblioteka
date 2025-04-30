import os
import boto3
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import hashlib
from concurrent.futures import ThreadPoolExecutor
from langfuse import Langfuse
import openai
import tempfile
from datetime import datetime

class SummaryEvaluator:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.cache = {}
        self._model = None
        self._langfuse = None
        self._s3 = None
        self._bucket_name = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return
            
        try:
            # Inicjalizacja Langfuse tylko jeśli nie jest jeszcze zainicjalizowany
            if not self._langfuse:
                self._langfuse = Langfuse(
                    public_key=os.environ["langfuse_public_key"],
                    secret_key=os.environ["langfuse_secret_key"],
                    host=os.environ["langfuse_host"]
                )
            
            # Inicjalizacja S3 tylko jeśli nie jest jeszcze zainicjalizowany
            if not self._s3:
                self._s3 = boto3.client(
                    's3',
                    endpoint_url=os.environ["DO_SPACES_ENDPOINT"],
                    aws_access_key_id=os.environ["DO_SPACES_KEY"],
                    aws_secret_access_key=os.environ["DO_SPACES_SECRET"]
                )
                self._bucket_name = os.environ["DO_SPACES_BUCKET"]
            
            # Wczytaj i wytrenuj model tylko jeśli nie jest jeszcze wczytany
            if not self._model:
                self._load_and_train_model()
            
            self._initialized = True
        except Exception as e:
            raise

    def _load_and_train_model(self):
        """Wczytuje dane treningowe i trenuje model regresji."""
        try:
            print("Rozpoczynam wczytywanie danych treningowych...")
            # Wczytaj streszczenia
            summaries = self._load_summaries()
            print(f"Wczytano {len(summaries)} streszczeń")
            
            # Wczytaj oceny
            ratings = self._load_ratings()
            print(f"Wczytano {len(ratings)} ocen")
            
            if not summaries or not ratings:
                print("Brak danych treningowych - nie można wytrenować modelu")
                self._model = None
                return
            
            # Połącz dane
            data = []
            for summary_id, summary_text in summaries.items():
                if summary_id in ratings:
                    data.append({
                        'summary_id': summary_id,
                        'text': summary_text,
                        'rating': ratings[summary_id]
                    })
            
            print(f"Połączono {len(data)} par streszczenie-ocena")
            
            if not data:
                print("Brak sparowanych danych treningowych")
                self._model = None
                return
            
            # Ekstrakcja cech
            X = []
            y = []
            
            for item in data:
                try:
                    features = self._extract_features(item['text'])
                    if features:
                        X.append(features)
                        y.append(item['rating'])
                except Exception as e:
                    print(f"Błąd podczas ekstrakcji cech: {str(e)}")
            
            print(f"Wyekstrahowano cechy dla {len(X)} streszczeń")
            
            if not X or not y:
                print("Brak danych do trenowania modelu")
                self._model = None
                return
            
            # Konwersja do numpy array
            X = np.array(X)
            y = np.array(y)
            
            # Podział na zbiór treningowy i testowy
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"Rozmiar zbioru treningowego: {len(X_train)}")
            print(f"Rozmiar zbioru testowego: {len(X_test)}")
            
            # Trenowanie modelu
            self._model = RandomForestRegressor(n_estimators=100, random_state=42)
            self._model.fit(X_train, y_train)
            
            # Ocena modelu
            y_pred = self._model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model wytrenowany. MSE: {mse:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            print(f"Błąd podczas trenowania modelu: {str(e)}")
            self._model = None

    def predict_rating(self, summary_text):
        try:
            # Inicjalizuj tylko jeśli nie jest jeszcze zainicjalizowany
            self._initialize()
            
            # Sprawdź cache
            summary_hash = hashlib.md5(summary_text.encode()).hexdigest()
            cache_key = f"predicted_rating_{summary_hash}"
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Ekstrakcja cech
            features = self._extract_features(summary_text)
            
            if self._model is None:
                return None
            
            # Przewidywanie oceny
            predicted_rating = self._model.predict([features])[0]
            predicted_rating = max(1, min(5, predicted_rating))
            
            # Rozpocznij śledzenie w Langfuse tylko przy przewidywaniu oceny
            if self._langfuse:
                try:
                    trace = self._langfuse.trace(
                        name="summary_rating_prediction",
                        metadata={
                            "summary_length": len(summary_text),
                            "features": features,
                            "predicted_rating": predicted_rating
                        }
                    )
                    
                    generation = trace.generation(
                        name="rating_prediction",
                        input={"summary": summary_text},
                        output=str(predicted_rating),
                        model="random_forest"
                    )
                    
                    trace.update(status="COMPLETED")
                except Exception as e:
                    pass
            
            # Zapisz w cache
            self.cache[cache_key] = predicted_rating
            
            return predicted_rating
        except Exception as e:
            return None

    def _extract_features(self, summary_text):
        try:
            # Sprawdź cache
            summary_hash = hashlib.md5(summary_text.encode()).hexdigest()
            cache_key = f"llm_features_{summary_hash}"
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Ekstrakcja cech z użyciem LLM
            client = openai.OpenAI(api_key=os.environ["openai_api_key"])
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": (
                        "Twoim zadaniem jest analiza streszczenia książki i wyciągnięcie cech numerycznych. "
                        "Oceń następujące aspekty w skali 1-5: "
                        "1. Zwięzłość (1 - zbyt długie, 5 - idealna długość) "
                        "2. Spójność (1 - niespójne, 5 - bardzo spójne) "
                        "3. Kompletność (1 - brak kluczowych elementów, 5 - kompletne) "
                        "4. Język (1 - słaby, 5 - doskonały) "
                        "5. Oryginalność (1 - banalne, 5 - oryginalne) "
                        "Zwróć tylko liczby oddzielone przecinkami, bez dodatkowych komentarzy."
                    )},
                    {"role": "user", "content": summary_text}
                ],
                max_tokens=100
            )
            
            features = [float(x) for x in response.choices[0].message.content.split(",")]
            
            # Zapisz w cache
            self.cache[cache_key] = features
            
            return features
        except Exception as e:
            return [3.0] * 5  # Zwróć wartości domyślne w przypadku błędu

    def _load_summaries(self):
        try:
            response = self._s3.list_objects_v2(Bucket=self._bucket_name, Prefix="summaries/")
            summaries = {}
            
            if 'Contents' not in response:
                return summaries
            
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    data = self._s3.get_object(Bucket=self._bucket_name, Key=obj['Key'])
                    content = data['Body'].read().decode('utf-8')
                    summary_data = json.loads(content)
                    summaries[summary_data['summary_id']] = summary_data['summary_text']
            
            return summaries
        except Exception as e:
            return None

    def _load_ratings(self):
        try:
            response = self._s3.list_objects_v2(Bucket=self._bucket_name, Prefix="ratings/")
            ratings = {}
            
            if 'Contents' not in response:
                return ratings
            
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    data = self._s3.get_object(Bucket=self._bucket_name, Key=obj['Key'])
                    content = data['Body'].read().decode('utf-8')
                    rating_data = json.loads(content)
                    ratings[rating_data['summary_id']] = rating_data['rating']
            
            return ratings
        except Exception as e:
            return None

    def run_training_pipeline(self):
        try:
            # Inicjalizuj tylko jeśli nie jest jeszcze zainicjalizowany
            self._initialize()
            
            # Wczytaj dane
            summaries = self._load_summaries()
            ratings = self._load_ratings()
            
            if not summaries or not ratings:
                print("Brak danych do trenowania modelu")
                return None
            
            # Połącz dane
            data = pd.merge(summaries, ratings, on='summary_id')
            
            # Ekstrakcja cech
            X = []
            for summary in data['summary_text']:
                features = self._extract_features(summary)
                X.append(features)
            
            X = np.array(X)
            y = data['rating'].values
            
            # Podział na zbiór treningowy i testowy
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Trenowanie modelu
            self._model = RandomForestRegressor(n_estimators=100, random_state=42)
            self._model.fit(X_train, y_train)
            
            # Ewaluacja
            y_pred = self._model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Wyczyść cache po trenowaniu
            self.cache.clear()
            
            return {"mae": mae, "mse": mse}
        except Exception as e:
            print(f"Błąd podczas trenowania modelu: {str(e)}")
            return None

    def train_model(self, X_train, y_train):
        """Trenuje model regresji"""
        self._model.fit(X_train, y_train)
        
    def evaluate_model(self, X_test, y_test):
        """Ewaluacja modelu"""
        y_pred = self._model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'mae': mae,
            'mse': mse,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self):
        """Zapisuje model lokalnie i w Digital Ocean Spaces"""
        try:
            print("Rozpoczynam zapis modelu...")
            
            # Zapis lokalny
            local_path = 'models/summary_evaluator.pkl'
            os.makedirs('models', exist_ok=True)
            joblib.dump(self._model, local_path)
            print(f"Zapisano model lokalnie: {local_path}")
            
            # Zapis w Digital Ocean Spaces
            print("Zapisywanie modelu w Digital Ocean Spaces...")
            with open(local_path, 'rb') as f:
                self._s3.upload_fileobj(f, self._bucket_name, 'models/summary_evaluator.pkl')
            print("Model zapisany w Digital Ocean Spaces")
            
        except Exception as e:
            print(f"Błąd podczas zapisywania modelu: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def save_evaluation_logs(self, metrics):
        """Zapisuje logi ewaluacji"""
        try:
            print("Rozpoczynam zapis logów...")
            logs_path = 'evaluation_logs.json'
            
            # Wczytanie istniejących logów
            try:
                print("Próba wczytania istniejących logów...")
                logs_obj = self._s3.get_object(Bucket=self._bucket_name, Key='logs/evaluation_logs.json')
                logs = json.loads(logs_obj['Body'].read().decode('utf-8'))
                print(f"Wczytano {len(logs)} istniejących logów")
            except:
                print("Brak istniejących logów, tworzenie nowych")
                logs = []
                
            # Dodanie nowych logów
            print("Dodawanie nowych logów...")
            logs.append(metrics)
            
            # Zapis lokalny
            print("Zapisywanie logów lokalnie...")
            with open(logs_path, 'w') as f:
                json.dump(logs, f)
            print(f"Zapisano logi lokalnie: {logs_path}")
            
            # Zapis w Digital Ocean Spaces
            print("Zapisywanie logów w Digital Ocean Spaces...")
            with open(logs_path, 'rb') as f:
                self._s3.upload_fileobj(f, self._bucket_name, 'logs/evaluation_logs.json')
            print("Logi zapisane w Digital Ocean Spaces")
            
        except Exception as e:
            print(f"Błąd podczas zapisywania logów: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def run_training_pipeline(self):
        """Uruchamia cały proces trenowania"""
        try:
            print("\n=== Rozpoczynam pipeline treningowy ===")
            
            # Wczytanie danych
            print("\n1. Wczytywanie danych...")
            summaries = self._load_summaries()
            ratings = self._load_ratings()
            
            if summaries.empty or ratings.empty:
                print("❌ Brak wystarczającej ilości danych do trenowania")
                return None
            
            print(f"✅ Wczytano {len(summaries)} streszczeń i {len(ratings)} ocen")
            
            # Przygotowanie danych
            print("\n2. Przygotowywanie danych...")
            X_train, X_test, y_train, y_test = self.prepare_data(summaries, ratings)
            
            if X_train is None:
                print("❌ Brak danych do trenowania po przygotowaniu")
                return None
            
            print(f"✅ Przygotowano dane: {X_train.shape[0]} próbek treningowych, {X_test.shape[0]} testowych")
            
            # Trenowanie modelu
            print("\n3. Trenowanie modelu...")
            self.train_model(X_train, y_train)
            print("✅ Model wytrenowany")
            
            # Ewaluacja
            print("\n4. Ewaluacja modelu...")
            metrics = self.evaluate_model(X_test, y_test)
            
            if metrics is None:
                print("❌ Błąd podczas ewaluacji modelu")
                return None
            
            print(f"✅ Wyniki ewaluacji: MAE={metrics['mae']:.2f}, MSE={metrics['mse']:.2f}")
            
            # Zapis modelu i logów
            print("\n5. Zapisywanie modelu i logów...")
            self.save_model()
            self.save_evaluation_logs(metrics)
            
            print("\n=== Pipeline treningowy zakończony pomyślnie ===")
            return metrics
            
        except Exception as e:
            print(f"\n❌ Błąd podczas wykonywania pipeline'u treningowego: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None 

    def prepare_data(self, summaries_df, ratings_df):
        """Przygotowanie danych do trenowania"""
        try:
            print("Rozpoczynam przygotowanie danych...")
            print(f"Liczba streszczeń: {len(summaries_df)}, liczba ocen: {len(ratings_df)}")
            
            if summaries_df.empty or ratings_df.empty:
                print("Brak danych do przygotowania")
                return None, None, None, None
            
            # Sprawdź kolumny w DataFrame'ach
            print("Kolumny w summaries_df:", summaries_df.columns.tolist())
            print("Kolumny w ratings_df:", ratings_df.columns.tolist())
            
            # Połączenie danych
            print("Łączenie danych...")
            merged_df = pd.merge(summaries_df, ratings_df, on='summary_id')
            print(f"Połączono dane. Liczba próbek: {len(merged_df)}")
            
            if merged_df.empty:
                print("Brak danych po połączeniu")
                return None, None, None, None
            
            # Ekstrakcja cech
            print("Ekstrakcja cech...")
            X = pd.DataFrame([self._extract_features(text) for text in merged_df['summary_text']])
            print(f"Wyekstrahowano cechy. Liczba cech: {X.shape[1]}")
            
            y = merged_df['rating']
            print(f"Przygotowano zmienną celu. Liczba próbek: {len(y)}")
            
            # Podział na zbiór treningowy i testowy
            print("Podział na zbiór treningowy i testowy...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Podzielono dane: {X_train.shape[0]} treningowych, {X_test.shape[0]} testowych")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Błąd podczas przygotowania danych: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, None, None, None
    
    def _get_embedding(self, text):
        """Pobiera embedding dla tekstu"""
        try:
            return self._model.encode([text])[0]
        except Exception as e:
            print(f"Błąd podczas pobierania embeddingu: {str(e)}")
            return None

    def extract_features(self, text):
        """Ekstrakcja cech z tekstu - połączenie cech z LLM i embeddingów"""
        try:
            # Równoległe przetwarzanie cech
            future_llm = self.thread_pool.submit(self._extract_features_with_llm, text)
            future_embedding = self.thread_pool.submit(self._get_embedding, text) if self._model else None

            # Pobierz cechy z LLM
            llm_features = future_llm.result()
            if llm_features is None:
                print("Nie udało się wyekstrahować cech za pomocą LLM")
                return None

            # Pobierz embeddingi jeśli dostępne
            if future_embedding:
                embedding = future_embedding.result()
                if embedding is not None:
                    for i, val in enumerate(embedding):
                        llm_features[f'embedding_{i}'] = float(val)

            return llm_features

        except Exception as e:
            print(f"Błąd podczas ekstrakcji cech: {str(e)}")
            return None

    def _extract_features_with_llm(self, text):
        """Ekstrakcja cech z tekstu za pomocą LLM z buforowaniem"""
        try:
            # Sprawdź cache
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = f"llm_features_{text_hash}"
            
            # Sprawdź czy mamy już cechy w cache
            if cache_key in self.cache:
                print(f"Używam cech z cache dla {cache_key}")
                return self.cache[cache_key]
            
            # Rozpoczęcie śledzenia w Langfuse tylko jeśli jest zainicjalizowany i nie mamy w cache
            trace = None
            generation = None
            if self._langfuse and cache_key not in self.cache:
                try:
                    trace = self._langfuse.trace(
                        name="llm_feature_extraction",
                        metadata={"text_length": len(text), "cache_key": cache_key}
                    )
                except Exception as e:
                    print(f"Błąd podczas inicjalizacji śledzenia Langfuse: {str(e)}")
            
            # Przygotowanie promptu
            prompt = f"""Przeanalizuj poniższe streszczenie książki i wyekstrahuj następujące cechy w formacie JSON:
            1. clarity - jasność i zrozumiałość tekstu (1-5)
            2. coherence - spójność i logiczna struktura (1-5)
            3. completeness - kompletność informacji (1-5)
            4. conciseness - zwięzłość (1-5)
            5. engagement - angażującośc (1-5)
            6. language_quality - jakość języka (1-5)
            7. main_points - liczba głównych punktów
            8. technical_terms - liczba terminów technicznych
            9. sentiment - ogólny sentyment (1-5)

            Streszczenie:
            {text}

            Zwróć tylko JSON bez dodatkowych komentarzy."""

            # Wywołanie LLM z timeoutem
            if trace:
                try:
                    generation = trace.generation(
                        name="feature_extraction",
                        model="gpt-3.5-turbo",
                        input={"prompt": prompt}
                    )
                except Exception as e:
                    print(f"Błąd podczas inicjalizacji generacji Langfuse: {str(e)}")

            try:
                client = openai.OpenAI(api_key=os.environ["openai_api_key"])
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Jesteś ekspertem w analizie tekstu. Twoim zadaniem jest wyekstrahowanie cech z tekstu w formacie JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
            except openai.RateLimitError as e:
                print(f"Błąd limitu API OpenAI: {str(e)}")
                print("Używam podstawowych cech z powodu przekroczenia limitu API")
                return self._get_basic_features(text)
            except Exception as e:
                print(f"Błąd podczas wywołania LLM: {str(e)}")
                return self._get_basic_features(text)

            # Pobranie odpowiedzi
            result = response.choices[0].message.content
            features = json.loads(result)

            # Zakończenie śledzenia w Langfuse tylko jeśli było zainicjalizowane
            if generation:
                try:
                    generation.end(
                        output=result,
                        metadata={"features": features}
                    )
                except Exception as e:
                    print(f"Błąd podczas kończenia generacji Langfuse: {str(e)}")
            
            if trace:
                try:
                    trace.update(status="COMPLETED")
                except Exception as e:
                    print(f"Błąd podczas aktualizacji śledzenia Langfuse: {str(e)}")

            # Zapisz w cache
            self.cache[cache_key] = features
            return features

        except Exception as e:
            print(f"Błąd podczas ekstrakcji cech za pomocą LLM: {str(e)}")
            return self._get_basic_features(text)

    def _get_basic_features(self, text):
        """Zwraca podstawowe cechy tekstu w przypadku błędu LLM"""
        words = text.split()
        sentences = text.split('.')
        return {
            'clarity': 3,
            'coherence': 3,
            'completeness': 3,
            'conciseness': 3,
            'engagement': 3,
            'language_quality': 3,
            'main_points': len(sentences),
            'technical_terms': 0,
            'sentiment': 3,
            'length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences)
        }

    def save_summary_and_rating(self, summary_id, summary_text, rating, book_title, book_author):
        try:
            # Upewnij się, że klient S3 jest zainicjalizowany
            self._initialize()
            
            # Przygotowanie danych
            summary_data = {
                'summary_id': summary_id,
                'summary_text': summary_text,
                'book_title': book_title,
                'book_author': book_author,
                'timestamp': datetime.now().isoformat()
            }
            
            rating_data = {
                'summary_id': summary_id,
                'rating': float(rating),  # Upewnij się, że rating jest float
                'timestamp': datetime.now().isoformat()
            }
            
            # Konwertuj dane na JSON i następnie na bytes
            summary_json = json.dumps(summary_data, ensure_ascii=False)
            rating_json = json.dumps(rating_data, ensure_ascii=False)
            
            # Zapisz bezpośrednio do S3
            self._s3.put_object(
                Bucket=self._bucket_name,
                Key=f'summaries/{summary_id}.json',
                Body=summary_json.encode('utf-8'),
                ContentType='application/json'
            )
            
            self._s3.put_object(
                Bucket=self._bucket_name,
                Key=f'ratings/{summary_id}.json',
                Body=rating_json.encode('utf-8'),
                ContentType='application/json'
            )
            
            # Po dodaniu nowej oceny, zaktualizuj model
            self._load_and_train_model()
            
            return True
        except Exception as e:
            raise