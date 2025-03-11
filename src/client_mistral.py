import pandas as pd
import json
from dotenv import load_dotenv
import os
from mistralai import Mistral

class MistralAgent:
    def __init__(self):
        self.client = None
        self.json_data_str = None

    def csv_to_json(self, df: pd.DataFrame) -> str:
        """Convertit un DataFrame en une chaîne JSON compacte."""
        json_data = df.to_json(orient='records')
        return json.dumps(json.loads(json_data), indent=0).replace('\n', '')

    def get_response(self, json_str: str) -> str:
        """Envoie une requête à l'API Mistral et retourne la réponse."""
        response = self.client.agents.complete(
            agent_id="ag:753184c9:20250311:untitled-agent:e063ae99",
            messages=[{"role": "user", "content": json_str}]
        )
        return response.choices[0].message.content.strip()

    def setup_client(self) -> None:
        """Configure le client Mistral avec la clé API."""
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("La clé API MISTRAL_API_KEY n'est pas définie dans l'environnement.")
        self.client = Mistral(api_key=api_key)

    def process_batch(self, df_batch: pd.DataFrame) -> str:
        """Traite un lot de données et retourne la réponse de l'API."""
        self.json_data_str = self.csv_to_json(df_batch)
        return self.get_response(self.json_data_str)

    def process_file(self, path: str) -> str:
        """Traite un fichier CSV en le divisant en lots si nécessaire."""
        self.setup_client()
        
        # Charger le fichier CSV
        df = pd.read_csv(path, sep=';')
        print(f"Nombre total de lignes : {len(df)}")  # Débogage
        
        # Définir la taille maximale d'un lot
        batch_size = 250
        
        # Si le nombre de lignes est <= 250, traiter en une seule fois
        if len(df) <= batch_size:
            self.json_data_str = self.csv_to_json(df)
            return self.get_response(self.json_data_str)
        
        # Sinon, diviser en lots
        responses = []
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            df_batch = df.iloc[start:end]
            print(f"Traitement du lot : lignes {start} à {end-1}")  # Débogage
            batch_response = self.process_batch(df_batch)
            responses.append(batch_response)
        
        # Combiner les réponses des lots
        combined_response = "\n\n".join(responses)
        print(f"Réponses combinées : {combined_response[:100]}...")  # Débogage (limité à 100 caractères)
        return combined_response

# Test de la classe (optionnel)
if __name__ == "__main__":
    agent = MistralAgent()
    result = agent.process_file("votre_fichier.csv")
    print(result)