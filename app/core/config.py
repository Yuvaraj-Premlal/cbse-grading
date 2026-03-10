from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # Azure Key Vault
    KEY_VAULT_URL: str = os.getenv("KEY_VAULT_URL")

    def __init__(self):
        self._client = None
        self._load_secrets()

    def _get_client(self):
        if not self._client:
            credential = DefaultAzureCredential()
            self._client = SecretClient(
                vault_url=self.KEY_VAULT_URL,
                credential=credential
            )
        return self._client

    def _load_secrets(self):
        try:
            client = self._get_client()
            self.OPENAI_ENDPOINT   = client.get_secret("AZURE-OPENAI-ENDPOINT").value
            self.OPENAI_KEY        = client.get_secret("AZURE-OPENAI-KEY").value
            self.OPENAI_DEPLOYMENT = client.get_secret("AZURE-OPENAI-DEPLOYMENT").value
            self.DB_CONNECTION     = client.get_secret("DB-CONNECTION-STRING").value
            self.JWT_SECRET        = client.get_secret("JWT-SECRET").value
        except Exception as e:
            print(f"Key Vault error: {e}")
            print("Falling back to .env file")
            self.OPENAI_ENDPOINT   = os.getenv("OPENAI_ENDPOINT")
            self.OPENAI_KEY        = os.getenv("OPENAI_KEY")
            self.OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")
            self.DB_CONNECTION     = os.getenv("DB_CONNECTION_STRING")
            self.JWT_SECRET        = os.getenv("JWT_SECRET")

    # App settings
    APP_NAME:    str = "CBSE AI Grading"
    APP_VERSION: str = "2.0.0"
    ALGORITHM:   str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

settings = Settings()
