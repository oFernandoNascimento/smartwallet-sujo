import unittest
from unittest.mock import MagicMock
import sys

# ==============================================================================
# 1. PREPARAÇÃO DO AMBIENTE (MOCKING TOTAL)
# Enganamos TODAS as bibliotecas externas para o teste rodar isolado.
# ==============================================================================

# 1. Mock do Streamlit (Interface)
# Isso impede o erro "No secrets found"
mock_st = MagicMock()
mock_st.secrets = {
    "GEMINI_KEY": "fake_key_123",
    "DATABASE_URL": "postgres://user:pass@localhost:5432/db"
}
sys.modules["streamlit"] = mock_st

# 2. Mock do Psycopg2 (Banco de Dados)
# Isso impede o erro de conexão com banco
mock_pg = MagicMock()
sys.modules["psycopg2"] = mock_pg

# 3. Mock do Google GenAI (BIBLIOTECA DE IA)
# Isso impede o erro "FutureWarning" e validação de chave
mock_genai = MagicMock()
sys.modules["google.generativeai"] = mock_genai

# ==============================================================================
# 2. IMPORTAÇÃO DO SISTEMA
# Agora que tudo está "falso", podemos importar sem medo.
# Note que NÃO importamos SecureTransactionDAO, pois ela não existe mais.
# ==============================================================================
from smart_wallet import UIManager, AIManager, SecurityManager

# ==============================================================================
# 3. TESTES DE LÓGICA
# ==============================================================================
class TestSmartWalletLogic(unittest.TestCase):
    
    def test_formatacao_dinheiro(self):
        """Testa se 1000 vira R$ 1.000,00"""
        val = UIManager.format_money(1000.0)
        self.assertIn("R$", val)
        self.assertIn("1.000,00", val)

    def test_seguranca_senha(self):
        """Testa se a senha é criptografada"""
        pwd = "senha_secreta"
        hashed = SecurityManager.hash_pwd(pwd)
        self.assertNotEqual(pwd, hashed)
        self.assertEqual(len(hashed), 64)

    def test_ai_extracao_json(self):
        """Testa se o sistema sabe ler a resposta da IA"""
        texto_ia = 'Aqui está: {"amount": 50.5, "category": "Transporte"}'
        dados = AIManager.extract_json(texto_ia)
        self.assertEqual(dados['amount'], 50.5)
        self.assertEqual(dados['category'], "Transporte")

if __name__ == '__main__':
    unittest.main()