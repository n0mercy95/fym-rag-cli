import logging
import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
from src.config.settings import REDIS_URL

# Se acopla a tu sistema de logs centralizado
logger = logging.getLogger(__name__)

class RedisMemoryManager:
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self._verificar_conexion()

    def _verificar_conexion(self):
        """Valida que el servicio Redis esté accesible al iniciar la clase."""
        try:
            client = redis.from_url(self.redis_url)
            client.ping()
            logger.info("Conexión a Redis establecida correctamente.")
        except redis.ConnectionError as e:
            logger.critical(f"Error crítico: No se pudo conectar a Redis en {self.redis_url}. Detalles: {e}")
            raise

    def get_session_history(self, session_id: str) -> RedisChatMessageHistory:
        """
        Recupera o inicializa el historial de chat para un ID de sesión específico.
        LangChain utilizará este objeto para leer y escribir mensajes automáticamente.
        """
        try:
            logger.debug(f"Accediendo al historial en Redis para la sesión: {session_id}")
            history = RedisChatMessageHistory(
                session_id=session_id,
                url=self.redis_url
            )
            return history
        except Exception as e:
            logger.error(f"Error al instanciar el historial para la sesión {session_id}. Detalles: {e}")
            raise