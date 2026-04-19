"""
Agente base para el sistema multi-agente financiero
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Mensaje entre agentes"""
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisResult:
    """Resultado de análisis de un agente"""
    agent_name: str
    analysis_type: str
    confidence: float
    recommendation: str
    reasoning: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """Clase base para todos los agentes"""
    
    def __init__(self, name: str, role: str, description: str = ""):
        self.name = name
        self.role = role
        self.description = description
        self.memory = []
        self.message_queue = []
        self.max_memory_size = 100
        
    def log(self, message: str):
        """Registra un mensaje de log"""
        logger.info(f"[{self.name}] {message}")
    
    def add_to_memory(self, item: Dict):
        """Añade información a la memoria del agente"""
        self.memory.append({
            'timestamp': datetime.now(),
            'data': item
        })
        
        # Limitar tamaño de memoria
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]
    
    def get_memory(self, n_recent: int = 10) -> List[Dict]:
        """Obtiene los n elementos más recientes de la memoria"""
        return self.memory[-n_recent:]
    
    def send_message(self, recipient: str, message_type: str, content: Dict):
        """Envía un mensaje a otro agente"""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content
        )
        self.message_queue.append(message)
        self.log(f"Mensaje enviado a {recipient}: {message_type}")
    
    def receive_message(self, message: AgentMessage):
        """Recibe un mensaje de otro agente"""
        self.log(f"Mensaje recibido de {message.sender}: {message.message_type}")
        self.process_message(message)
    
    @abstractmethod
    def process_message(self, message: AgentMessage):
        """Procesa un mensaje recibido - debe ser implementado por subclases"""
        pass
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """Realiza análisis - debe ser implementado por subclases"""
        pass
    
    def get_context(self) -> str:
        """Obtiene el contexto del agente para prompts"""
        return f"""
Eres {self.name}, un agente especializado en {self.role}.
{self.description}

Tu objetivo es analizar la información proporcionada y ofrecer insights valiosos.
        """.strip()


class AgentCommunicationBus:
    """Bus de comunicación entre agentes"""
    
    def __init__(self):
        self.agents = {}
        self.message_history = []
    
    def register_agent(self, agent: BaseAgent):
        """Registra un agente en el bus"""
        self.agents[agent.name] = agent
        logger.info(f"Agente registrado: {agent.name}")
    
    def send_message(self, sender: str, recipient: str, message_type: str, content: Dict):
        """Envía un mensaje entre agentes"""
        message = AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content
        )
        
        self.message_history.append(message)
        
        if recipient in self.agents:
            self.agents[recipient].receive_message(message)
        else:
            logger.warning(f"Agente destinatario no encontrado: {recipient}")
    
    def broadcast(self, sender: str, message_type: str, content: Dict, 
                 exclude: List[str] = None):
        """Envía un mensaje a todos los agentes"""
        exclude = exclude or []
        
        for agent_name, agent in self.agents.items():
            if agent_name not in exclude:
                self.send_message(sender, agent_name, message_type, content)
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Obtiene un agente por nombre"""
        return self.agents.get(name)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Obtiene todos los agentes registrados"""
        return list(self.agents.values())
