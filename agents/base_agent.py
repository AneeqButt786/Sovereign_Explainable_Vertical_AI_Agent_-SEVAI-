"""
Base Agent - Abstract class for all agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from core.logging_config import get_logger

logger = get_logger("base_agent")


@dataclass
class AgentResult:
    """Result from agent execution"""
    agent_id: str
    output: Any
    confidence: float  # 0-1
    reasoning_steps: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    duration_ms: float
    metadata: Dict[str, Any]
    timestamp: datetime


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Defines common interface and shared functionality.
    """
    
    def __init__(self, agent_id: str, description: str = ""):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description
        """
        self.agent_id = agent_id
        self.description = description
        self.logger = get_logger(f"agent.{agent_id}")
    
    @abstractmethod
    def ingest(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate input data
        
        Args:
            input_data: Raw input data
            
        Returns:
            Processed input ready for reasoning
        """
        pass
    
    @abstractmethod
    def reason(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply reasoning logic to processed input
        
        Args:
            processed_input: Output from ingest()
            
        Returns:
            Reasoning result with conclusions
        """
        pass
    
    @abstractmethod
    def output(self, reasoning_result: Dict[str, Any]) -> Any:
        """
        Format reasoning result for output
        
        Args:
            reasoning_result: Output from reason()
            
        Returns:
            Formatted output
        """
        pass
    
    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute full agent pipeline: ingest -> reason -> output
        
        Args:
            input_data: Input data
            context: Additional context for reasoning
            
        Returns:
            AgentResult with complete execution details
        """
        start_time = time.time()
        self.logger.info(f"Executing agent: {self.agent_id}")
        
        try:
            # Track reasoning steps and tool calls
            reasoning_steps = []
            tool_calls = []
            
            # Ingest
            self.logger.debug("Ingesting input")
            processed_input = self.ingest(input_data)
            reasoning_steps.append({
                "step": "ingest",
                "result": "Input validated and processed"
            })
            
            # Add context if provided
            if context:
                processed_input["context"] = context
            
            # Reason
            self.logger.debug("Applying reasoning")
            reasoning_result = self.reason(processed_input)
            reasoning_steps.append({
                "step": "reason",
                "result": f"Reasoning complete with {len(reasoning_result.get('conclusions', []))} conclusions"
            })
            
            # Output
            self.logger.debug("Formatting output")
            final_output = self.output(reasoning_result)
            reasoning_steps.append({
                "step": "output",
                "result": "Output formatted"
            })
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract confidence (default to 1.0 if not provided)
            confidence = reasoning_result.get("confidence", 1.0)
            
            # Create result
            result = AgentResult(
                agent_id=self.agent_id,
                output=final_output,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                tool_calls=tool_calls,
                duration_ms=duration_ms,
                metadata={
                    "description": self.description,
                    "input_keys": list(input_data.keys()),
                    "has_context": context is not None
                },
                timestamp=datetime.now(timezone.utc)
            )
            
            self.logger.info(
                f"Agent execution complete: {duration_ms:.2f}ms, "
                f"confidence: {confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}", exc_info=True)
            raise
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}')>"
