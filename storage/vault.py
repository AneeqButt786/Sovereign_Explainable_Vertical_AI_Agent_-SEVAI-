"""
Explainability Vault - Immutable Audit Trail Storage
SQLite-based append-only storage for HIPAA compliance
"""

from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime, timezone
import hashlib
import json
from core.config import get_settings
from core.logging_config import get_logger, get_audit_logger

logger = get_logger("vault")
audit_logger = get_audit_logger()

Base = declarative_base()


class Input(Base):
    """Input records - raw inputs to the system"""
    __tablename__ = "inputs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True)
    meta_data = Column(JSON)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prev_hash = Column(String(64))  # Hash chaining for immutability


class AgentExecution(Base):
    """Agent execution records"""
    __tablename__ = "agent_executions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    input_id = Column(Integer, nullable=False)
    agent_id = Column(String(100), nullable=False)
    agent_input = Column(Text)
    agent_output = Column(Text)
    tool_calls = Column(JSON)  # List of tool calls made
    duration_ms = Column(Float)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prev_hash = Column(String(64))  # Hash chaining


class CausalStep(Base):
    """Causal reasoning steps"""
    __tablename__ = "causal_steps"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(Integer, nullable=False)
    premise = Column(Text, nullable=False)
    conclusion = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    evidence_refs = Column(JSON)  # References to supporting evidence
    reasoning_type = Column(String(50))  # symbolic, probabilistic, etc.
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prev_hash = Column(String(64))


class PolicyCheck(Base):
    """Policy compliance checks"""
    __tablename__ = "policy_checks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(Integer, nullable=False)
    policy_name = Column(String(100), nullable=False)
    result = Column(String(20), nullable=False)  # pass, warn, fail
    details = Column(JSON)
    violations = Column(JSON)  # List of violations if any
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prev_hash = Column(String(64))


class Output(Base):
    """Final outputs"""
    __tablename__ = "outputs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(Integer, nullable=False)
    conclusion = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_flags = Column(JSON)  # List of risk flags
    recommendations = Column(JSON)  # Recommended actions
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prev_hash = Column(String(64))


class ExplainabilityVault:
    """
    Manages immutable storage of all reasoning steps.
    Implements hash chaining for tamper detection.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize Explainability Vault
        
        Args:
            database_url: SQLAlchemy database URL
        """
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        
        # Create engine and session
        self.engine = create_engine(self.database_url, echo=False)
        Base.metadata.create_all(self.engine)
        
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info(f"Explainability Vault initialized: {self.database_url}")
        audit_logger.info("Vault initialized")
    
    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of data"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _get_last_hash(self, session: Session, table_class) -> Optional[str]:
        """Get hash of most recent record in table"""
        last_record = session.query(table_class).order_by(
            table_class.id.desc()
        ).first()
        return last_record.prev_hash if last_record else None
    
    def log_input(
        self,
        source: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log an input to the vault
        
        Args:
            source: Source of the input (e.g., "user", "file", "api")
            content: Input content
            metadata: Additional metadata
            
        Returns:
            Input ID
        """
        session = self.SessionLocal()
        try:
            # Compute content hash
            content_hash = self._compute_hash({"content": content})
            
            # Get previous hash for chaining
            prev_hash = self._get_last_hash(session, Input)
            
            # Create input record
            input_record = Input(
                source=source,
                content=content,
                content_hash=content_hash,
                meta_data=metadata or {},
                prev_hash=prev_hash
            )
            
            session.add(input_record)
            session.commit()
            
            input_id = input_record.id
            
            audit_logger.info(f"Input logged: ID={input_id}, source={source}")
            logger.debug(f"Input {input_id} logged with hash: {content_hash[:16]}...")
            
            return input_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log input: {e}")
            raise
        finally:
            session.close()
    
    def log_agent_execution(
        self,
        input_id: int,
        agent_id: str,
        agent_input: str,
        agent_output: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        duration_ms: Optional[float] = None
    ) -> int:
        """
        Log an agent execution
        
        Args:
            input_id: Related input ID
            agent_id: Identifier for the agent
            agent_input: Input to the agent
            agent_output: Output from the agent
            tool_calls: List of tool calls made
            duration_ms: Execution duration in milliseconds
            
        Returns:
            Execution ID
        """
        session = self.SessionLocal()
        try:
            prev_hash = self._get_last_hash(session, AgentExecution)
            
            execution = AgentExecution(
                input_id=input_id,
                agent_id=agent_id,
                agent_input=agent_input,
                agent_output=agent_output,
                tool_calls=tool_calls or [],
                duration_ms=duration_ms,
                prev_hash=prev_hash
            )
            
            session.add(execution)
            session.commit()
            
            execution_id = execution.id
            
            audit_logger.info(
                f"Agent execution logged: ID={execution_id}, agent={agent_id}"
            )
            
            return execution_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log agent execution: {e}")
            raise
        finally:
            session.close()
    
    def log_causal_step(
        self,
        execution_id: int,
        premise: str,
        conclusion: str,
        confidence: float,
        evidence_refs: Optional[List[str]] = None,
        reasoning_type: str = "symbolic"
    ) -> int:
        """
        Log a causal reasoning step
        
        Args:
            execution_id: Related execution ID
            premise: Premise/input for reasoning
            conclusion: Conclusion/output
            confidence: Confidence score (0-1)
            evidence_refs: References to supporting evidence
            reasoning_type: Type of reasoning used
            
        Returns:
            Step ID
        """
        session = self.SessionLocal()
        try:
            prev_hash = self._get_last_hash(session, CausalStep)
            
            step = CausalStep(
                execution_id=execution_id,
                premise=premise,
                conclusion=conclusion,
                confidence=confidence,
                evidence_refs=evidence_refs or [],
                reasoning_type=reasoning_type,
                prev_hash=prev_hash
            )
            
            session.add(step)
            session.commit()
            
            return step.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log causal step: {e}")
            raise
        finally:
            session.close()
    
    def log_policy_check(
        self,
        execution_id: int,
        policy_name: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        violations: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Log a policy check
        
        Args:
            execution_id: Related execution ID
            policy_name: Name of the policy checked
            result: Result (pass, warn, fail)
            details: Additional details
            violations: List of violations if any
            
        Returns:
            Check ID
        """
        session = self.SessionLocal()
        try:
            prev_hash = self._get_last_hash(session, PolicyCheck)
            
            check = PolicyCheck(
                execution_id=execution_id,
                policy_name=policy_name,
                result=result,
                details=details or {},
                violations=violations or [],
                prev_hash=prev_hash
            )
            
            session.add(check)
            session.commit()
            
            audit_logger.info(
                f"Policy check logged: policy={policy_name}, result={result}"
            )
            
            # Alert on failures
            if result == "fail":
                audit_logger.warning(
                    f"POLICY VIOLATION: {policy_name} - {violations}"
                )
            
            return check.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log policy check: {e}")
            raise
        finally:
            session.close()
    
    def log_output(
        self,
        execution_id: int,
        conclusion: str,
        confidence: float,
        risk_flags: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None
    ) -> int:
        """
        Log final output
        
        Args:
            execution_id: Related execution ID
            conclusion: Final conclusion
            confidence: Overall confidence score
            risk_flags: List of risk flags
            recommendations: Recommended actions
            
        Returns:
            Output ID
        """
        session = self.SessionLocal()
        try:
            prev_hash = self._get_last_hash(session, Output)
            
            output = Output(
                execution_id=execution_id,
                conclusion=conclusion,
                confidence=confidence,
                risk_flags=risk_flags or [],
                recommendations=recommendations or [],
                prev_hash=prev_hash
            )
            
            session.add(output)
            session.commit()
            
            audit_logger.info(
                f"Output logged: ID={output.id}, confidence={confidence:.2f}"
            )
            
            return output.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log output: {e}")
            raise
        finally:
            session.close()
    
    def get_reasoning_trail(self, execution_id: int) -> Dict[str, Any]:
        """
        Retrieve complete reasoning trail for an execution
        
        Args:
            execution_id: Execution ID to retrieve
            
        Returns:
            Complete reasoning trail with all steps
        """
        session = self.SessionLocal()
        try:
            # Get execution
            execution = session.query(AgentExecution).filter_by(
                id=execution_id
            ).first()
            
            if not execution:
                raise ValueError(f"Execution {execution_id} not found")
            
            # Get causal steps
            causal_steps = session.query(CausalStep).filter_by(
                execution_id=execution_id
            ).all()
            
            # Get policy checks
            policy_checks = session.query(PolicyCheck).filter_by(
                execution_id=execution_id
            ).all()
            
            # Get output
            output = session.query(Output).filter_by(
                execution_id=execution_id
            ).first()
            
            # Format trail
            trail = {
                "execution_id": execution_id,
                "agent_id": execution.agent_id,
                "agent_input": execution.agent_input,
                "agent_output": execution.agent_output,
                "tool_calls": execution.tool_calls,
                "duration_ms": execution.duration_ms,
                "causal_steps": [
                    {
                        "premise": step.premise,
                        "conclusion": step.conclusion,
                        "confidence": step.confidence,
                        "evidence_refs": step.evidence_refs,
                        "reasoning_type": step.reasoning_type
                    }
                    for step in causal_steps
                ],
                "policy_checks": [
                    {
                        "policy_name": check.policy_name,
                        "result": check.result,
                        "details": check.details,
                        "violations": check.violations
                    }
                    for check in policy_checks
                ],
                "output": {
                    "conclusion": output.conclusion if output else None,
                    "confidence": output.confidence if output else None,
                    "risk_flags": output.risk_flags if output else [],
                    "recommendations": output.recommendations if output else []
                } if output else None
            }
            
            return trail
            
        finally:
            session.close()


# Singleton instance
_vault_instance: Optional[ExplainabilityVault] = None


def get_vault() -> ExplainabilityVault:
    """Get or create singleton Vault instance"""
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = ExplainabilityVault()
    return _vault_instance
