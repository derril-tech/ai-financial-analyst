"""Security and access control utilities."""

import hashlib
import secrets
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

import jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.core.config import settings
from app.core.observability import trace_function
from app.models.user import User
from app.models.audit_log import AuditLog


class UserRole(Enum):
    """User roles with different access levels."""
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class PolicyMode(Enum):
    """Policy modes for answer framing."""
    INFORMATIONAL = "informational"
    OPINIONATED = "opinionated"


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Central security manager for authentication and authorization."""
    
    def __init__(self) -> None:
        """Initialize security manager."""
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            return None
    
    @trace_function("security_manager.check_permissions")
    def check_permissions(
        self, 
        user_role: str, 
        required_role: str, 
        resource: str = None
    ) -> bool:
        """Check if user has required permissions."""
        role_hierarchy = {
            UserRole.VIEWER.value: 1,
            UserRole.ANALYST.value: 2,
            UserRole.ADMIN.value: 3,
            UserRole.SUPER_ADMIN.value: 4,
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    @trace_function("security_manager.log_access")
    async def log_access(
        self,
        db: AsyncSession,
        user_id: str,
        org_id: str,
        action: str,
        resource: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log user access for audit trail."""
        audit_entry = AuditLog(
            id=secrets.token_urlsafe(16),
            org_id=org_id,
            user_id=user_id,
            action=action,
            resource=resource,
            meta=metadata or {},
        )
        
        db.add(audit_entry)
        await db.commit()


class RowLevelSecurity:
    """Row-level security implementation for multitenancy."""
    
    @staticmethod
    async def create_rls_policies(db: AsyncSession) -> None:
        """Create row-level security policies for all tables."""
        
        # Enable RLS on all tenant tables
        tables_with_org_id = [
            "organizations",
            "users", 
            "documents",
            "artifacts",
            "facts_xbrl",
            "transcripts",
            "vector_index",
            "queries",
            "answers",
            "alerts",
            "audit_log",
        ]
        
        for table in tables_with_org_id:
            # Enable RLS
            await db.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;"))
            
            # Create policy for org isolation
            policy_sql = f"""
                CREATE POLICY {table}_org_isolation ON {table}
                FOR ALL
                TO authenticated_users
                USING (org_id = current_setting('app.current_org_id')::text);
            """
            
            try:
                await db.execute(text(policy_sql))
            except Exception as e:
                # Policy might already exist
                print(f"Policy creation for {table} failed (might exist): {e}")
        
        await db.commit()
    
    @staticmethod
    async def set_org_context(db: AsyncSession, org_id: str) -> None:
        """Set organization context for RLS."""
        await db.execute(text(f"SET app.current_org_id = '{org_id}';"))
    
    @staticmethod
    async def create_authenticated_role(db: AsyncSession) -> None:
        """Create authenticated users role for RLS."""
        try:
            await db.execute(text("CREATE ROLE authenticated_users;"))
            await db.execute(text("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated_users;"))
            await db.commit()
        except Exception as e:
            print(f"Role creation failed (might exist): {e}")


class SecretManager:
    """Secure secret management for API keys and sensitive data."""
    
    def __init__(self) -> None:
        """Initialize secret manager."""
        self.encryption_key = self._derive_key(settings.SECRET_KEY)
    
    def _derive_key(self, master_key: str) -> bytes:
        """Derive encryption key from master key."""
        return hashlib.pbkdf2_hmac('sha256', master_key.encode(), b'salt', 100000)[:32]
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret value."""
        # In production, use proper encryption like Fernet
        # For now, use simple base64 encoding (NOT SECURE)
        import base64
        return base64.b64encode(secret.encode()).decode()
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret value."""
        # In production, use proper decryption
        import base64
        return base64.b64decode(encrypted_secret.encode()).decode()
    
    @trace_function("secret_manager.store_api_key")
    async def store_api_key(
        self,
        db: AsyncSession,
        org_id: str,
        provider: str,
        api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store encrypted API key for organization."""
        encrypted_key = self.encrypt_secret(api_key)
        
        # Store in database (would need api_keys table)
        # For now, return the encrypted key
        return encrypted_key
    
    @trace_function("secret_manager.get_api_key")
    async def get_api_key(
        self,
        db: AsyncSession,
        org_id: str,
        provider: str,
    ) -> Optional[str]:
        """Retrieve and decrypt API key for organization."""
        # In production, query database for encrypted key
        # For now, return None
        return None


class ContentFilter:
    """Content filtering for MNPI/PII and sensitive information."""
    
    def __init__(self) -> None:
        """Initialize content filter."""
        # MNPI keywords
        self.mnpi_keywords = [
            "insider", "material", "non-public", "confidential",
            "merger", "acquisition", "earnings", "guidance",
            "restructuring", "layoffs", "partnership", "contract",
        ]
        
        # PII patterns
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
    
    @trace_function("content_filter.scan_content")
    def scan_content(self, content: str) -> Dict[str, Any]:
        """Scan content for MNPI/PII and return risk assessment."""
        import re
        
        risks = {
            "mnpi_risk": False,
            "pii_risk": False,
            "risk_score": 0.0,
            "flagged_terms": [],
            "recommendations": [],
        }
        
        content_lower = content.lower()
        
        # Check for MNPI keywords
        mnpi_matches = []
        for keyword in self.mnpi_keywords:
            if keyword in content_lower:
                mnpi_matches.append(keyword)
                risks["mnpi_risk"] = True
        
        # Check for PII patterns
        pii_matches = []
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, content)
            if matches:
                pii_matches.extend(matches)
                risks["pii_risk"] = True
        
        # Calculate risk score
        risk_score = 0.0
        if risks["mnpi_risk"]:
            risk_score += 0.7
        if risks["pii_risk"]:
            risk_score += 0.8
        
        risks["risk_score"] = min(risk_score, 1.0)
        risks["flagged_terms"] = mnpi_matches + pii_matches
        
        # Add recommendations
        if risks["mnpi_risk"]:
            risks["recommendations"].append("Content may contain material non-public information")
        if risks["pii_risk"]:
            risks["recommendations"].append("Content contains personally identifiable information")
        
        return risks
    
    @trace_function("content_filter.quarantine_content")
    async def quarantine_content(
        self,
        db: AsyncSession,
        content: str,
        risk_assessment: Dict[str, Any],
        org_id: str,
        user_id: str,
    ) -> str:
        """Quarantine risky content and return quarantine ID."""
        quarantine_id = secrets.token_urlsafe(16)
        
        # In production, store in quarantine table
        # Log the quarantine action
        security_manager = SecurityManager()
        await security_manager.log_access(
            db=db,
            user_id=user_id,
            org_id=org_id,
            action="content_quarantined",
            resource=f"quarantine:{quarantine_id}",
            metadata={
                "risk_assessment": risk_assessment,
                "content_length": len(content),
            },
        )
        
        return quarantine_id


class PromptInjectionGuard:
    """Protection against prompt injection attacks."""
    
    def __init__(self) -> None:
        """Initialize prompt injection guard."""
        self.injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"forget\s+everything",
            r"you\s+are\s+now",
            r"new\s+instructions",
            r"system\s*:",
            r"assistant\s*:",
            r"human\s*:",
            r"<\|.*?\|>",
            r"\[INST\].*?\[/INST\]",
        ]
        
        self.suspicious_tokens = [
            "jailbreak", "roleplay", "pretend", "imagine",
            "hypothetically", "what if you were",
        ]
    
    @trace_function("prompt_injection_guard.scan_prompt")
    def scan_prompt(self, prompt: str) -> Dict[str, Any]:
        """Scan prompt for injection attempts."""
        import re
        
        result = {
            "is_safe": True,
            "risk_score": 0.0,
            "detected_patterns": [],
            "suspicious_tokens": [],
        }
        
        prompt_lower = prompt.lower()
        
        # Check for injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                result["detected_patterns"].append(pattern)
                result["is_safe"] = False
                result["risk_score"] += 0.3
        
        # Check for suspicious tokens
        for token in self.suspicious_tokens:
            if token in prompt_lower:
                result["suspicious_tokens"].append(token)
                result["risk_score"] += 0.1
        
        result["risk_score"] = min(result["risk_score"], 1.0)
        
        if result["risk_score"] > 0.5:
            result["is_safe"] = False
        
        return result
    
    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt by removing potentially dangerous content."""
        import re
        
        sanitized = prompt
        
        # Remove system/assistant/human markers
        sanitized = re.sub(r'(system|assistant|human)\s*:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove instruction markers
        sanitized = re.sub(r'<\|.*?\|>', '', sanitized)
        sanitized = re.sub(r'\[INST\].*?\[/INST\]', '', sanitized)
        
        # Limit length
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000] + "..."
        
        return sanitized.strip()


class ComplianceFramework:
    """Compliance framework for regulatory requirements."""
    
    def __init__(self) -> None:
        """Initialize compliance framework."""
        self.policy_mode = PolicyMode.INFORMATIONAL
        
        self.disclaimers = {
            PolicyMode.INFORMATIONAL: (
                "This analysis is for informational purposes only and does not constitute "
                "investment advice. Please consult with a qualified financial advisor "
                "before making investment decisions."
            ),
            PolicyMode.OPINIONATED: (
                "This analysis contains forward-looking statements and opinions that are "
                "subject to significant risks and uncertainties. Past performance does not "
                "guarantee future results. This is not a recommendation to buy or sell securities."
            ),
        }
    
    def set_policy_mode(self, mode: PolicyMode) -> None:
        """Set compliance policy mode."""
        self.policy_mode = mode
    
    def frame_response(self, response: str, mode: Optional[PolicyMode] = None) -> str:
        """Frame response according to compliance policy."""
        active_mode = mode or self.policy_mode
        disclaimer = self.disclaimers[active_mode]
        
        if active_mode == PolicyMode.INFORMATIONAL:
            framed_response = f"Based on available information: {response}\n\n{disclaimer}"
        else:
            framed_response = f"{response}\n\n{disclaimer}"
        
        return framed_response
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """Validate response for compliance issues."""
        validation = {
            "is_compliant": True,
            "issues": [],
            "recommendations": [],
        }
        
        # Check for investment advice language
        advice_keywords = ["buy", "sell", "invest", "recommend", "should purchase"]
        for keyword in advice_keywords:
            if keyword in response.lower():
                validation["issues"].append(f"Contains potential investment advice: '{keyword}'")
                validation["is_compliant"] = False
        
        # Check for price targets without disclaimers
        if "target price" in response.lower() and "not investment advice" not in response.lower():
            validation["issues"].append("Price target without proper disclaimer")
            validation["is_compliant"] = False
        
        if not validation["is_compliant"]:
            validation["recommendations"].append("Add appropriate disclaimers")
            validation["recommendations"].append("Review content for investment advice language")
        
        return validation
