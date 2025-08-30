"""Observability and tracing utilities."""

import logging
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, TypeVar

from fastapi import Request, Response

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

# Logger
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def get_request_id() -> str:
    """Get current request ID."""
    return request_id_var.get()


def get_trace_id() -> str:
    """Get current trace ID."""
    return trace_id_var.get()


def set_request_context(request_id: str, trace_id: str | None = None) -> None:
    """Set request context variables."""
    request_id_var.set(request_id)
    trace_id_var.set(trace_id or request_id)


async def request_middleware(request: Request, call_next: Callable) -> Response:
    """Request middleware for observability."""
    # Generate request ID
    request_id = str(uuid.uuid4())
    trace_id = request.headers.get("x-trace-id", request_id)
    
    # Set context
    set_request_context(request_id, trace_id)
    
    # Add to request state
    request.state.request_id = request_id
    request.state.trace_id = trace_id
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["x-request-id"] = request_id
    response.headers["x-trace-id"] = trace_id
    response.headers["x-process-time"] = str(process_time)
    
    # Log request
    logger.info(
        "Request processed",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "process_time": process_time,
        },
    )
    
    return response


def trace_function(name: str | None = None) -> Callable[[F], F]:
    """Decorator to trace function execution."""
    def decorator(func: F) -> F:
        function_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            request_id = get_request_id()
            trace_id = get_trace_id()
            
            logger.info(
                f"Starting {function_name}",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "function": function_name,
                },
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Completed {function_name}",
                    extra={
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "function": function_name,
                        "duration": duration,
                        "status": "success",
                    },
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"Failed {function_name}",
                    extra={
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "function": function_name,
                        "duration": duration,
                        "status": "error",
                        "error": str(e),
                    },
                    exc_info=True,
                )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            request_id = get_request_id()
            trace_id = get_trace_id()
            
            logger.info(
                f"Starting {function_name}",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "function": function_name,
                },
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Completed {function_name}",
                    extra={
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "function": function_name,
                        "duration": duration,
                        "status": "success",
                    },
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"Failed {function_name}",
                    extra={
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "function": function_name,
                        "duration": duration,
                        "status": "error",
                        "error": str(e),
                    },
                    exc_info=True,
                )
                
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator
