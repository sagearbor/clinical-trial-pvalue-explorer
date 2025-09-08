"""Application monitoring and metrics collection."""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import asyncio
from dataclasses import dataclass, asdict
import json


logger = logging.getLogger(__name__)


# Prometheus metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['endpoint', 'method']
)

active_connections = Gauge(
    'websocket_active_connections',
    'Number of active WebSocket connections'
)

calculation_errors = Counter(
    'calculation_errors_total',
    'Total calculation errors',
    ['test_type', 'error_type']
)

llm_requests = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['provider', 'model', 'status']
)

llm_latency = Histogram(
    'llm_request_latency_seconds',
    'LLM API request latency',
    ['provider', 'model']
)

cpu_usage = Gauge('system_cpu_usage_percent', 'System CPU usage')
memory_usage = Gauge('system_memory_usage_percent', 'System memory usage')
disk_usage = Gauge('system_disk_usage_percent', 'System disk usage')


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    timestamp: datetime
    endpoint: str
    method: str
    duration: float
    status_code: int
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class HealthStatus:
    """System health status."""
    healthy: bool
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """Collect and aggregate application metrics."""
    
    def __init__(self, window_size: int = 3600):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Time window in seconds for metrics aggregation
        """
        self.window_size = window_size
        self.metrics_buffer = deque(maxlen=10000)
        self.error_buffer = deque(maxlen=1000)
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'p95_latency': 2.0,  # 2 seconds
            'cpu_usage': 80.0,  # 80% CPU
            'memory_usage': 85.0,  # 85% memory
            'disk_usage': 90.0  # 90% disk
        }
        self.alerts = []
        
        # Start background monitoring
        asyncio.create_task(self._monitor_system())
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        duration: float,
        status_code: int,
        error: Optional[str] = None
    ) -> None:
        """
        Record API request metrics.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration: Request duration in seconds
            status_code: HTTP status code
            error: Error message if any
        """
        # Update Prometheus metrics
        request_count.labels(
            endpoint=endpoint,
            method=method,
            status=str(status_code)
        ).inc()
        
        request_duration.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
        
        # Store in buffer
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            endpoint=endpoint,
            method=method,
            duration=duration,
            status_code=status_code,
            error=error
        )
        self.metrics_buffer.append(metric)
        
        # Check for alerts
        if error:
            self.error_buffer.append(metric)
            self._check_error_rate()
        
        if duration > self.alert_thresholds['p95_latency']:
            self._create_alert(
                'high_latency',
                f"High latency detected: {duration:.2f}s on {endpoint}"
            )
    
    def record_calculation_error(
        self,
        test_type: str,
        error_type: str,
        error_message: str
    ) -> None:
        """
        Record calculation error.
        
        Args:
            test_type: Statistical test type
            error_type: Type of error
            error_message: Error message
        """
        calculation_errors.labels(
            test_type=test_type,
            error_type=error_type
        ).inc()
        
        logger.error(f"Calculation error in {test_type}: {error_message}")
        
        self._create_alert(
            'calculation_error',
            f"Calculation error in {test_type}: {error_type}"
        )
    
    def record_llm_request(
        self,
        provider: str,
        model: str,
        duration: float,
        success: bool
    ) -> None:
        """
        Record LLM API request.
        
        Args:
            provider: LLM provider (openai, anthropic, gemini)
            model: Model name
            duration: Request duration
            success: Whether request succeeded
        """
        status = 'success' if success else 'error'
        
        llm_requests.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        llm_latency.labels(
            provider=provider,
            model=model
        ).observe(duration)
    
    async def _monitor_system(self) -> None:
        """Monitor system resources."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage.set(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_usage.set(disk.percent)
                
                # Check thresholds
                if cpu_percent > self.alert_thresholds['cpu_usage']:
                    self._create_alert(
                        'high_cpu',
                        f"High CPU usage: {cpu_percent:.1f}%"
                    )
                
                if memory.percent > self.alert_thresholds['memory_usage']:
                    self._create_alert(
                        'high_memory',
                        f"High memory usage: {memory.percent:.1f}%"
                    )
                
                if disk.percent > self.alert_thresholds['disk_usage']:
                    self._create_alert(
                        'high_disk',
                        f"High disk usage: {disk.percent:.1f}%"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _check_error_rate(self) -> None:
        """Check error rate and create alert if threshold exceeded."""
        if not self.metrics_buffer:
            return
        
        # Calculate error rate for last window
        cutoff_time = datetime.now() - timedelta(seconds=self.window_size)
        recent_metrics = [
            m for m in self.metrics_buffer
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return
        
        error_count = sum(1 for m in recent_metrics if m.error)
        error_rate = error_count / len(recent_metrics)
        
        if error_rate > self.alert_thresholds['error_rate']:
            self._create_alert(
                'high_error_rate',
                f"High error rate: {error_rate:.1%}"
            )
    
    def _create_alert(self, alert_type: str, message: str) -> None:
        """
        Create an alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
        """
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': self._get_severity(alert_type)
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {alert}")
        
        # Trigger notification (implement based on requirements)
        # self._send_notification(alert)
    
    def _get_severity(self, alert_type: str) -> str:
        """Get alert severity level."""
        severity_map = {
            'high_cpu': 'warning',
            'high_memory': 'warning',
            'high_disk': 'critical',
            'high_error_rate': 'critical',
            'high_latency': 'warning',
            'calculation_error': 'error'
        }
        return severity_map.get(alert_type, 'info')
    
    def get_metrics_summary(
        self,
        time_range: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Args:
            time_range: Time range in seconds (default: window_size)
            
        Returns:
            Metrics summary
        """
        if time_range is None:
            time_range = self.window_size
        
        cutoff_time = datetime.now() - timedelta(seconds=time_range)
        recent_metrics = [
            m for m in self.metrics_buffer
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {
                'total_requests': 0,
                'error_rate': 0,
                'avg_latency': 0,
                'p95_latency': 0,
                'p99_latency': 0
            }
        
        # Calculate statistics
        latencies = [m.duration for m in recent_metrics]
        latencies.sort()
        
        error_count = sum(1 for m in recent_metrics if m.error)
        
        p95_index = int(len(latencies) * 0.95)
        p99_index = int(len(latencies) * 0.99)
        
        return {
            'total_requests': len(recent_metrics),
            'error_rate': error_count / len(recent_metrics),
            'avg_latency': sum(latencies) / len(latencies),
            'p95_latency': latencies[p95_index] if p95_index < len(latencies) else 0,
            'p99_latency': latencies[p99_index] if p99_index < len(latencies) else 0,
            'requests_per_second': len(recent_metrics) / time_range
        }
    
    def get_endpoint_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics grouped by endpoint."""
        endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'errors': 0,
            'total_duration': 0,
            'latencies': []
        })
        
        for metric in self.metrics_buffer:
            endpoint = metric.endpoint
            endpoint_metrics[endpoint]['count'] += 1
            endpoint_metrics[endpoint]['total_duration'] += metric.duration
            endpoint_metrics[endpoint]['latencies'].append(metric.duration)
            if metric.error:
                endpoint_metrics[endpoint]['errors'] += 1
        
        # Calculate statistics
        result = {}
        for endpoint, data in endpoint_metrics.items():
            if data['count'] > 0:
                latencies = sorted(data['latencies'])
                p95_index = int(len(latencies) * 0.95)
                
                result[endpoint] = {
                    'count': data['count'],
                    'error_rate': data['errors'] / data['count'],
                    'avg_latency': data['total_duration'] / data['count'],
                    'p95_latency': latencies[p95_index] if p95_index < len(latencies) else 0
                }
        
        return result
    
    def get_health_status(self) -> HealthStatus:
        """
        Get system health status.
        
        Returns:
            Health status
        """
        checks = {
            'api': self._check_api_health(),
            'database': self._check_database_health(),
            'llm': self._check_llm_health(),
            'system': self._check_system_health()
        }
        
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        healthy = all(checks.values())
        
        return HealthStatus(
            healthy=healthy,
            checks=checks,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def _check_api_health(self) -> bool:
        """Check API health."""
        # Check error rate
        summary = self.get_metrics_summary(time_range=300)  # Last 5 minutes
        return summary['error_rate'] < self.alert_thresholds['error_rate']
    
    def _check_database_health(self) -> bool:
        """Check database health."""
        # Implement database health check
        # For now, return True
        return True
    
    def _check_llm_health(self) -> bool:
        """Check LLM service health."""
        # Could check recent LLM request success rate
        return True
    
    def _check_system_health(self) -> bool:
        """Check system resource health."""
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        return (
            cpu < self.alert_thresholds['cpu_usage'] and
            memory < self.alert_thresholds['memory_usage'] and
            disk < self.alert_thresholds['disk_usage']
        )
    
    def export_metrics_prometheus(self) -> bytes:
        """Export metrics in Prometheus format."""
        return generate_latest()
    
    def export_metrics_json(self) -> str:
        """Export metrics in JSON format."""
        data = {
            'summary': self.get_metrics_summary(),
            'endpoints': self.get_endpoint_metrics(),
            'health': self.get_health_status().to_dict(),
            'alerts': self.alerts[-10:]  # Last 10 alerts
        }
        return json.dumps(data, indent=2, default=str)


# Global metrics collector instance
metrics_collector = MetricsCollector()


class PerformanceTracker:
    """Context manager for tracking request performance."""
    
    def __init__(self, endpoint: str, method: str):
        """
        Initialize performance tracker.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
        """
        self.endpoint = endpoint
        self.method = method
        self.start_time = None
        self.error = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metrics."""
        duration = time.time() - self.start_time
        
        if exc_type:
            self.error = str(exc_val)
            status_code = 500
        else:
            status_code = 200
        
        metrics_collector.record_request(
            endpoint=self.endpoint,
            method=self.method,
            duration=duration,
            status_code=status_code,
            error=self.error
        )
        
        return False  # Don't suppress exceptions