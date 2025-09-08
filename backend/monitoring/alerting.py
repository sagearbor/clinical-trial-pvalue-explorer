"""Alerting and notification system."""

import smtplib
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
import json
import aiohttp
from enum import Enum


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert data class."""
    id: str
    type: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'type': self.type,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = self._setup_channels()
        self.alert_rules = self._setup_rules()
        self.alert_counter = 0
        
        # Start background tasks
        asyncio.create_task(self._process_alerts())
        asyncio.create_task(self._check_alert_resolution())
    
    def _setup_channels(self) -> Dict[AlertChannel, Any]:
        """Setup notification channels."""
        channels = {}
        
        # Email channel
        if self.config.get('email', {}).get('enabled'):
            channels[AlertChannel.EMAIL] = EmailNotifier(
                self.config['email']
            )
        
        # Slack channel
        if self.config.get('slack', {}).get('enabled'):
            channels[AlertChannel.SLACK] = SlackNotifier(
                self.config['slack']
            )
        
        # Webhook channel
        if self.config.get('webhook', {}).get('enabled'):
            channels[AlertChannel.WEBHOOK] = WebhookNotifier(
                self.config['webhook']
            )
        
        # Log channel (always enabled)
        channels[AlertChannel.LOG] = LogNotifier()
        
        return channels
    
    def _setup_rules(self) -> List[Dict[str, Any]]:
        """Setup alert rules."""
        default_rules = [
            {
                'type': 'high_error_rate',
                'severity': AlertSeverity.CRITICAL,
                'channels': [AlertChannel.EMAIL, AlertChannel.SLACK],
                'cooldown': 300  # 5 minutes
            },
            {
                'type': 'high_cpu',
                'severity': AlertSeverity.WARNING,
                'channels': [AlertChannel.LOG],
                'cooldown': 600  # 10 minutes
            },
            {
                'type': 'high_memory',
                'severity': AlertSeverity.WARNING,
                'channels': [AlertChannel.LOG],
                'cooldown': 600
            },
            {
                'type': 'high_disk',
                'severity': AlertSeverity.CRITICAL,
                'channels': [AlertChannel.EMAIL],
                'cooldown': 1800  # 30 minutes
            },
            {
                'type': 'calculation_error',
                'severity': AlertSeverity.ERROR,
                'channels': [AlertChannel.LOG, AlertChannel.SLACK],
                'cooldown': 60
            },
            {
                'type': 'high_latency',
                'severity': AlertSeverity.WARNING,
                'channels': [AlertChannel.LOG],
                'cooldown': 300
            }
        ]
        
        # Merge with custom rules from config
        custom_rules = self.config.get('rules', [])
        return default_rules + custom_rules
    
    def create_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: Optional[AlertSeverity] = None
    ) -> str:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            details: Additional details
            severity: Alert severity (auto-detected if not provided)
            
        Returns:
            Alert ID
        """
        # Determine severity
        if severity is None:
            severity = self._get_severity_for_type(alert_type)
        
        # Check cooldown
        if self._is_in_cooldown(alert_type):
            logger.debug(f"Alert {alert_type} is in cooldown period")
            return ""
        
        # Create alert
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            message=message,
            details=details or {},
            timestamp=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.info(f"Alert created: {alert_id} - {alert_type}: {message}")
        
        return alert_id
    
    def _get_severity_for_type(self, alert_type: str) -> AlertSeverity:
        """Get severity for alert type from rules."""
        for rule in self.alert_rules:
            if rule['type'] == alert_type:
                return rule.get('severity', AlertSeverity.INFO)
        return AlertSeverity.INFO
    
    def _is_in_cooldown(self, alert_type: str) -> bool:
        """Check if alert type is in cooldown period."""
        # Find cooldown period for this alert type
        cooldown = 60  # Default 1 minute
        for rule in self.alert_rules:
            if rule['type'] == alert_type:
                cooldown = rule.get('cooldown', 60)
                break
        
        # Check recent alerts
        cutoff_time = datetime.now() - timedelta(seconds=cooldown)
        recent_alerts = [
            a for a in self.alert_history
            if a.type == alert_type and a.timestamp > cutoff_time
        ]
        
        return len(recent_alerts) > 0
    
    async def _process_alerts(self) -> None:
        """Process and send alert notifications."""
        while True:
            try:
                for alert_id, alert in list(self.active_alerts.items()):
                    if not alert.resolved:
                        await self._send_notifications(alert)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(10)
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        # Find channels for this alert type
        channels = [AlertChannel.LOG]  # Always log
        for rule in self.alert_rules:
            if rule['type'] == alert.type:
                channels = rule.get('channels', [AlertChannel.LOG])
                break
        
        # Send to each channel
        for channel in channels:
            if channel in self.notification_channels:
                try:
                    notifier = self.notification_channels[channel]
                    await notifier.send(alert)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel}: {e}")
    
    async def _check_alert_resolution(self) -> None:
        """Check if alerts can be auto-resolved."""
        while True:
            try:
                # Auto-resolve alerts older than 1 hour with no recurrence
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                for alert_id, alert in list(self.active_alerts.items()):
                    if not alert.resolved and alert.timestamp < cutoff_time:
                        self.resolve_alert(alert_id, "Auto-resolved after 1 hour")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error checking alert resolution: {e}")
                await asyncio.sleep(300)
    
    def resolve_alert(self, alert_id: str, resolution: str = "") -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            resolution: Resolution message
            
        Returns:
            Whether alert was resolved
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.details['resolution'] = resolution
            
            logger.info(f"Alert resolved: {alert_id} - {resolution}")
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(
        self,
        limit: int = 100,
        alert_type: Optional[str] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of alerts
            alert_type: Filter by type
            severity: Filter by severity
            
        Returns:
            Alert history
        """
        alerts = self.alert_history
        
        # Apply filters
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        # Apply limit
        alerts = alerts[:limit]
        
        return [alert.to_dict() for alert in alerts]


class EmailNotifier:
    """Send email notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize email notifier."""
        self.smtp_host = config['smtp_host']
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config['username']
        self.password = config['password']
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config['to_emails']
    
    async def send(self, alert: Alert) -> None:
        """Send email notification."""
        subject = f"[{alert.severity.value.upper()}] {alert.type}: {alert.message}"
        
        body = f"""
Alert Details:
--------------
Type: {alert.type}
Severity: {alert.severity.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}

Details:
{json.dumps(alert.details, indent=2)}

--
Clinical Trial Analysis Platform Alert System
"""
        
        msg = MIMEMultipart()
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise


class SlackNotifier:
    """Send Slack notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Slack notifier."""
        self.webhook_url = config['webhook_url']
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'Alert Bot')
    
    async def send(self, alert: Alert) -> None:
        """Send Slack notification."""
        # Color based on severity
        color_map = {
            AlertSeverity.INFO: '#36a64f',
            AlertSeverity.WARNING: '#ff9900',
            AlertSeverity.ERROR: '#ff0000',
            AlertSeverity.CRITICAL: '#990000'
        }
        
        payload = {
            'channel': self.channel,
            'username': self.username,
            'attachments': [{
                'color': color_map.get(alert.severity, '#808080'),
                'title': f"{alert.severity.value.upper()}: {alert.type}",
                'text': alert.message,
                'fields': [
                    {
                        'title': 'Time',
                        'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'short': True
                    },
                    {
                        'title': 'Alert ID',
                        'value': alert.id,
                        'short': True
                    }
                ],
                'footer': 'Clinical Trial Analysis Platform'
            }]
        }
        
        # Add details if present
        if alert.details:
            for key, value in alert.details.items():
                payload['attachments'][0]['fields'].append({
                    'title': key,
                    'value': str(value),
                    'short': True
                })
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to send Slack alert: {response.status}")
                    raise Exception(f"Slack webhook failed: {response.status}")
                
                logger.info(f"Slack alert sent for {alert.id}")


class WebhookNotifier:
    """Send webhook notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize webhook notifier."""
        self.url = config['url']
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 30)
    
    async def send(self, alert: Alert) -> None:
        """Send webhook notification."""
        payload = alert.to_dict()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status not in [200, 201, 202, 204]:
                    logger.error(f"Failed to send webhook alert: {response.status}")
                    raise Exception(f"Webhook failed: {response.status}")
                
                logger.info(f"Webhook alert sent for {alert.id}")


class LogNotifier:
    """Log notifications."""
    
    async def send(self, alert: Alert) -> None:
        """Log alert."""
        log_message = (
            f"ALERT [{alert.severity.value.upper()}] "
            f"{alert.type}: {alert.message} "
            f"(ID: {alert.id})"
        )
        
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)


# Default configuration
DEFAULT_ALERT_CONFIG = {
    'email': {
        'enabled': False,
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': '',
        'password': '',
        'to_emails': []
    },
    'slack': {
        'enabled': False,
        'webhook_url': '',
        'channel': '#alerts'
    },
    'webhook': {
        'enabled': False,
        'url': '',
        'headers': {}
    },
    'rules': []
}