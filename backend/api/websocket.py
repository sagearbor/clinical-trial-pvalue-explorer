"""WebSocket support for real-time parameter updates."""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Optional
import json
import asyncio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalculationRequest:
    """WebSocket calculation request."""
    request_id: str
    test_type: str
    parameters: Dict[str, Any]
    visualization_type: Optional[str] = None


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: List[WebSocket] = []
        self.calculation_cache: Dict[str, Any] = {}
        self.pending_calculations: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "WebSocket connection established"
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client."""
        await websocket.send_text(message)
    
    async def send_personal_json(self, data: Dict[str, Any], websocket: WebSocket):
        """Send JSON data to specific client."""
        await websocket.send_json(data)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON data to all connected clients."""
        message = json.dumps(data)
        await self.broadcast(message)
    
    async def handle_calculation_request(
        self,
        websocket: WebSocket,
        request: CalculationRequest
    ):
        """Handle real-time calculation request."""
        # Check cache first
        cache_key = f"{request.test_type}_{json.dumps(request.parameters, sort_keys=True)}"
        
        if cache_key in self.calculation_cache:
            # Send cached result immediately
            await self.send_personal_json({
                "type": "calculation_result",
                "request_id": request.request_id,
                "cached": True,
                "result": self.calculation_cache[cache_key]
            }, websocket)
            return
        
        # Cancel any pending calculation for this client
        if request.request_id in self.pending_calculations:
            self.pending_calculations[request.request_id].cancel()
        
        # Start new calculation
        task = asyncio.create_task(
            self._perform_calculation(websocket, request, cache_key)
        )
        self.pending_calculations[request.request_id] = task
    
    async def _perform_calculation(
        self,
        websocket: WebSocket,
        request: CalculationRequest,
        cache_key: str
    ):
        """Perform the actual calculation asynchronously."""
        try:
            # Send progress update
            await self.send_personal_json({
                "type": "calculation_progress",
                "request_id": request.request_id,
                "status": "calculating",
                "progress": 0
            }, websocket)
            
            # Import calculation functions
            from backend.statistical.frequentist import StatisticalTestFactory
            from backend.statistical.visualizations import TrialVisualizer
            
            factory = StatisticalTestFactory()
            visualizer = TrialVisualizer()
            
            result = {}
            
            # Perform statistical test
            if request.test_type in factory.available_tests:
                test = factory.get_test(request.test_type)
                
                # Send progress update
                await self.send_personal_json({
                    "type": "calculation_progress",
                    "request_id": request.request_id,
                    "status": "calculating",
                    "progress": 30
                }, websocket)
                
                # Calculate p-value and power
                p_value, p_error = test.calculate_p_value(**request.parameters)
                power, power_error = test.calculate_power(**request.parameters)
                
                result['p_value'] = p_value
                result['power'] = power
                result['errors'] = {
                    'p_value': p_error,
                    'power': power_error
                }
                
                # Send progress update
                await self.send_personal_json({
                    "type": "calculation_progress",
                    "request_id": request.request_id,
                    "status": "calculating",
                    "progress": 60
                }, websocket)
            
            # Generate visualization if requested
            if request.visualization_type:
                if request.visualization_type == "power_curve":
                    n_range = list(range(10, 501, 10))
                    effect_sizes = [
                        request.parameters.get('effect_size', 0.5) * f
                        for f in [0.5, 1.0, 1.5]
                    ]
                    
                    plot_json = visualizer.create_power_curve(
                        n_range=n_range,
                        effect_sizes=effect_sizes,
                        alpha=request.parameters.get('alpha', 0.05),
                        test_type=request.test_type
                    )
                    result['visualization'] = json.loads(plot_json)
                
                # Send progress update
                await self.send_personal_json({
                    "type": "calculation_progress",
                    "request_id": request.request_id,
                    "status": "calculating",
                    "progress": 90
                }, websocket)
            
            # Cache result
            self.calculation_cache[cache_key] = result
            
            # Send final result
            await self.send_personal_json({
                "type": "calculation_result",
                "request_id": request.request_id,
                "cached": False,
                "result": result
            }, websocket)
            
        except asyncio.CancelledError:
            logger.info(f"Calculation {request.request_id} cancelled")
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            await self.send_personal_json({
                "type": "calculation_error",
                "request_id": request.request_id,
                "error": str(e)
            }, websocket)
        finally:
            # Clean up
            if request.request_id in self.pending_calculations:
                del self.pending_calculations[request.request_id]
    
    async def handle_parameter_update(
        self,
        websocket: WebSocket,
        parameter: str,
        value: Any,
        test_type: str
    ):
        """Handle real-time parameter update."""
        # Validate parameter
        from backend.statistical.frequentist import StatisticalTestFactory
        
        factory = StatisticalTestFactory()
        if test_type not in factory.available_tests:
            await self.send_personal_json({
                "type": "parameter_error",
                "parameter": parameter,
                "error": f"Unknown test type: {test_type}"
            }, websocket)
            return
        
        test = factory.get_test(test_type)
        required_params = test.get_required_params()
        
        # Send acknowledgment
        await self.send_personal_json({
            "type": "parameter_update",
            "parameter": parameter,
            "value": value,
            "status": "accepted"
        }, websocket)
        
        # Trigger recalculation if all required parameters are present
        # This would be handled by the client sending a calculation request


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "calculation":
                # Handle calculation request
                request = CalculationRequest(
                    request_id=data.get("request_id"),
                    test_type=data.get("test_type"),
                    parameters=data.get("parameters", {}),
                    visualization_type=data.get("visualization_type")
                )
                await manager.handle_calculation_request(websocket, request)
            
            elif message_type == "parameter_update":
                # Handle parameter update
                await manager.handle_parameter_update(
                    websocket,
                    parameter=data.get("parameter"),
                    value=data.get("value"),
                    test_type=data.get("test_type")
                )
            
            elif message_type == "ping":
                # Respond to ping
                await manager.send_personal_json({
                    "type": "pong",
                    "timestamp": data.get("timestamp")
                }, websocket)
            
            elif message_type == "subscribe":
                # Subscribe to specific updates
                topics = data.get("topics", [])
                await manager.send_personal_json({
                    "type": "subscription",
                    "topics": topics,
                    "status": "subscribed"
                }, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)