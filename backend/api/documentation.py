"""API documentation generation and OpenAPI schema."""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any, List
import yaml
import json
from pathlib import Path


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Generate custom OpenAPI schema with comprehensive documentation.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        OpenAPI schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Clinical Trial Statistical Analysis API",
        version="2.0.0",
        description="""
        ## Overview
        
        Comprehensive statistical analysis platform for clinical trials with:
        - Multiple statistical test types
        - Bayesian analysis
        - Adaptive trial designs
        - Real-time WebSocket updates
        - Domain-specific specializations
        
        ## Features
        
        ### Statistical Tests
        - **T-tests**: Independent, paired, Welch's
        - **ANOVA**: One-way, two-way, repeated measures
        - **Chi-square**: Independence and goodness-of-fit
        - **Correlation**: Pearson, Spearman, Kendall
        - **Regression**: Linear, logistic, Cox proportional hazards
        - **Non-parametric**: Mann-Whitney, Kruskal-Wallis, Wilcoxon
        
        ### Advanced Features
        - **Bayesian Analysis**: Conjugate priors, MCMC, Bayes factors
        - **Adaptive Designs**: Group sequential, sample size re-estimation
        - **Master Protocols**: Basket, umbrella, platform trials
        - **Simulations**: Monte Carlo power analysis
        
        ### Integration
        - **LLM Support**: OpenAI, Anthropic, Google Gemini
        - **Export Formats**: PDF, CSV, JSON, PNG, SVG
        - **Data Upload**: CSV, Excel, SAS, SPSS
        - **Real-time Updates**: WebSocket for live calculations
        
        ## Authentication
        
        Currently uses API key authentication. Include your API key in the header:
        ```
        Authorization: Bearer YOUR_API_KEY
        ```
        
        ## Rate Limiting
        
        - 100 requests per minute for standard endpoints
        - 10 requests per minute for LLM endpoints
        - WebSocket connections limited to 5 per user
        
        ## Error Codes
        
        | Code | Description |
        |------|-------------|
        | 400  | Bad Request - Invalid parameters |
        | 401  | Unauthorized - Invalid API key |
        | 403  | Forbidden - Insufficient permissions |
        | 404  | Not Found - Resource doesn't exist |
        | 429  | Too Many Requests - Rate limit exceeded |
        | 500  | Internal Server Error |
        | 503  | Service Unavailable - Temporary outage |
        """,
        routes=app.routes,
        tags=[
            {
                "name": "health",
                "description": "Health check endpoints"
            },
            {
                "name": "statistical",
                "description": "Core statistical calculations"
            },
            {
                "name": "bayesian",
                "description": "Bayesian analysis endpoints"
            },
            {
                "name": "adaptive",
                "description": "Adaptive trial design endpoints"
            },
            {
                "name": "visualization",
                "description": "Data visualization endpoints"
            },
            {
                "name": "llm",
                "description": "LLM-powered analysis"
            },
            {
                "name": "export",
                "description": "Export and reporting"
            },
            {
                "name": "upload",
                "description": "Data upload and processing"
            },
            {
                "name": "domain",
                "description": "Domain-specific analysis"
            },
            {
                "name": "websocket",
                "description": "Real-time WebSocket endpoints"
            }
        ],
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.clinicaltrials.example.com",
                "description": "Production server"
            }
        ],
        components={
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                },
                "apiKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        }
    )
    
    # Add custom examples
    add_request_examples(openapi_schema)
    
    # Add response examples
    add_response_examples(openapi_schema)
    
    # Add webhook documentation
    add_webhook_documentation(openapi_schema)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def add_request_examples(schema: Dict[str, Any]) -> None:
    """Add request examples to OpenAPI schema."""
    examples = {
        "/api/statistical/calculate": {
            "t_test": {
                "summary": "Two-sample t-test",
                "value": {
                    "test_type": "t_test",
                    "parameters": {
                        "group1_mean": 100,
                        "group1_std": 15,
                        "group1_n": 50,
                        "group2_mean": 105,
                        "group2_std": 15,
                        "group2_n": 50,
                        "alpha": 0.05
                    }
                }
            },
            "chi_square": {
                "summary": "Chi-square test",
                "value": {
                    "test_type": "chi_square",
                    "parameters": {
                        "contingency_table": [[20, 30], [25, 35]],
                        "alpha": 0.05
                    }
                }
            }
        },
        "/api/bayesian/analyze": {
            "bayesian_t_test": {
                "summary": "Bayesian t-test",
                "value": {
                    "test_type": "t_test",
                    "data": {
                        "group1": [98, 102, 101, 99, 103],
                        "group2": [105, 107, 104, 106, 108]
                    },
                    "prior": {
                        "type": "normal",
                        "params": {"mean": 0, "std": 1}
                    },
                    "rope": [-0.1, 0.1]
                }
            }
        }
    }
    
    # Add examples to paths
    if "paths" in schema:
        for path, methods in schema["paths"].items():
            if path in examples:
                for method in methods.values():
                    if isinstance(method, dict) and "requestBody" in method:
                        if "content" in method["requestBody"]:
                            for content_type in method["requestBody"]["content"].values():
                                if isinstance(content_type, dict):
                                    content_type["examples"] = examples[path]


def add_response_examples(schema: Dict[str, Any]) -> None:
    """Add response examples to OpenAPI schema."""
    response_examples = {
        "statistical_result": {
            "summary": "Statistical test result",
            "value": {
                "success": True,
                "result": {
                    "test_statistic": -2.357,
                    "p_value": 0.021,
                    "confidence_interval": [-8.9, -0.7],
                    "effect_size": 0.471,
                    "power": 0.642,
                    "interpretation": "Statistically significant difference detected"
                }
            }
        },
        "bayesian_result": {
            "summary": "Bayesian analysis result",
            "value": {
                "success": True,
                "result": {
                    "posterior_mean": 4.8,
                    "credible_interval": [2.1, 7.5],
                    "bayes_factor": 8.3,
                    "rope_probability": 0.02,
                    "interpretation": "Strong evidence for difference"
                }
            }
        }
    }
    
    # Add to components
    if "components" not in schema:
        schema["components"] = {}
    if "examples" not in schema["components"]:
        schema["components"]["examples"] = response_examples


def add_webhook_documentation(schema: Dict[str, Any]) -> None:
    """Add webhook documentation to OpenAPI schema."""
    webhooks = {
        "calculation_complete": {
            "post": {
                "summary": "Calculation completed",
                "description": "Webhook fired when a long-running calculation completes",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "calculation_id": {"type": "string"},
                                    "status": {"type": "string"},
                                    "result": {"type": "object"},
                                    "timestamp": {"type": "string", "format": "date-time"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    if "webhooks" not in schema:
        schema["webhooks"] = webhooks


def generate_api_documentation(app: FastAPI, output_dir: Path) -> None:
    """
    Generate API documentation in multiple formats.
    
    Args:
        app: FastAPI application
        output_dir: Directory to save documentation
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate OpenAPI schema
    schema = custom_openapi(app)
    
    # Save as JSON
    with open(output_dir / "openapi.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    # Save as YAML
    with open(output_dir / "openapi.yaml", "w") as f:
        yaml.dump(schema, f, default_flow_style=False)
    
    # Generate Markdown documentation
    generate_markdown_docs(schema, output_dir / "API_REFERENCE.md")
    
    # Generate Postman collection
    generate_postman_collection(schema, output_dir / "postman_collection.json")


def generate_markdown_docs(schema: Dict[str, Any], output_file: Path) -> None:
    """Generate Markdown API documentation."""
    md_content = f"""# {schema.get('info', {}).get('title', 'API Documentation')}

Version: {schema.get('info', {}).get('version', '1.0.0')}

{schema.get('info', {}).get('description', '')}

## Endpoints

"""
    
    # Document each endpoint
    for path, methods in schema.get('paths', {}).items():
        for method, details in methods.items():
            if isinstance(details, dict):
                md_content += f"### {method.upper()} {path}\n\n"
                md_content += f"{details.get('summary', '')}\n\n"
                md_content += f"{details.get('description', '')}\n\n"
                
                # Parameters
                if 'parameters' in details:
                    md_content += "**Parameters:**\n\n"
                    for param in details['parameters']:
                        required = "required" if param.get('required') else "optional"
                        md_content += f"- `{param['name']}` ({param.get('schema', {}).get('type', 'string')}, {required}): {param.get('description', '')}\n"
                    md_content += "\n"
                
                # Request body
                if 'requestBody' in details:
                    md_content += "**Request Body:**\n\n"
                    content = details['requestBody'].get('content', {})
                    for content_type, schema_info in content.items():
                        md_content += f"Content-Type: `{content_type}`\n\n"
                        if 'examples' in schema_info:
                            md_content += "Examples:\n```json\n"
                            for example in schema_info['examples'].values():
                                md_content += json.dumps(example.get('value', {}), indent=2)
                                md_content += "\n```\n\n"
                
                # Responses
                if 'responses' in details:
                    md_content += "**Responses:**\n\n"
                    for status_code, response in details['responses'].items():
                        md_content += f"- `{status_code}`: {response.get('description', '')}\n"
                    md_content += "\n"
                
                md_content += "---\n\n"
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(md_content)


def generate_postman_collection(schema: Dict[str, Any], output_file: Path) -> None:
    """Generate Postman collection from OpenAPI schema."""
    collection = {
        "info": {
            "name": schema.get('info', {}).get('title', 'API Collection'),
            "description": schema.get('info', {}).get('description', ''),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": []
    }
    
    # Convert paths to Postman items
    for path, methods in schema.get('paths', {}).items():
        for method, details in methods.items():
            if isinstance(details, dict):
                item = {
                    "name": details.get('summary', path),
                    "request": {
                        "method": method.upper(),
                        "url": {
                            "raw": "{{base_url}}" + path,
                            "host": ["{{base_url}}"],
                            "path": path.strip('/').split('/')
                        },
                        "description": details.get('description', '')
                    }
                }
                
                # Add headers
                item["request"]["header"] = [
                    {
                        "key": "Content-Type",
                        "value": "application/json"
                    },
                    {
                        "key": "Authorization",
                        "value": "Bearer {{api_key}}"
                    }
                ]
                
                # Add request body if present
                if 'requestBody' in details:
                    content = details['requestBody'].get('content', {})
                    if 'application/json' in content:
                        schema_info = content['application/json']
                        if 'examples' in schema_info:
                            # Use first example
                            example = list(schema_info['examples'].values())[0]
                            item["request"]["body"] = {
                                "mode": "raw",
                                "raw": json.dumps(example.get('value', {}), indent=2),
                                "options": {
                                    "raw": {
                                        "language": "json"
                                    }
                                }
                            }
                
                collection["item"].append(item)
    
    # Add variables
    collection["variable"] = [
        {
            "key": "base_url",
            "value": "http://localhost:8000",
            "type": "string"
        },
        {
            "key": "api_key",
            "value": "",
            "type": "string"
        }
    ]
    
    # Write collection
    with open(output_file, "w") as f:
        json.dump(collection, f, indent=2)


def generate_sdk_examples() -> Dict[str, str]:
    """Generate SDK usage examples for different languages."""
    examples = {
        "python": """
import requests

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "your_api_key"

# Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Example: T-test calculation
data = {
    "test_type": "t_test",
    "parameters": {
        "group1_mean": 100,
        "group1_std": 15,
        "group1_n": 50,
        "group2_mean": 105,
        "group2_std": 15,
        "group2_n": 50,
        "alpha": 0.05
    }
}

response = requests.post(
    f"{API_URL}/api/statistical/calculate",
    headers=headers,
    json=data
)

result = response.json()
print(f"P-value: {result['result']['p_value']}")
print(f"Effect size: {result['result']['effect_size']}")
""",
        "javascript": """
const API_URL = 'http://localhost:8000';
const API_KEY = 'your_api_key';

// Example: T-test calculation
async function calculateTTest() {
    const data = {
        test_type: 't_test',
        parameters: {
            group1_mean: 100,
            group1_std: 15,
            group1_n: 50,
            group2_mean: 105,
            group2_std: 15,
            group2_n: 50,
            alpha: 0.05
        }
    };
    
    const response = await fetch(`${API_URL}/api/statistical/calculate`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${API_KEY}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
    
    const result = await response.json();
    console.log(`P-value: ${result.result.p_value}`);
    console.log(`Effect size: ${result.result.effect_size}`);
}

calculateTTest();
""",
        "r": """
library(httr)
library(jsonlite)

# Configuration
API_URL <- "http://localhost:8000"
API_KEY <- "your_api_key"

# Example: T-test calculation
data <- list(
    test_type = "t_test",
    parameters = list(
        group1_mean = 100,
        group1_std = 15,
        group1_n = 50,
        group2_mean = 105,
        group2_std = 15,
        group2_n = 50,
        alpha = 0.05
    )
)

response <- POST(
    paste0(API_URL, "/api/statistical/calculate"),
    add_headers(
        Authorization = paste("Bearer", API_KEY),
        "Content-Type" = "application/json"
    ),
    body = toJSON(data, auto_unbox = TRUE)
)

result <- content(response, "parsed")
cat("P-value:", result$result$p_value, "\n")
cat("Effect size:", result$result$effect_size, "\n")
"""
    }
    
    return examples