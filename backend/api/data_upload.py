"""File upload and data integration functionality."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
import chardet
from scipy import stats

router = APIRouter(prefix="/api/data", tags=["data"])


class DataProcessor:
    """Process uploaded data files for analysis."""
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.tsv', '.json']
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    @staticmethod
    def detect_encoding(file_bytes: bytes) -> str:
        """Detect file encoding."""
        result = chardet.detect(file_bytes[:10000])  # Check first 10KB
        return result['encoding'] or 'utf-8'
    
    @staticmethod
    def validate_file(file: UploadFile) -> None:
        """
        Validate uploaded file.
        
        Args:
            file: Uploaded file
            
        Raises:
            HTTPException: If file is invalid
        """
        # Check file extension
        file_ext = '.' + file.filename.split('.')[-1].lower()
        if file_ext not in DataProcessor.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {', '.join(DataProcessor.SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > DataProcessor.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {DataProcessor.MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
            )
    
    @staticmethod
    async def read_file(file: UploadFile) -> pd.DataFrame:
        """
        Read uploaded file into DataFrame.
        
        Args:
            file: Uploaded file
            
        Returns:
            DataFrame with file contents
        """
        DataProcessor.validate_file(file)
        
        contents = await file.read()
        file_ext = '.' + file.filename.split('.')[-1].lower()
        
        try:
            if file_ext == '.csv':
                encoding = DataProcessor.detect_encoding(contents)
                df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
            
            elif file_ext == '.tsv':
                encoding = DataProcessor.detect_encoding(contents)
                df = pd.read_csv(io.BytesIO(contents), sep='\t', encoding=encoding)
            
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(io.BytesIO(contents))
            
            elif file_ext == '.json':
                encoding = DataProcessor.detect_encoding(contents)
                data = json.loads(contents.decode(encoding))
                df = pd.DataFrame(data)
            
            else:
                raise ValueError(f"Unsupported format: {file_ext}")
            
            return df
        
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(e)}"
            )
    
    @staticmethod
    def infer_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer statistical data types for columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of column names to data types
        """
        data_types = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 10:
                    data_types[col] = 'categorical'
                else:
                    data_types[col] = 'continuous'
            
            elif df[col].dtype == 'bool':
                data_types[col] = 'binary'
            
            elif df[col].dtype == 'object':
                # Check if it can be converted to numeric
                try:
                    pd.to_numeric(df[col])
                    data_types[col] = 'continuous'
                except:
                    # Check if binary
                    if df[col].nunique() == 2:
                        data_types[col] = 'binary'
                    else:
                        data_types[col] = 'categorical'
            
            else:
                data_types[col] = 'unknown'
        
        return data_types
    
    @staticmethod
    def calculate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of summary statistics
        """
        summary = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': {}
        }
        
        data_types = DataProcessor.infer_data_types(df)
        
        for col in df.columns:
            col_stats = {
                'data_type': data_types[col],
                'n_missing': df[col].isna().sum(),
                'missing_percentage': df[col].isna().sum() / len(df) * 100
            }
            
            if data_types[col] == 'continuous':
                col_stats.update({
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75)),
                    'skewness': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis())
                })
            
            elif data_types[col] in ['categorical', 'binary']:
                value_counts = df[col].value_counts()
                col_stats.update({
                    'unique_values': int(df[col].nunique()),
                    'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'value_counts': value_counts.head(10).to_dict()
                })
            
            summary['columns'][col] = col_stats
        
        return summary
    
    @staticmethod
    def prepare_for_analysis(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        group_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare data for statistical analysis.
        
        Args:
            df: DataFrame to prepare
            target_column: Target/outcome variable
            feature_columns: Feature/predictor variables
            group_column: Grouping variable for comparisons
            
        Returns:
            Prepared data dictionary
        """
        prepared = {
            'n_samples': len(df),
            'data': {}
        }
        
        # Handle missing values
        df = df.dropna(subset=[target_column] if target_column else df.columns)
        
        if target_column:
            prepared['target'] = {
                'name': target_column,
                'values': df[target_column].tolist(),
                'type': DataProcessor.infer_data_types(df[[target_column]])[target_column]
            }
        
        if feature_columns:
            prepared['features'] = {}
            for col in feature_columns:
                if col in df.columns:
                    prepared['features'][col] = {
                        'values': df[col].tolist(),
                        'type': DataProcessor.infer_data_types(df[[col]])[col]
                    }
        
        if group_column and group_column in df.columns:
            prepared['groups'] = {}
            for group_name, group_df in df.groupby(group_column):
                prepared['groups'][str(group_name)] = {
                    'n': len(group_df),
                    'target_values': group_df[target_column].tolist() if target_column else None
                }
        
        return prepared


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    analysis_type: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Upload data file for analysis.
    
    Args:
        file: File to upload
        analysis_type: Type of analysis to perform
        
    Returns:
        File summary and initial analysis
    """
    try:
        # Read file
        df = await DataProcessor.read_file(file)
        
        # Calculate summary statistics
        summary = DataProcessor.calculate_summary_statistics(df)
        
        # Infer data types
        data_types = DataProcessor.infer_data_types(df)
        
        # Store in session (in production, use proper storage)
        file_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "summary": summary,
            "data_types": data_types,
            "suggested_analyses": suggest_analyses(data_types, summary)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{file_id}")
async def analyze_uploaded_data(
    file_id: str,
    analysis_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform statistical analysis on uploaded data.
    
    Args:
        file_id: ID of uploaded file
        analysis_config: Analysis configuration
        
    Returns:
        Analysis results
    """
    try:
        # In production, retrieve from proper storage
        # For now, return mock results
        
        analysis_type = analysis_config.get('type', 'descriptive')
        
        if analysis_type == 'descriptive':
            return perform_descriptive_analysis(analysis_config)
        
        elif analysis_type == 't_test':
            return perform_t_test_analysis(analysis_config)
        
        elif analysis_type == 'anova':
            return perform_anova_analysis(analysis_config)
        
        elif analysis_type == 'correlation':
            return perform_correlation_analysis(analysis_config)
        
        elif analysis_type == 'regression':
            return perform_regression_analysis(analysis_config)
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown analysis type: {analysis_type}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def suggest_analyses(data_types: Dict[str, str], summary: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Suggest appropriate analyses based on data characteristics.
    
    Args:
        data_types: Column data types
        summary: Data summary statistics
        
    Returns:
        List of suggested analyses
    """
    suggestions = []
    
    # Count variable types
    continuous_vars = [col for col, dtype in data_types.items() if dtype == 'continuous']
    categorical_vars = [col for col, dtype in data_types.items() if dtype == 'categorical']
    binary_vars = [col for col, dtype in data_types.items() if dtype == 'binary']
    
    # Descriptive statistics (always suggest)
    suggestions.append({
        'type': 'descriptive',
        'name': 'Descriptive Statistics',
        'description': 'Summary statistics for all variables'
    })
    
    # T-test suggestions
    if len(continuous_vars) >= 1 and (len(binary_vars) >= 1 or len(categorical_vars) >= 1):
        suggestions.append({
            'type': 't_test',
            'name': 'Two-Sample t-Test',
            'description': f'Compare {continuous_vars[0]} between groups'
        })
    
    # ANOVA suggestions
    if len(continuous_vars) >= 1 and len(categorical_vars) >= 1:
        cat_var = categorical_vars[0]
        n_groups = summary['columns'][cat_var].get('unique_values', 0)
        if n_groups > 2:
            suggestions.append({
                'type': 'anova',
                'name': 'One-Way ANOVA',
                'description': f'Compare {continuous_vars[0]} across {n_groups} groups'
            })
    
    # Correlation suggestions
    if len(continuous_vars) >= 2:
        suggestions.append({
            'type': 'correlation',
            'name': 'Correlation Analysis',
            'description': f'Analyze relationships between continuous variables'
        })
    
    # Regression suggestions
    if len(continuous_vars) >= 2:
        suggestions.append({
            'type': 'regression',
            'name': 'Linear Regression',
            'description': f'Predict {continuous_vars[0]} from other variables'
        })
    
    if len(binary_vars) >= 1 and len(continuous_vars) >= 1:
        suggestions.append({
            'type': 'logistic_regression',
            'name': 'Logistic Regression',
            'description': f'Predict {binary_vars[0]} from other variables'
        })
    
    # Chi-square suggestions
    if len(categorical_vars) >= 2 or len(binary_vars) >= 2:
        suggestions.append({
            'type': 'chi_square',
            'name': 'Chi-Square Test',
            'description': 'Test association between categorical variables'
        })
    
    return suggestions


def perform_descriptive_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform descriptive statistical analysis."""
    # Mock implementation - in production, use actual data
    return {
        'analysis_type': 'descriptive',
        'results': {
            'summary_statistics': {
                'mean': 50.5,
                'median': 51.0,
                'std': 10.2,
                'min': 20.0,
                'max': 80.0
            },
            'distribution': {
                'normality_test': {
                    'statistic': 0.98,
                    'p_value': 0.234,
                    'is_normal': True
                }
            }
        }
    }


def perform_t_test_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform t-test analysis."""
    # Mock implementation
    return {
        'analysis_type': 't_test',
        'results': {
            't_statistic': 2.45,
            'p_value': 0.018,
            'degrees_of_freedom': 98,
            'confidence_interval': [-5.2, -0.8],
            'effect_size': 0.49,
            'interpretation': 'Significant difference between groups'
        }
    }


def perform_anova_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform ANOVA analysis."""
    # Mock implementation
    return {
        'analysis_type': 'anova',
        'results': {
            'f_statistic': 4.32,
            'p_value': 0.012,
            'degrees_of_freedom': [2, 97],
            'eta_squared': 0.082,
            'interpretation': 'Significant differences among groups'
        }
    }


def perform_correlation_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform correlation analysis."""
    # Mock implementation
    return {
        'analysis_type': 'correlation',
        'results': {
            'correlation_matrix': {
                'var1': {'var1': 1.0, 'var2': 0.65, 'var3': -0.32},
                'var2': {'var1': 0.65, 'var2': 1.0, 'var3': -0.18},
                'var3': {'var1': -0.32, 'var2': -0.18, 'var3': 1.0}
            },
            'significant_correlations': [
                {'variables': ['var1', 'var2'], 'r': 0.65, 'p_value': 0.001}
            ]
        }
    }


def perform_regression_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform regression analysis."""
    # Mock implementation
    return {
        'analysis_type': 'regression',
        'results': {
            'coefficients': {
                'intercept': 10.5,
                'var1': 2.3,
                'var2': -1.8
            },
            'r_squared': 0.72,
            'adjusted_r_squared': 0.71,
            'f_statistic': 45.6,
            'p_value': 0.0001,
            'interpretation': 'Model explains 72% of variance'
        }
    }


@router.get("/templates")
async def get_data_templates() -> Dict[str, Any]:
    """
    Get example data templates for different analysis types.
    
    Returns:
        Dictionary of template descriptions and download links
    """
    return {
        'templates': [
            {
                'name': 'Two-Sample t-Test',
                'description': 'Template for comparing two groups',
                'columns': ['group', 'outcome'],
                'example_data': {
                    'group': ['A', 'A', 'B', 'B'],
                    'outcome': [23.5, 24.1, 28.3, 27.9]
                }
            },
            {
                'name': 'ANOVA',
                'description': 'Template for comparing multiple groups',
                'columns': ['group', 'outcome'],
                'example_data': {
                    'group': ['A', 'A', 'B', 'B', 'C', 'C'],
                    'outcome': [23.5, 24.1, 28.3, 27.9, 31.2, 30.8]
                }
            },
            {
                'name': 'Regression',
                'description': 'Template for regression analysis',
                'columns': ['predictor1', 'predictor2', 'outcome'],
                'example_data': {
                    'predictor1': [1.2, 2.3, 3.1, 4.5],
                    'predictor2': [5.6, 6.7, 7.8, 8.9],
                    'outcome': [10.1, 15.2, 18.3, 23.4]
                }
            },
            {
                'name': 'Correlation',
                'description': 'Template for correlation analysis',
                'columns': ['variable1', 'variable2', 'variable3'],
                'example_data': {
                    'variable1': [1.2, 2.3, 3.1, 4.5],
                    'variable2': [5.6, 6.7, 7.8, 8.9],
                    'variable3': [10.1, 11.2, 12.3, 13.4]
                }
            }
        ]
    }