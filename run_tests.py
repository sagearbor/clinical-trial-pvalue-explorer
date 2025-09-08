#!/usr/bin/env python
"""
Comprehensive test runner for Clinical Trial P-Value Explorer.
Runs all test suites and generates coverage report.
"""

import sys
import subprocess
from pathlib import Path
import argparse

def run_tests(test_type="all", verbose=False, coverage=True):
    """
    Run specified test suite.
    
    Args:
        test_type: Type of tests to run (all, unit, integration, statistical)
        verbose: Enable verbose output
        coverage: Generate coverage report
    """
    base_cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        base_cmd.append("-v")
    else:
        base_cmd.append("-q")
    
    # Add coverage
    if coverage:
        base_cmd.extend(["--cov=backend", "--cov=frontend", "--cov-report=term", "--cov-report=html"])
    
    # Select test files
    test_files = {
        "all": ["tests/"],
        "unit": [
            "tests/test_visualizations.py",
            "tests/test_nonparametric.py",
            "tests/test_bayesian.py"
        ],
        "integration": [
            "tests/test_*integration*.py",
            "tests/test_*api*.py"
        ],
        "statistical": [
            "tests/test_*statistical*.py",
            "tests/test_*validation*.py"
        ],
        "visualization": ["tests/test_visualizations.py"],
        "bayesian": ["tests/test_bayesian.py"],
        "nonparametric": ["tests/test_nonparametric.py"]
    }
    
    if test_type not in test_files:
        print(f"Unknown test type: {test_type}")
        print(f"Available types: {', '.join(test_files.keys())}")
        return 1
    
    # Add test files
    base_cmd.extend(test_files[test_type])
    
    # Run tests
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(base_cmd)}")
    print("-" * 50)
    
    result = subprocess.run(base_cmd, capture_output=False)
    
    if coverage and result.returncode == 0:
        print("\n" + "=" * 50)
        print("Coverage report generated in htmlcov/index.html")
    
    return result.returncode

def run_linters(fix=False):
    """
    Run code quality checks.
    
    Args:
        fix: Auto-fix issues where possible
    """
    print("Running code quality checks...")
    print("-" * 50)
    
    # Black formatting
    print("\n📝 Checking code formatting with black...")
    black_cmd = ["black", "backend/", "frontend/", "--line-length=120"]
    if not fix:
        black_cmd.append("--check")
    subprocess.run(black_cmd)
    
    # Flake8 linting
    print("\n🔍 Linting with flake8...")
    subprocess.run([
        "flake8", "backend/", "frontend/",
        "--max-line-length=120",
        "--ignore=E203,W503",
        "--statistics"
    ])
    
    # Type checking with mypy
    print("\n🔎 Type checking with mypy...")
    subprocess.run([
        "mypy", "backend/",
        "--ignore-missing-imports",
        "--no-strict-optional"
    ])
    
    print("\n" + "=" * 50)
    print("Code quality checks complete!")

def run_security_scan():
    """Run security vulnerability scan."""
    print("Running security scan...")
    print("-" * 50)
    
    # Bandit security scan
    print("\n🔒 Security scan with bandit...")
    subprocess.run([
        "bandit", "-r", "backend/",
        "-ll", "--skip=B101,B601"
    ])
    
    # Safety check for dependencies
    print("\n📦 Checking dependencies with safety...")
    subprocess.run(["safety", "check", "--json"])
    
    print("\n" + "=" * 50)
    print("Security scan complete!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for Clinical Trial P-Value Explorer"
    )
    parser.add_argument(
        "command",
        choices=["test", "lint", "security", "all"],
        help="Command to run"
    )
    parser.add_argument(
        "--type",
        default="all",
        help="Type of tests to run (for test command)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage report"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix linting issues"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    if args.command == "test":
        return run_tests(
            test_type=args.type,
            verbose=args.verbose,
            coverage=not args.no_coverage
        )
    elif args.command == "lint":
        return run_linters(fix=args.fix)
    elif args.command == "security":
        return run_security_scan()
    elif args.command == "all":
        # Run everything
        print("🚀 Running complete test suite...\n")
        
        # Tests
        if run_tests(verbose=args.verbose) != 0:
            print("❌ Tests failed!")
            return 1
        
        # Linters
        run_linters()
        
        # Security
        run_security_scan()
        
        print("\n" + "=" * 50)
        print("✅ All checks passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())