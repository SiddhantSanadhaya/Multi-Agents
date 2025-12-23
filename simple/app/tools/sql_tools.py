# app/tools/sql_tools.py

from langchain_core.tools import tool
import json
import re
from typing import Dict, Any

# Mock database schema and data for demonstration
MOCK_SCHEMA = {
    "suppliers": {
        "columns": ["id", "name", "category", "location", "rating", "created_date"],
        "description": "Supplier master data"
    },
    "purchase_orders": {
        "columns": ["po_id", "supplier_id", "amount", "category", "order_date", "status"],
        "description": "Purchase order transactions"
    },
    "spend_analytics": {
        "columns": ["supplier_id", "category", "month", "year", "spend_amount", "volume"],
        "description": "Monthly spend aggregations"
    }
}

MOCK_QUERY_EXAMPLES = [
    {
        "query": "top suppliers by spend",
        "sql": "SELECT s.name, SUM(sa.spend_amount) as total_spend FROM suppliers s JOIN spend_analytics sa ON s.id = sa.supplier_id WHERE sa.year = 2024 GROUP BY s.id, s.name ORDER BY total_spend DESC LIMIT 10",
        "description": "Get top 10 suppliers by YTD spend"
    },
    {
        "query": "bearing spend analysis",
        "sql": "SELECT s.name, SUM(sa.spend_amount) as bearing_spend FROM suppliers s JOIN spend_analytics sa ON s.id = sa.supplier_id WHERE s.category = 'bearings' AND sa.year = 2024 GROUP BY s.id, s.name ORDER BY bearing_spend DESC",
        "description": "Analyze spending on bearing suppliers"
    },
    {
        "query": "monthly spend trends",
        "sql": "SELECT month, year, SUM(spend_amount) as monthly_spend FROM spend_analytics WHERE year >= 2023 GROUP BY year, month ORDER BY year, month",
        "description": "Get monthly spend trends over time"
    }
]

# Mock spend data for query results
MOCK_SPEND_DATA = [
    {"supplier_name": "SupplierX Industries", "total_spend": 2500000, "category": "bearings"},
    {"supplier_name": "Global Parts Inc", "total_spend": 3200000, "category": "electronics"},
    {"supplier_name": "SupplierY Corp", "total_spend": 1800000, "category": "bearings"},
    {"supplier_name": "SupplierZ Ltd", "total_spend": 950000, "category": "bearings"}
]

@tool
def search_similar_queries(query_description: str) -> str:
    """
    Search for similar SQL query examples in the knowledge base.
    
    Args:
        query_description: Description of what you want to query
        
    Returns:
        JSON string with similar query examples
    """
    query_lower = query_description.lower()
    
    # Simple keyword matching for demonstration
    matching_queries = []
    for example in MOCK_QUERY_EXAMPLES:
        if any(keyword in query_lower for keyword in example["query"].split()):
            matching_queries.append(example)
    
    return json.dumps({
        "search_query": query_description,
        "matching_queries": matching_queries,
        "total_matches": len(matching_queries)
    }, indent=2)


@tool
def get_database_schema(table_name: str = None) -> str:
    """
    Get database schema information.
    
    Args:
        table_name: Specific table name, or None for all tables
        
    Returns:
        JSON string with schema information
    """
    if table_name:
        if table_name in MOCK_SCHEMA:
            return json.dumps({
                "table": table_name,
                "schema": MOCK_SCHEMA[table_name]
            }, indent=2)
        else:
            return json.dumps({
                "error": f"Table '{table_name}' not found",
                "available_tables": list(MOCK_SCHEMA.keys())
            })
    
    return json.dumps({
        "database_schema": MOCK_SCHEMA,
        "total_tables": len(MOCK_SCHEMA)
    }, indent=2)


@tool
def validate_sql_query(sql_query: str) -> str:
    """
    Validate SQL query for safety and syntax.
    
    Args:
        sql_query: SQL query to validate
        
    Returns:
        JSON string with validation results
    """
    validation_result = {
        "query": sql_query,
        "is_valid": True,
        "is_safe": True,
        "warnings": [],
        "errors": []
    }
    
    # Basic safety checks
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    query_upper = sql_query.upper()
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            validation_result["is_safe"] = False
            validation_result["errors"].append(f"Unsafe keyword detected: {keyword}")
    
    # Check if it's a SELECT query
    if not query_upper.strip().startswith("SELECT"):
        validation_result["is_safe"] = False
        validation_result["errors"].append("Only SELECT queries are allowed")
    
    # Basic syntax validation
    if not re.search(r'\bFROM\b', query_upper):
        validation_result["is_valid"] = False
        validation_result["errors"].append("Missing FROM clause")
    
    # Check for LIMIT clause (recommended for large results)
    if not re.search(r'\bLIMIT\b', query_upper):
        validation_result["warnings"].append("Consider adding LIMIT clause to prevent large result sets")
    
    return json.dumps(validation_result, indent=2)


@tool
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a validated SQL query (SELECT only).
    
    Args:
        sql_query: SQL query to execute
        
    Returns:
        JSON string with query results
    """
    # First validate the query
    validation = json.loads(validate_sql_query(sql_query))
    
    if not validation["is_safe"]:
        return json.dumps({
            "error": "Query failed safety validation",
            "validation_errors": validation["errors"]
        })
    
    # Mock query execution based on content
    query_upper = sql_query.upper()
    
    if "TOP" in query_upper and "SPEND" in query_upper:
        # Top suppliers by spend query
        results = sorted(MOCK_SPEND_DATA, key=lambda x: x["total_spend"], reverse=True)
        
        return json.dumps({
            "query": sql_query,
            "results": results[:10],  # Top 10
            "row_count": len(results),
            "execution_time_ms": 245
        }, indent=2)
        
    elif "BEARING" in query_upper:
        # Bearing-specific query
        bearing_results = [row for row in MOCK_SPEND_DATA if row["category"] == "bearings"]
        
        return json.dumps({
            "query": sql_query,
            "results": bearing_results,
            "row_count": len(bearing_results),
            "execution_time_ms": 156
        }, indent=2)
        
    elif "TREND" in query_upper or "MONTH" in query_upper:
        # Monthly trend query
        monthly_trends = [
            {"month": 1, "year": 2024, "monthly_spend": 820000},
            {"month": 2, "year": 2024, "monthly_spend": 750000},
            {"month": 3, "year": 2024, "monthly_spend": 890000},
            {"month": 4, "year": 2024, "monthly_spend": 920000},
            {"month": 5, "year": 2024, "monthly_spend": 980000},
            {"month": 6, "year": 2024, "monthly_spend": 1100000}
        ]
        
        return json.dumps({
            "query": sql_query,
            "results": monthly_trends,
            "row_count": len(monthly_trends),
            "execution_time_ms": 189
        }, indent=2)
    
    else:
        # Generic response
        return json.dumps({
            "query": sql_query,
            "results": MOCK_SPEND_DATA,
            "row_count": len(MOCK_SPEND_DATA),
            "execution_time_ms": 123,
            "note": "Mock data returned - adjust query for specific results"
        }, indent=2)