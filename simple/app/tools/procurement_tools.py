# app/tools/procurement_tools.py

from langchain_core.tools import tool
from typing import List, Dict, Any
import json

# Mock data for demonstration - in production, these would connect to real databases/APIs
MOCK_SUPPLIERS = [
    {
        "id": "SUP001",
        "name": "SupplierX Industries",
        "category": "bearings",
        "location": "Michigan, USA",
        "rating": 4.5,
        "ytd_spend": 2500000,
        "kraljic_position": "strategic"
    },
    {
        "id": "SUP002", 
        "name": "SupplierY Corp",
        "category": "bearings",
        "location": "Ohio, USA", 
        "rating": 4.2,
        "ytd_spend": 1800000,
        "kraljic_position": "leverage"
    },
    {
        "id": "SUP003",
        "name": "SupplierZ Ltd",
        "category": "bearings",
        "location": "California, USA",
        "rating": 4.0,
        "ytd_spend": 950000,
        "kraljic_position": "routine"
    },
    {
        "id": "SUP004",
        "name": "Global Parts Inc",
        "category": "electronics",
        "location": "Texas, USA",
        "rating": 4.3,
        "ytd_spend": 3200000,
        "kraljic_position": "strategic"
    }
]

@tool
def query_suppliers(category: str = None, location: str = None, min_rating: float = 0.0) -> str:
    """
    Query suppliers based on category, location, and minimum rating.
    
    Args:
        category: Product category (e.g., "bearings", "electronics")
        location: Location filter (e.g., "USA", "Michigan")
        min_rating: Minimum supplier rating (0.0 to 5.0)
    
    Returns:
        JSON string with supplier information
    """
    filtered_suppliers = MOCK_SUPPLIERS.copy()
    
    if category:
        filtered_suppliers = [s for s in filtered_suppliers if s["category"].lower() == category.lower()]
    
    if location:
        filtered_suppliers = [s for s in filtered_suppliers if location.lower() in s["location"].lower()]
    
    if min_rating:
        filtered_suppliers = [s for s in filtered_suppliers if s["rating"] >= min_rating]
    
    return json.dumps({
        "suppliers": filtered_suppliers,
        "count": len(filtered_suppliers),
        "filters_applied": {
            "category": category,
            "location": location,
            "min_rating": min_rating
        }
    }, indent=2)


@tool
def get_kraljic_positioning(supplier_ids: List[str] = None) -> str:
    """
    Get Kraljic matrix positioning for suppliers.
    
    Args:
        supplier_ids: List of supplier IDs to analyze
    
    Returns:
        JSON string with Kraljic positioning analysis
    """
    if not supplier_ids:
        # Return all suppliers' positioning
        positioning = {s["id"]: s["kraljic_position"] for s in MOCK_SUPPLIERS}
    else:
        positioning = {}
        for supplier in MOCK_SUPPLIERS:
            if supplier["id"] in supplier_ids:
                positioning[supplier["id"]] = supplier["kraljic_position"]
    
    # Add strategic recommendations
    recommendations = {}
    for supplier_id, position in positioning.items():
        if position == "strategic":
            recommendations[supplier_id] = "Focus on partnership and long-term contracts"
        elif position == "leverage":
            recommendations[supplier_id] = "Aggressive price negotiation, potential 10-15% savings"
        elif position == "routine":
            recommendations[supplier_id] = "Standardize and automate purchasing"
        else:
            recommendations[supplier_id] = "Monitor closely, develop alternatives"
    
    return json.dumps({
        "kraljic_positioning": positioning,
        "strategic_recommendations": recommendations
    }, indent=2)


@tool
def get_ytd_spend(supplier_ids: List[str] = None) -> str:
    """
    Get year-to-date spend by supplier.
    
    Args:
        supplier_ids: List of supplier IDs to query
    
    Returns:
        JSON string with YTD spend data
    """
    if not supplier_ids:
        # Return all suppliers' spend
        spend_data = {s["id"]: {"name": s["name"], "ytd_spend": s["ytd_spend"]} for s in MOCK_SUPPLIERS}
    else:
        spend_data = {}
        for supplier in MOCK_SUPPLIERS:
            if supplier["id"] in supplier_ids:
                spend_data[supplier["id"]] = {"name": supplier["name"], "ytd_spend": supplier["ytd_spend"]}
    
    # Calculate totals and percentages
    total_spend = sum(data["ytd_spend"] for data in spend_data.values())
    
    for supplier_id, data in spend_data.items():
        data["percentage_of_total"] = round((data["ytd_spend"] / total_spend) * 100, 2) if total_spend > 0 else 0
    
    return json.dumps({
        "ytd_spend_data": spend_data,
        "total_spend": total_spend,
        "currency": "USD"
    }, indent=2)


@tool 
def calculate_negotiation_strategy(supplier_id: str, spend_data: Dict = None, kraljic_position: str = None) -> str:
    """
    Calculate negotiation strategy based on supplier analysis.
    
    Args:
        supplier_id: Supplier ID to analyze
        spend_data: Optional spend data context
        kraljic_position: Optional Kraljic position
    
    Returns:
        JSON string with negotiation strategy
    """
    # Find supplier info
    supplier_info = None
    for supplier in MOCK_SUPPLIERS:
        if supplier["id"] == supplier_id:
            supplier_info = supplier
            break
    
    if not supplier_info:
        return json.dumps({"error": f"Supplier {supplier_id} not found"})
    
    position = kraljic_position or supplier_info["kraljic_position"]
    spend = spend_data or supplier_info["ytd_spend"]
    
    strategy = {
        "supplier_id": supplier_id,
        "supplier_name": supplier_info["name"],
        "current_position": position,
        "ytd_spend": spend,
        "strategy": {},
        "key_tactics": [],
        "expected_savings": "0-2%"
    }
    
    if position == "strategic":
        strategy["strategy"] = {
            "approach": "Partnership-focused",
            "priority": "Long-term relationship building",
            "risk_level": "Low"
        }
        strategy["key_tactics"] = [
            "Joint cost reduction initiatives",
            "Multi-year contracts with performance incentives",
            "Collaborative innovation projects"
        ]
        strategy["expected_savings"] = "3-5%"
        
    elif position == "leverage":
        strategy["strategy"] = {
            "approach": "Competitive pressure",
            "priority": "Price reduction",
            "risk_level": "Medium"
        }
        strategy["key_tactics"] = [
            "Request competitive bids",
            "Negotiate volume discounts", 
            "Consider supplier substitution"
        ]
        strategy["expected_savings"] = "10-15%"
        
    elif position == "routine":
        strategy["strategy"] = {
            "approach": "Efficiency-focused",
            "priority": "Process optimization",
            "risk_level": "Low"
        }
        strategy["key_tactics"] = [
            "Standardize specifications",
            "Automate ordering processes",
            "Consolidate suppliers"
        ]
        strategy["expected_savings"] = "5-8%"
        
    else:  # bottleneck
        strategy["strategy"] = {
            "approach": "Risk mitigation",
            "priority": "Supply security",
            "risk_level": "High"
        }
        strategy["key_tactics"] = [
            "Develop alternative suppliers",
            "Build safety stock",
            "Long-term contracts with guarantees"
        ]
        strategy["expected_savings"] = "0-2%"
    
    return json.dumps(strategy, indent=2)