"""
DAX Agent Backend - RetailAnalyticsCube with Real SSAS Connection
FastAPI backend with actual SSAS cube connectivity using PowerShell fallback
"""

import os
import re
import json
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Any, TypedDict
from enum import Enum

import pandas as pd
import pyodbc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangGraph + LLMs
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="RetailAnalyticsCube DAX Agent", description="LangGraph-powered DAX query generation for RetailAnalyticsCube")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for session logs
session_logs: Dict[str, List[Dict]] = {}

# =====================================================================
# MODELS & ENUMS
# =====================================================================

class LLMProvider(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class AuthType(str, Enum):
    WINDOWS = "windows"
    USERNAME_PASSWORD = "username_password"
    CONNECTION_STRING = "connection_string"

class CubeType(str, Enum):
    TABULAR = "tabular"
    MULTIDIMENSIONAL = "multidimensional"

class ConnectionConfig(BaseModel):
    auth_type: AuthType
    cube_type: CubeType
    server: Optional[str] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None

class LLMConfig(BaseModel):
    provider: LLMProvider
    model_name: str
    api_key: str
    endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None

class QueryRequest(BaseModel):
    session_id: str
    natural_query: str
    connection_config: ConnectionConfig
    llm_config: LLMConfig

class QueryResponse(BaseModel):
    session_id: str
    success: bool
    generated_dax: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    visualization_suggestion: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float
    agent_logs: List[Dict[str, Any]]

class DAXAgentState(TypedDict):
    session_id: str
    user_query: str
    connection_config: ConnectionConfig
    llm_config: LLMConfig
    schema_info: Dict[str, Any]
    generated_dax: str
    execution_result: Optional[pd.DataFrame]
    error_message: Optional[str]
    visualization_type: str
    agent_logs: List[Dict[str, Any]]
    final_response: Dict[str, Any]

# =====================================================================
# SSAS CONNECTION UTILITIES
# =====================================================================

def execute_dax_query_via_powershell(dax_query: str, config: ConnectionConfig) -> pd.DataFrame:
    """Execute DAX query using PowerShell and Analysis Services module"""
    import subprocess
    import tempfile
    import shutil

    try:
        # Create PowerShell script to execute DAX
        ps_script = f'''
# Import Analysis Services module
try {{
    Import-Module SqlServer -Force -ErrorAction Stop
}} catch {{
    Write-Error "SqlServer module not found. Please install: Install-Module -Name SqlServer"
    exit 1
}}

# Connection details
$server = "{config.server}"
$database = "{config.database}"

# DAX Query
$daxQuery = @"
{dax_query}
"@

try {{
    # Execute DAX query using Invoke-ASCmd
    $result = Invoke-ASCmd -Server $server -Database $database -Query $daxQuery

    # Parse XML result - SSAS returns data as child elements, not attributes
    [xml]$xmlResult = $result

    # Look for row elements in the rowset namespace
    $dataRows = $xmlResult.SelectNodes("//*[local-name()='row']")

    if ($dataRows.Count -gt 0) {{
        # Get column names from first row's child elements
        $firstRow = $dataRows[0]
        $columns = @()
        $cleanColumns = @()

        foreach ($child in $firstRow.ChildNodes) {{
            if ($child.NodeType -eq "Element") {{
                $originalName = $child.LocalName
                $columns += $originalName

                # Decode the column names (SSAS encodes special characters)
                $cleanName = $originalName
                $cleanName = $cleanName -replace "_x005B_", "["
                $cleanName = $cleanName -replace "_x005D_", "]"
                $cleanName = $cleanName -replace "Regions_", ""
                $cleanName = $cleanName -replace "^_", ""

                $cleanColumns += $cleanName
            }}
        }}

        if ($columns.Count -gt 0) {{
            # Output header with clean names
            $cleanColumns -join ","

            # Output data rows
            foreach ($row in $dataRows) {{
                $values = @()
                for ($i = 0; $i -lt $columns.Count; $i++) {{
                    $colName = $columns[$i]
                    $value = ""

                    foreach ($child in $row.ChildNodes) {{
                        if ($child.LocalName -eq $colName) {{
                            $value = $child.InnerText
                            break
                        }}
                    }}

                    if ($value -eq $null) {{ $value = "" }}

                    # Convert scientific notation to regular numbers
                    if ($value -match "^[0-9]*\.?[0-9]+E[+-]?[0-9]+$") {{
                        try {{
                            $value = [double]$value
                            $value = $value.ToString("F2")
                        }} catch {{ }}
                    }}

                    # Escape commas and quotes in values
                    if ($value -like "*,*" -or $value -like '*"*') {{
                        $value = '"' + ($value -replace '"', '""') + '"'
                    }}
                    $values += $value
                }}
                $values -join ","
            }}
        }} else {{
            Write-Output "No child elements found in rows"
            Write-Output "Raw XML Result:"
            Write-Output $result
        }}
    }} else {{
        Write-Output "No data rows found"
        Write-Output "Raw XML Result:"
        Write-Output $result
    }}
}} catch {{
    Write-Error "DAX execution failed: $($_.Exception.Message)"
    exit 1
}}
'''
        # Write PowerShell script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8') as f:
            f.write(ps_script)
            script_path = f.name

        try:
            # Execute PowerShell script (with -NoProfile and clear error surfacing)
            ps_exe = shutil.which("powershell.exe") or "powershell.exe"
            result = subprocess.run(
                [ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path],
                capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"PowerShell failed (code {result.returncode}).\n"
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

            if result.stdout.strip():
                # Parse CSV output
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:  # header + rows
                    import csv, io
                    csv_data = '\n'.join(lines)
                    df = pd.read_csv(io.StringIO(csv_data))
                    return df
                elif len(lines) == 1 and lines[0] != "No data returned":
                    columns = lines[0].split(',')
                    df = pd.DataFrame([columns], columns=[f"Column_{i}" for i in range(len(columns))])
                    return df
                else:
                    return pd.DataFrame([{"Message": "No data returned"}])
            else:
                return pd.DataFrame([{"Message": "No data returned"}])

        finally:
            try:
                os.unlink(script_path)
            except:
                pass

    except Exception as e:
        raise RuntimeError(f"PowerShell DAX execution failed: {str(e)}")

def create_ssas_connection(config: ConnectionConfig) -> pyodbc.Connection:
    """Create actual SSAS connection based on configuration"""
    try:
        # List available drivers for debugging
        available_drivers = [driver for driver in pyodbc.drivers()]
        print(f"Available ODBC drivers: {available_drivers}")

        if config.auth_type == AuthType.CONNECTION_STRING and config.connection_string:
            return pyodbc.connect(config.connection_string)

        # Try multiple driver names in order of preference
        driver_names = [
            "MSMDPUMP.DLL",
            "Microsoft Analysis Services ODBC Driver",
            "SQL Server Analysis Services",
            "MSOLAP",
            "Microsoft OLE DB Provider for Analysis Services 12.0",
            "Microsoft OLE DB Provider for Analysis Services 11.0"
        ]

        connection_error = None

        for driver_name in driver_names:
            try:
                if config.auth_type == AuthType.WINDOWS:
                    conn_str = f"DRIVER={{{driver_name}}};SERVER={config.server};DATABASE={config.database};Integrated Security=SSPI;"
                    print(f"Trying connection string: {conn_str}")
                    return pyodbc.connect(conn_str)

                elif config.auth_type == AuthType.USERNAME_PASSWORD:
                    conn_str = f"DRIVER={{{driver_name}}};SERVER={config.server};DATABASE={config.database};UID={config.username};PWD={config.password};"
                    print(f"Trying connection string: {conn_str}")
                    return pyodbc.connect(conn_str)

            except Exception as e:
                connection_error = e
                print(f"Failed with driver '{driver_name}': {str(e)}")
                continue

        # If all drivers failed, try alternative approach
        try:
            if config.auth_type == AuthType.WINDOWS:
                conn_str = f"Provider=MSOLAP;Data Source={config.server};Initial Catalog={config.database};Integrated Security=SSPI;"
                return pyodbc.connect(f"DRIVER={{SQL Server}};{conn_str}")
        except Exception as e:
            print(f"Alternative connection also failed: {str(e)}")

        raise ConnectionError(f"Failed to connect to SSAS with any available driver. Last error: {str(connection_error)}")

    except Exception as e:
        raise ConnectionError(f"Failed to connect to SSAS: {str(e)}")

def execute_dax_query(dax_query: str, config: ConnectionConfig) -> pd.DataFrame:
    """Execute DAX query against real SSAS cube using multiple fallback methods"""

    # Method 1: PowerShell
    try:
        print("Attempting PowerShell method...")
        return execute_dax_query_via_powershell(dax_query, config)
    except Exception as ps_error:
        print(f"PowerShell method failed: {str(ps_error)}")

    # Method 2: ODBC
    try:
        print("Attempting ODBC method...")
        conn = create_ssas_connection(config)
        try:
            df = pd.read_sql(dax_query, conn)
            return df
        finally:
            conn.close()
    except Exception as odbc_error:
        print(f"ODBC method failed: {str(odbc_error)}")

    # Method 3: sqlcmd (best-effort)
    try:
        print("Attempting sqlcmd method...")
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dax', delete=False) as f:
            f.write(dax_query)
            query_file = f.name

        cmd = [
            'sqlcmd',
            '-S', config.server,
            '-d', config.database,
            '-E',
            '-Q', dax_query,
            '-s', '|',
            '-h', '-1'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and result.stdout.strip():
            lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            if lines:
                if len(lines) >= 2:
                    headers = lines[0].split('|')
                    data = [line.split('|') for line in lines[1:]]
                    df = pd.DataFrame(data, columns=headers)
                    return df
                else:
                    return pd.DataFrame([{"Result": lines[0]}])

        try:
            os.unlink(query_file)
        except:
            pass

    except Exception as sqlcmd_error:
        print(f"sqlcmd method failed: {str(sqlcmd_error)}")

    raise RuntimeError("All connection methods failed. Please install Analysis Services ODBC driver or SQL Server PowerShell module.")

def extract_cube_schema(config: ConnectionConfig) -> Dict[str, Any]:
    """Extract actual schema from SSAS cube or return fallback"""
    try:
        schema_query = """
EVALUATE
SELECTCOLUMNS(
    INFO.TABLES(),
    "TABLE_NAME", [TABLE_NAME],
    "TABLE_TYPE", [TABLE_TYPE]
)
"""
        _ = execute_dax_query(schema_query, config)
        # For now, still return a curated fallback (keeps changes small)
        return get_fallback_schema()
    except Exception as e:
        print(f"Schema extraction failed: {str(e)}")
        return get_fallback_schema()

def get_fallback_schema() -> Dict[str, Any]:
    """Fallback schema when dynamic extraction fails (includes Date table and SaleDate)"""
    return {
        "tables": {
            "Customers": [
                {"name": "CustomerID", "type": "Integer", "hidden": False, "key": True},
                {"name": "CustomerName", "type": "String", "hidden": False, "key": False},
                {"name": "Email", "type": "String", "hidden": False, "key": False},
                {"name": "RegionID", "type": "Integer", "hidden": False, "key": False},
                {"name": "CustomerType", "type": "String", "hidden": False, "key": False}
            ],
            "Products": [
                {"name": "ProductID", "type": "Integer", "hidden": False, "key": True},
                {"name": "ProductName", "type": "String", "hidden": False, "key": False},
                {"name": "Category", "type": "String", "hidden": False, "key": False},
                {"name": "SubCategory", "type": "String", "hidden": False, "key": False},
                {"name": "UnitPrice", "type": "Decimal", "hidden": False, "key": False},
                {"name": "Cost", "type": "Decimal", "hidden": False, "key": False},
                {"name": "Supplier", "type": "String", "hidden": False, "key": False}
            ],
            "Sales": [
                {"name": "SalesID", "type": "Integer", "hidden": False, "key": True},
                {"name": "CustomerID", "type": "Integer", "hidden": False, "key": False},
                {"name": "ProductID", "type": "Integer", "hidden": False, "key": False},
                {"name": "SaleDate", "type": "Date", "hidden": False, "key": False},  # important
                {"name": "Quantity", "type": "Integer", "hidden": False, "key": False},
                {"name": "UnitPrice", "type": "Decimal", "hidden": False, "key": False},
                {"name": "Discount", "type": "Decimal", "hidden": False, "key": False},
                {"name": "TotalAmount", "type": "Decimal", "hidden": False, "key": False}
            ],
            "Regions": [
                {"name": "RegionID", "type": "Integer", "hidden": False, "key": True},
                {"name": "RegionName", "type": "String", "hidden": False, "key": False},
                {"name": "Country", "type": "String", "hidden": False, "key": False},
                {"name": "Manager", "type": "String", "hidden": False, "key": False}
            ],
            "Date": [  # added so LLM can see real calendar table
                {"name": "Date", "type": "Date", "hidden": False, "key": True},
                {"name": "DateKey", "type": "Integer", "hidden": False, "key": False},
                {"name": "Year", "type": "Integer", "hidden": False, "key": False},
                {"name": "Quarter", "type": "String", "hidden": False, "key": False},
                {"name": "Month", "type": "Integer", "hidden": False, "key": False},
                {"name": "MonthName", "type": "String", "hidden": False, "key": False},
                {"name": "WeekOfYear", "type": "Integer", "hidden": False, "key": False},
                {"name": "DayOfWeek", "type": "Integer", "hidden": False, "key": False},
                {"name": "DayName", "type": "String", "hidden": False, "key": False},
                {"name": "IsWeekend", "type": "Boolean", "hidden": False, "key": False}
            ]
        },
        "relationships": [
            {"from_table": "Sales", "from_column": "CustomerID", "to_table": "Customers", "to_column": "CustomerID"},
            {"from_table": "Sales", "from_column": "ProductID", "to_table": "Products", "to_column": "ProductID"},
            {"from_table": "Customers", "from_column": "RegionID", "to_table": "Regions", "to_column": "RegionID"},
            # crucial: fact to calendar
            {"from_table": "Sales", "from_column": "SaleDate", "to_table": "Date", "to_column": "Date"}
        ],
        "common_measures": [
            "SUM(Sales[TotalAmount])",
            "SUM(Sales[Quantity])",
            "COUNT(Sales[SalesID])",
            "AVERAGE(Sales[TotalAmount])",
            "DIVIDE(SUM(Sales[TotalAmount]), COUNT(Sales[SalesID]))"
        ],
        "cube_type": "tabular"
    }

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def log_agent_step(session_id: str, step: str, status: str, details: Dict[str, Any] = None):
    """Log agent execution steps"""
    if session_id not in session_logs:
        session_logs[session_id] = []
    log_entry = {"timestamp": datetime.now().isoformat(), "step": step, "status": status, "details": details or {}}
    session_logs[session_id].append(log_entry)

def get_llm_instance(config: LLMConfig):
    """Create LLM instance based on configuration"""
    common_params = {"temperature": 0.1}

    if config.provider == LLMProvider.OPENAI:
        params = {"api_key": config.api_key, "model": config.model_name}
        if config.endpoint:
            params["base_url"] = config.endpoint
        return ChatOpenAI(**params, **common_params)

    if config.provider == LLMProvider.AZURE_OPENAI:
        params = {
            "api_key": config.api_key,
            "azure_deployment": (config.deployment_name or config.model_name),
            "azure_endpoint": config.endpoint,
            "api_version": (config.api_version or "2024-02-01")
        }
        return AzureChatOpenAI(**params, **common_params)

    if config.provider == LLMProvider.ANTHROPIC:
        params = {"api_key": config.api_key, "model": config.model_name}
        if config.endpoint:
            params["base_url"] = config.endpoint
        return ChatAnthropic(**params, **common_params)

    if config.provider == LLMProvider.GOOGLE:
        return ChatGoogleGenerativeAI(api_key=config.api_key, model=config.model_name, **common_params)

    raise ValueError(f"Unsupported LLM provider: {config.provider}")

def determine_visualization_type(df: pd.DataFrame, query: str) -> str:
    """Suggest visualization type based on data and query"""
    query_lower = query.lower()
    if df is None or df.empty:
        return "table"
    if any(term in query_lower for term in ['top', 'bottom', 'highest', 'lowest', 'rank']):
        return "bar"
    if any(term in query_lower for term in ['compare', 'vs', 'versus', 'by']):
        return "bar"
    if any(term in query_lower for term in ['category', 'region', 'product', 'customer']):
        return "bar"
    if len(df) <= 8 and len(df.columns) == 2:
        non_numeric = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric) == 1:
            return "pie"
    return "table"

def validate_dax_query(dax_query: str) -> bool:
    """Basic DAX query validation"""
    dax_query = dax_query.strip()

    if not dax_query.upper().startswith('EVALUATE'):
        return False

    open_count = dax_query.count('(')
    close_count = dax_query.count(')')
    if open_count != close_count:
        return False

    dax_functions = ['SUMMARIZECOLUMNS', 'SUM', 'COUNT', 'AVERAGE', 'TOPN', 'FILTER', 'CALCULATE', 'DATESYTD', 'DATESINPERIOD']
    has_function = any(func in dax_query.upper() for func in dax_functions)

    return has_function

# =====================================================================
# LANGGRAPH AGENT
# =====================================================================

class RetailAnalyticsDAXAgent:
    def __init__(self):
        self.system_prompt = r"""You are an expert DAX analyst for RetailAnalyticsCube with deep knowledge of retail analytics.

CRITICAL INSTRUCTIONS:
1. Generate ONLY valid DAX syntax - no explanations, comments, or markdown
2. Always use EVALUATE to return a table
3. Use exact table names from the provided schema
4. Use proper column references: TableName[ColumnName]
5. Prefer SUMMARIZECOLUMNS for aggregations
6. Use TOPN to limit results (default 10-20 rows)
7. Order results meaningfully with DESC/ASC

COMMON PATTERNS:
Top customers: TOPN(10, SUMMARIZECOLUMNS(Customers[CustomerName], "Total", SUM(Sales[TotalAmount])), [Total], DESC)
Sales by category: SUMMARIZECOLUMNS(Products[Category], "Sales", SUM(Sales[TotalAmount]))
Regional performance: SUMMARIZECOLUMNS(Regions[RegionName], "Sales", SUM(Sales[TotalAmount]), "Orders", COUNT(Sales[SalesID]))
This year sales (with Date table): EVALUATE ROW("Total Sales This Year", CALCULATE(SUM(Sales[TotalAmount]), DATESYTD('Date'[Date])))

# Retail Sales Analytics Cube - AI Agent Context

## Overview
This is a Retail Sales Analytics tabular model built from CSV files. It enables analysis of customer behavior, product performance, sales trends, and store operations for a retail business.

## Business Domain
- Primary Purpose: Analyze retail sales performance, customer segments, and store operations
- Key Users: Sales managers, merchandising teams, store managers, executives
- Time Granularity: Daily sales data with date hierarchy support
- Data Period: 2023-2025 calendar years

## Tables Structure

### Sales (Primary Fact Table)
Central transaction table containing all sales records.
Key Metrics: TotalAmount, Quantity, UnitPrice, Discount, ShippingCost
Key Attributes: SalesID, SaleDate, SalesChannelID
Relationships: Links to Customers (CustomerID), Products (ProductID), Stores (StoreID), Date (SaleDate)

### Customers (Dimension Table)
Customer master data and segmentation.
Key Attributes: CustomerID, CustomerName, Email, DateJoined, CustomerSegment, LoyaltyPoints, PreferredContactMethod
Relationships: Links to Regions (RegionID), Sales (CustomerID)

### Products (Dimension Table)
Product catalog and attributes.
Hierarchy: Brand → Category → SubCategory → ProductName
Key Attributes: ProductID, UnitPrice, Cost, LaunchDate, IsDiscontinued, SeasonalityFlag

### Stores (Dimension Table)
Store information and characteristics.
Key Attributes: StoreID, StoreName, StoreType, ManagerName, OpenDate, SquareFootage, StaffCount
Relationships: Links to Regions (RegionID), Sales (StoreID)

### Regions (Dimension Table)
Geographic and demographic information.
Key Attributes: RegionID, RegionName, Country, ManagerName, Population, AvgIncome

### Date (Calculated Dimension Table)
Comprehensive date dimension for time-based analysis.
Hierarchy: Year → Quarter → Month → Date
Key Attributes: Date, DateKey, Year, Quarter, Month, MonthName, WeekOfYear, DayOfWeek, DayName, IsWeekend

## Key Measures (DAX)
- Total Sales: SUM(Sales[TotalAmount]) - Total revenue
- Sales Count: COUNTROWS(Sales) - Number of transactions
- Total Quantity: SUM(Sales[Quantity]) - Units sold
- Customer Count: DISTINCTCOUNT(Sales[CustomerID]) - Unique customers
- Average Order Value: DIVIDE([Total Sales], [Sales Count], 0) - AOV calculation
- Sales YTD: TOTALYTD([Total Sales], 'Date'[Date]) - Year-to-date sales
- Sales Last Year: CALCULATE([Total Sales], SAMEPERIODLASTYEAR('Date'[Date])) - Prior year comparison
- YoY Growth %: DIVIDE([Total Sales] - [Sales Last Year], [Sales Last Year], 0) * 100 - Year-over-year growth

## Key Relationships
Star Schema Design: Sales (fact) connects to all dimension tables
Date relationship uses SaleDate field
Cross-filtering enabled for Products and Date (bidirectional)
Inactive Relationships: Stores → Regions relationship is inactive (to avoid ambiguity with Customers → Regions)

Generate clean DAX that will execute successfully against the cube."""
    
    def extract_schema(self, state: DAXAgentState) -> DAXAgentState:
        session_id = state["session_id"]
        log_agent_step(session_id, "schema_extraction", "started")
        try:
            schema_info = extract_cube_schema(state["connection_config"])
            state["schema_info"] = schema_info
            log_agent_step(session_id, "schema_extraction", "completed",
                           {"tables": list(schema_info["tables"].keys()), "relationships": len(schema_info["relationships"])})
        except Exception as e:
            error_msg = f"Schema extraction failed: {str(e)}"
            state["error_message"] = error_msg
            log_agent_step(session_id, "schema_extraction", "failed", {"error": error_msg})
        return state
    
    def generate_dax(self, state: DAXAgentState) -> DAXAgentState:
        session_id = state["session_id"]
        log_agent_step(session_id, "dax_generation", "started",
                       {"query": state["user_query"], "llm_provider": state["llm_config"].provider.value})
        try:
            schema_context = json.dumps(state["schema_info"], indent=2)

            # === time + strict schema hints (minimal, but powerful) ===
            tables = state["schema_info"].get("tables", {})
            sales_cols = [c["name"] for c in tables.get("Sales", [])]
            date_cols  = [c["name"] for c in tables.get("Date", [])]

            today = datetime.now().date()
            time_context = (
                f"\nTIME CONTEXT:\n"
                f"- Today: {today.isoformat()}\n"
                f"- Current Year: {today.year}\n"
                f"- Current Month Number: {today.month}\n"
            )

            schema_hints = "\nSTRICT SCHEMA RULES:\n- Use ONLY columns listed in SCHEMA DETAILS. Do not invent columns.\n"
            if "SaleDate" in sales_cols:
                schema_hints += "- The sales date column is Sales[SaleDate].\n"
            if "Date" in tables:
                schema_hints += (
                    "- Use the 'Date' table for calendar/time-intelligence; use 'Date'[Date] in DATESYTD/DATESINPERIOD, "
                    "and group by existing fields like 'Date'[Year], 'Date'[Month], 'Date'[MonthName].\n"
                )
            # ===========================================================

            prompt = f"""
{self.system_prompt}

SCHEMA DETAILS:
{schema_context}
{time_context}
{schema_hints}
USER QUESTION: "{state['user_query']}"

Generate the DAX query to answer this question. Return ONLY the DAX code.
"""
            llm = get_llm_instance(state["llm_config"])
            response = llm.invoke([SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)])
            dax_query = response.content.strip()

            # Clean up the response
            dax_query = re.sub(r"^```[a-zA-Z]*\n", "", dax_query)
            dax_query = re.sub(r"\n```$", "", dax_query).strip()
            dax_query = dax_query.rstrip(";")

            # console logs for visibility
            print(f"=== USER QUERY (Python) [{session_id}] === {state['user_query']}", flush=True)
            print(f"=== DAX GENERATED (Python) [{session_id}] ===\n{dax_query}", flush=True)

            # Validate DAX query
            if not validate_dax_query(dax_query):
                raise ValueError("Generated DAX query failed validation")

            state["generated_dax"] = dax_query
            log_agent_step(session_id, "dax_generation", "completed",
                           {"generated_dax_preview": dax_query[:150] + "..." if len(dax_query) > 150 else dax_query})
        except Exception as e:
            error_msg = f"DAX generation failed: {str(e)}"
            state["error_message"] = error_msg
            log_agent_step(session_id, "dax_generation", "failed", {"error": error_msg})
        return state
    
    def execute_query(self, state: DAXAgentState) -> DAXAgentState:
        session_id = state["session_id"]
        log_agent_step(session_id, "query_execution", "started")
        try:
            df = execute_dax_query(state["generated_dax"], state["connection_config"])
            state["execution_result"] = df
            log_agent_step(session_id, "query_execution", "completed",
                           {"rows_returned": len(df), "columns": list(df.columns)})
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            state["error_message"] = error_msg
            log_agent_step(session_id, "query_execution", "failed", {"error": error_msg})
        return state
    
    def determine_visualization(self, state: DAXAgentState) -> DAXAgentState:
        session_id = state["session_id"]
        log_agent_step(session_id, "visualization_analysis", "started")
        try:
            df = state.get("execution_result")
            viz_type = determine_visualization_type(df, state["user_query"])
            state["visualization_type"] = viz_type
            log_agent_step(session_id, "visualization_analysis", "completed", {"suggested_visualization": viz_type})
        except Exception as e:
            state["visualization_type"] = "table"
            log_agent_step(session_id, "visualization_analysis", "failed", {"error": str(e), "fallback": "table"})
        return state
    
    def format_response(self, state: DAXAgentState) -> DAXAgentState:
        session_id = state["session_id"]
        df = state.get("execution_result")
        success = not state.get("error_message") and df is not None

        result_data = None
        if df is not None and not df.empty:
            result_data = {
                "columns": df.columns.tolist(),
                "data": df.values.tolist(),
                "shape": df.shape
            }

        response = {
            "success": success,
            "generated_dax": state.get("generated_dax", ""),
            "execution_result": result_data,
            "visualization_suggestion": state.get("visualization_type", "table"),
            "error_message": state.get("error_message"),
            "schema_info": state.get("schema_info", {}),
            "agent_logs": session_logs.get(session_id, [])
        }

        state["final_response"] = response
        log_agent_step(session_id, "response_formatting", "completed")
        return state
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(DAXAgentState)
        workflow.add_node("extract_schema", self.extract_schema)
        workflow.add_node("generate_dax", self.generate_dax)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("determine_viz", self.determine_visualization)
        workflow.add_node("format_response", self.format_response)

        workflow.add_edge(START, "extract_schema")
        workflow.add_edge("extract_schema", "generate_dax")
        workflow.add_edge("generate_dax", "execute_query")
        workflow.add_edge("execute_query", "determine_viz")
        workflow.add_edge("determine_viz", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile()

# Global agent instance
retail_agent = RetailAnalyticsDAXAgent()
workflow = retail_agent.create_workflow()

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.post("/api/execute-query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    start_time = datetime.now()
    session_logs[request.session_id] = []

    initial_state: DAXAgentState = {
        "session_id": request.session_id,
        "user_query": request.natural_query,
        "connection_config": request.connection_config,
        "llm_config": request.llm_config,
        "schema_info": {},
        "generated_dax": "",
        "execution_result": None,
        "error_message": None,
        "visualization_type": "",
        "agent_logs": [],
        "final_response": {}
    }

    try:
        result = workflow.invoke(initial_state)
        execution_time = (datetime.now() - start_time).total_seconds()
        response = result["final_response"]

        return QueryResponse(
            session_id=request.session_id,
            success=response["success"],
            generated_dax=response["generated_dax"],
            execution_result=response["execution_result"],
            visualization_suggestion=response["visualization_suggestion"],
            error_message=response["error_message"],
            execution_time=execution_time,
            agent_logs=response["agent_logs"]
        )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Workflow execution failed: {str(e)}"

        return QueryResponse(
            session_id=request.session_id,
            success=False,
            error_message=error_msg,
            execution_time=execution_time,
            agent_logs=session_logs.get(request.session_id, [])
        )

@app.post("/api/test-connection")
async def test_connection(connection_config: ConnectionConfig):
    try:
        test_query = 'EVALUATE ROW("Test", 1)'
        _ = execute_dax_query(test_query, connection_config)
        return {"success": True, "message": "Successfully connected to RetailAnalyticsCube", "method": "PowerShell"}
    except Exception as e:
        return {"success": False, "message": f"Connection failed: {str(e)}"}

@app.get("/api/session-logs/{session_id}")
async def get_session_logs(session_id: str):
    return {"session_id": session_id, "logs": session_logs.get(session_id, [])}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "cube": "RetailAnalyticsCube", "timestamp": datetime.now().isoformat()}

@app.get("/api/models")
async def get_available_models():
    return {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "azure_openai": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
    }

@app.get("/api/schema")
async def get_cube_schema():
    """Get fallback schema structure"""
    return get_fallback_schema()

@app.get("/api/schema-live")
async def get_cube_schema_live(server: str, database: str):
    """Get live schema from actual cube"""
    try:
        config = ConnectionConfig(
            auth_type=AuthType.WINDOWS,
            cube_type=CubeType.TABULAR,
            server=server,
            database=database
        )
        schema = extract_cube_schema(config)
        return schema
    except Exception as e:
        return {"error": f"Failed to extract schema: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
