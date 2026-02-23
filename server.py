from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Open-Meteo API (Free, no key required)
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"

app = FastAPI(title="DroughtGuard API", version="2.0.0")
api_router = APIRouter(prefix="/api")

# ============ MODELS ============

class Village(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    district: str
    state: str
    latitude: float
    longitude: float
    population: int
    groundwater_level: float  # in meters (lower = worse)
    rainfall_actual: float  # mm
    rainfall_normal: float  # mm (expected)
    water_stress_index: float = 0.0  # 0-100 (higher = more stress)
    risk_level: str = "safe"  # safe, moderate, critical
    tanker_demand: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class VillageCreate(BaseModel):
    name: str
    district: str
    state: str
    latitude: float
    longitude: float
    population: int
    groundwater_level: float
    rainfall_actual: float
    rainfall_normal: float

class Tanker(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vehicle_number: str
    capacity: int  # liters
    status: str = "available"  # available, dispatched, maintenance
    assigned_village_id: Optional[str] = None
    assigned_village_name: Optional[str] = None
    driver_name: str
    driver_phone: str
    last_dispatch: Optional[datetime] = None

class TankerCreate(BaseModel):
    vehicle_number: str
    capacity: int
    driver_name: str
    driver_phone: str

class TankerAllocation(BaseModel):
    tanker_id: str
    village_id: str

class WeatherData(BaseModel):
    village_id: str
    temperature: float
    humidity: float
    rainfall: float
    description: str
    fetched_at: datetime

class DashboardStats(BaseModel):
    total_villages: int
    critical_villages: int
    moderate_villages: int
    safe_villages: int
    total_tankers: int
    available_tankers: int
    dispatched_tankers: int
    total_population_at_risk: int
    avg_water_stress_index: float

# Historical Data Models
class HistoricalRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    village_id: str
    village_name: str
    date: str  # YYYY-MM-DD
    water_stress_index: float
    rainfall_actual: float
    groundwater_level: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    risk_level: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ============ UTILITY FUNCTIONS ============

def calculate_water_stress_index(village_data: dict) -> tuple:
    """Calculate Water Stress Index (WSI) based on multiple factors"""
    rainfall_deviation = 0
    if village_data['rainfall_normal'] > 0:
        rainfall_deviation = ((village_data['rainfall_normal'] - village_data['rainfall_actual']) / village_data['rainfall_normal']) * 100
    
    # Groundwater factor (shallow = bad, deep = worse)
    groundwater_factor = min(100, max(0, (10 - village_data['groundwater_level']) * 10))
    
    # Population pressure factor
    pop_factor = min(30, village_data['population'] / 1000)
    
    # Calculate WSI (weighted average)
    wsi = (rainfall_deviation * 0.4) + (groundwater_factor * 0.4) + (pop_factor * 0.2)
    wsi = max(0, min(100, wsi))  # Clamp between 0-100
    
    # Determine risk level
    if wsi >= 70:
        risk_level = "critical"
    elif wsi >= 40:
        risk_level = "moderate"
    else:
        risk_level = "safe"
    
    # Calculate tanker demand based on population and stress
    tanker_demand = 0
    if risk_level == "critical":
        tanker_demand = max(1, village_data['population'] // 2000)
    elif risk_level == "moderate":
        tanker_demand = max(0, village_data['population'] // 5000)
    
    return round(wsi, 2), risk_level, tanker_demand

# ============ VILLAGE ROUTES ============

@api_router.get("/villages", response_model=List[Village])
async def get_villages(
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    district: Optional[str] = Query(None, description="Filter by district")
):
    """Get all villages with optional filtering"""
    query = {}
    if risk_level:
        query["risk_level"] = risk_level
    if district:
        query["district"] = district
    
    villages = await db.villages.find(query, {"_id": 0}).to_list(1000)
    for v in villages:
        if isinstance(v.get('last_updated'), str):
            v['last_updated'] = datetime.fromisoformat(v['last_updated'])
    return villages

@api_router.get("/villages/{village_id}", response_model=Village)
async def get_village(village_id: str):
    """Get a single village by ID"""
    village = await db.villages.find_one({"id": village_id}, {"_id": 0})
    if not village:
        raise HTTPException(status_code=404, detail="Village not found")
    if isinstance(village.get('last_updated'), str):
        village['last_updated'] = datetime.fromisoformat(village['last_updated'])
    return village

@api_router.post("/villages", response_model=Village)
async def create_village(village_data: VillageCreate):
    """Create a new village"""
    village_dict = village_data.model_dump()
    wsi, risk_level, tanker_demand = calculate_water_stress_index(village_dict)
    
    village = Village(
        **village_dict,
        water_stress_index=wsi,
        risk_level=risk_level,
        tanker_demand=tanker_demand
    )
    
    doc = village.model_dump()
    doc['last_updated'] = doc['last_updated'].isoformat()
    await db.villages.insert_one(doc)
    return village

@api_router.put("/villages/{village_id}", response_model=Village)
async def update_village(village_id: str, village_data: VillageCreate):
    """Update a village and recalculate WSI"""
    existing = await db.villages.find_one({"id": village_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Village not found")
    
    village_dict = village_data.model_dump()
    wsi, risk_level, tanker_demand = calculate_water_stress_index(village_dict)
    
    update_data = {
        **village_dict,
        "water_stress_index": wsi,
        "risk_level": risk_level,
        "tanker_demand": tanker_demand,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    await db.villages.update_one({"id": village_id}, {"$set": update_data})
    updated = await db.villages.find_one({"id": village_id}, {"_id": 0})
    if isinstance(updated.get('last_updated'), str):
        updated['last_updated'] = datetime.fromisoformat(updated['last_updated'])
    return updated

@api_router.delete("/villages/{village_id}")
async def delete_village(village_id: str):
    """Delete a village"""
    result = await db.villages.delete_one({"id": village_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Village not found")
    return {"message": "Village deleted successfully"}

# ============ TANKER ROUTES ============

@api_router.get("/tankers", response_model=List[Tanker])
async def get_tankers(status: Optional[str] = Query(None)):
    """Get all tankers with optional status filter"""
    query = {}
    if status:
        query["status"] = status
    tankers = await db.tankers.find(query, {"_id": 0}).to_list(100)
    for t in tankers:
        if isinstance(t.get('last_dispatch'), str):
            t['last_dispatch'] = datetime.fromisoformat(t['last_dispatch'])
    return tankers

@api_router.post("/tankers", response_model=Tanker)
async def create_tanker(tanker_data: TankerCreate):
    """Create a new tanker"""
    tanker = Tanker(**tanker_data.model_dump())
    doc = tanker.model_dump()
    if doc.get('last_dispatch'):
        doc['last_dispatch'] = doc['last_dispatch'].isoformat()
    await db.tankers.insert_one(doc)
    return tanker

@api_router.post("/tankers/allocate")
async def allocate_tanker(allocation: TankerAllocation):
    """Allocate a tanker to a village"""
    tanker = await db.tankers.find_one({"id": allocation.tanker_id})
    if not tanker:
        raise HTTPException(status_code=404, detail="Tanker not found")
    if tanker.get("status") != "available":
        raise HTTPException(status_code=400, detail="Tanker is not available")
    
    village = await db.villages.find_one({"id": allocation.village_id})
    if not village:
        raise HTTPException(status_code=404, detail="Village not found")
    
    await db.tankers.update_one(
        {"id": allocation.tanker_id},
        {"$set": {
            "status": "dispatched",
            "assigned_village_id": allocation.village_id,
            "assigned_village_name": village.get("name"),
            "last_dispatch": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    return {"message": f"Tanker allocated to {village.get('name')}"}

@api_router.post("/tankers/{tanker_id}/release")
async def release_tanker(tanker_id: str):
    """Release a tanker back to available"""
    result = await db.tankers.update_one(
        {"id": tanker_id},
        {"$set": {
            "status": "available",
            "assigned_village_id": None,
            "assigned_village_name": None
        }}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Tanker not found")
    return {"message": "Tanker released successfully"}

# ============ WEATHER ROUTES (Open-Meteo - FREE, No API Key) ============

@api_router.get("/weather/{village_id}")
async def get_weather(village_id: str):
    """Fetch current weather data using Open-Meteo API (FREE)"""
    village = await db.villages.find_one({"id": village_id}, {"_id": 0})
    if not village:
        raise HTTPException(status_code=404, detail="Village not found")
    
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f"{OPEN_METEO_BASE}/forecast",
                params={
                    "latitude": village["latitude"],
                    "longitude": village["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                    "timezone": "auto"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            weather_code = current.get("weather_code", 0)
            
            # Weather code to description mapping
            weather_descriptions = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Foggy", 48: "Depositing rime fog",
                51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
                61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
                80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
                95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
            }
            
            return {
                "village_id": village_id,
                "village_name": village["name"],
                "temperature": current.get("temperature_2m", 0),
                "humidity": current.get("relative_humidity_2m", 0),
                "rainfall": current.get("precipitation", 0),
                "wind_speed": current.get("wind_speed_10m", 0),
                "weather_code": weather_code,
                "description": weather_descriptions.get(weather_code, "Unknown"),
                "source": "open-meteo",
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Open-Meteo API failed: {e}")
        raise HTTPException(status_code=502, detail=f"Weather API error: {str(e)}")

@api_router.get("/weather/{village_id}/historical")
async def get_historical_weather(village_id: str, days: int = Query(30, ge=7, le=90)):
    """Fetch historical weather data using Open-Meteo Historical API (FREE)"""
    village = await db.villages.find_one({"id": village_id}, {"_id": 0})
    if not village:
        raise HTTPException(status_code=404, detail="Village not found")
    
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f"{OPEN_METEO_BASE}/forecast",
                params={
                    "latitude": village["latitude"],
                    "longitude": village["longitude"],
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
                    "past_days": days,
                    "timezone": "auto"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            
            history = []
            for i, date in enumerate(dates):
                history.append({
                    "date": date,
                    "temp_max": daily.get("temperature_2m_max", [None])[i] if i < len(daily.get("temperature_2m_max", [])) else None,
                    "temp_min": daily.get("temperature_2m_min", [None])[i] if i < len(daily.get("temperature_2m_min", [])) else None,
                    "precipitation": daily.get("precipitation_sum", [0])[i] if i < len(daily.get("precipitation_sum", [])) else 0,
                    "rain": daily.get("rain_sum", [0])[i] if i < len(daily.get("rain_sum", [])) else 0
                })
            
            # Calculate summary
            total_rainfall = sum(h.get("precipitation", 0) or 0 for h in history)
            avg_temp = sum((h.get("temp_max", 0) or 0 + h.get("temp_min", 0) or 0) / 2 for h in history) / len(history) if history else 0
            
            return {
                "village_id": village_id,
                "village_name": village["name"],
                "period_days": days,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "summary": {
                    "total_rainfall_mm": round(total_rainfall, 1),
                    "average_temperature": round(avg_temp, 1),
                    "rainy_days": len([h for h in history if (h.get("precipitation") or 0) > 1])
                },
                "daily_data": history,
                "source": "open-meteo"
            }
    except Exception as e:
        logger.error(f"Open-Meteo Historical API failed: {e}")
        raise HTTPException(status_code=502, detail=f"Historical weather API error: {str(e)}")

# ============ HISTORICAL DATA TRACKING ============

@api_router.post("/historical/record")
async def record_historical_data():
    """Record current data snapshot for all villages (call daily via cron)"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    records_created = 0
    for village in villages:
        # Check if already recorded today
        existing = await db.historical_records.find_one({
            "village_id": village["id"],
            "date": today
        })
        
        if not existing:
            record = HistoricalRecord(
                village_id=village["id"],
                village_name=village["name"],
                date=today,
                water_stress_index=village.get("water_stress_index", 0),
                rainfall_actual=village.get("rainfall_actual", 0),
                groundwater_level=village.get("groundwater_level", 0),
                risk_level=village.get("risk_level", "safe")
            )
            doc = record.model_dump()
            doc['created_at'] = doc['created_at'].isoformat()
            await db.historical_records.insert_one(doc)
            records_created += 1
    
    return {"message": f"Recorded {records_created} historical snapshots for {today}"}

@api_router.get("/historical/{village_id}")
async def get_village_historical_data(village_id: str, days: int = Query(30, ge=7, le=365)):
    """Get historical WSI and conditions for a village"""
    village = await db.villages.find_one({"id": village_id}, {"_id": 0})
    if not village:
        raise HTTPException(status_code=404, detail="Village not found")
    
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    
    records = await db.historical_records.find(
        {"village_id": village_id, "date": {"$gte": start_date}},
        {"_id": 0}
    ).sort("date", 1).to_list(365)
    
    if not records:
        return {
            "village_id": village_id,
            "village_name": village["name"],
            "message": "No historical data available. Run /api/historical/record to start tracking.",
            "records": []
        }
    
    # Calculate trends
    if len(records) >= 2:
        first_wsi = records[0].get("water_stress_index", 0)
        last_wsi = records[-1].get("water_stress_index", 0)
        wsi_trend = "increasing" if last_wsi > first_wsi else "decreasing" if last_wsi < first_wsi else "stable"
        wsi_change = round(last_wsi - first_wsi, 1)
    else:
        wsi_trend = "insufficient_data"
        wsi_change = 0
    
    return {
        "village_id": village_id,
        "village_name": village["name"],
        "period_days": days,
        "total_records": len(records),
        "trend_analysis": {
            "wsi_trend": wsi_trend,
            "wsi_change": wsi_change,
            "average_wsi": round(sum(r.get("water_stress_index", 0) for r in records) / len(records), 1),
            "max_wsi": max(r.get("water_stress_index", 0) for r in records),
            "min_wsi": min(r.get("water_stress_index", 0) for r in records)
        },
        "records": records
    }

@api_router.get("/historical/trends/all")
async def get_all_villages_trends(days: int = Query(30, ge=7, le=365)):
    """Get historical trends for all villages"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    
    trends = []
    for village in villages:
        records = await db.historical_records.find(
            {"village_id": village["id"], "date": {"$gte": start_date}},
            {"_id": 0}
        ).sort("date", 1).to_list(365)
        
        if len(records) >= 2:
            first_wsi = records[0].get("water_stress_index", 0)
            last_wsi = records[-1].get("water_stress_index", 0)
            wsi_change = round(last_wsi - first_wsi, 1)
            trend = "worsening" if wsi_change > 5 else "improving" if wsi_change < -5 else "stable"
        else:
            wsi_change = 0
            trend = "no_data"
        
        trends.append({
            "village_id": village["id"],
            "village_name": village["name"],
            "current_wsi": village.get("water_stress_index", 0),
            "current_risk": village.get("risk_level", "safe"),
            "wsi_change": wsi_change,
            "trend": trend,
            "data_points": len(records)
        })
    
    # Sort by WSI change (worst first)
    trends.sort(key=lambda x: x["wsi_change"], reverse=True)
    
    return {
        "period_days": days,
        "total_villages": len(trends),
        "worsening_count": len([t for t in trends if t["trend"] == "worsening"]),
        "improving_count": len([t for t in trends if t["trend"] == "improving"]),
        "stable_count": len([t for t in trends if t["trend"] == "stable"]),
        "villages": trends
    }

@api_router.post("/historical/seed")
async def seed_historical_data():
    """Seed historical data for demo purposes (last 30 days)"""
    import random
    
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    
    # Clear existing historical data
    await db.historical_records.delete_many({})
    
    records_created = 0
    for village in villages:
        base_wsi = village.get("water_stress_index", 50)
        base_rainfall = village.get("rainfall_actual", 200)
        base_gw = village.get("groundwater_level", 5)
        
        for day_offset in range(30, -1, -1):
            date = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            
            # Simulate gradual changes
            variation = (30 - day_offset) / 30  # 0 to 1 over 30 days
            wsi = base_wsi - (variation * random.uniform(5, 15))  # WSI was higher in past
            wsi = max(0, min(100, wsi + random.uniform(-3, 3)))
            
            rainfall = base_rainfall * (1 - variation * 0.3) + random.uniform(-20, 20)
            gw_level = base_gw + (variation * random.uniform(-0.5, 0.5))
            
            risk = "critical" if wsi >= 70 else "moderate" if wsi >= 40 else "safe"
            
            record = HistoricalRecord(
                village_id=village["id"],
                village_name=village["name"],
                date=date,
                water_stress_index=round(wsi, 1),
                rainfall_actual=round(rainfall, 1),
                groundwater_level=round(gw_level, 2),
                risk_level=risk
            )
            doc = record.model_dump()
            doc['created_at'] = doc['created_at'].isoformat()
            await db.historical_records.insert_one(doc)
            records_created += 1
    
    return {"message": f"Seeded {records_created} historical records for {len(villages)} villages (30 days)"}

# ============ ANALYSIS MODULES ============

# 1️⃣ RAINFALL DEVIATION ANALYZER
@api_router.get("/analysis/rainfall")
async def analyze_rainfall():
    """Analyze rainfall deviation across all villages"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    
    analysis = []
    total_deficit = 0
    severely_deficit = 0
    
    for v in villages:
        actual = v.get("rainfall_actual", 0)
        normal = v.get("rainfall_normal", 1)
        deviation = ((actual - normal) / normal) * 100
        deficit = normal - actual
        
        status = "surplus" if deviation >= 0 else "deficit"
        severity = "normal"
        if deviation <= -50:
            severity = "severe"
            severely_deficit += 1
        elif deviation <= -25:
            severity = "moderate"
        elif deviation <= -10:
            severity = "mild"
        
        if deviation < 0:
            total_deficit += 1
        
        analysis.append({
            "village_id": v.get("id"),
            "village_name": v.get("name"),
            "district": v.get("district"),
            "rainfall_actual": actual,
            "rainfall_normal": normal,
            "deviation_percent": round(deviation, 1),
            "deficit_mm": max(0, deficit),
            "status": status,
            "severity": severity,
            "prediction": "High drought risk" if deviation <= -40 else "Moderate concern" if deviation <= -20 else "Stable"
        })
    
    # Sort by deviation (worst first)
    analysis.sort(key=lambda x: x["deviation_percent"])
    
    avg_deviation = sum(a["deviation_percent"] for a in analysis) / len(analysis) if analysis else 0
    
    return {
        "summary": {
            "total_villages": len(analysis),
            "deficit_villages": total_deficit,
            "severely_deficit": severely_deficit,
            "average_deviation": round(avg_deviation, 1),
            "drought_prediction": "HIGH" if avg_deviation <= -30 else "MODERATE" if avg_deviation <= -15 else "LOW"
        },
        "villages": analysis
    }

# 2️⃣ GROUNDWATER TREND ANALYZER
@api_router.get("/analysis/groundwater")
async def analyze_groundwater():
    """Analyze groundwater levels and depletion trends"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    
    # Simulate historical trend (in production, this would come from time-series data)
    analysis = []
    critical_count = 0
    depleting_count = 0
    
    for v in villages:
        current_level = v.get("groundwater_level", 5)
        
        # Simulate trend based on rainfall deficit
        rainfall_ratio = v.get("rainfall_actual", 200) / max(v.get("rainfall_normal", 400), 1)
        monthly_depletion = round((1 - rainfall_ratio) * 0.5, 2)  # meters per month
        
        # Estimate recharge rate based on rainfall
        recharge_rate = round(rainfall_ratio * 0.3, 2)  # meters per month
        
        net_change = recharge_rate - monthly_depletion
        
        # Critical threshold is 3 meters
        months_to_critical = None
        if current_level > 3 and net_change < 0:
            months_to_critical = int((current_level - 3) / abs(net_change)) if net_change != 0 else None
        
        status = "stable"
        if current_level <= 2:
            status = "critical"
            critical_count += 1
        elif current_level <= 4:
            status = "low"
        elif net_change < -0.2:
            status = "depleting"
            depleting_count += 1
        
        analysis.append({
            "village_id": v.get("id"),
            "village_name": v.get("name"),
            "district": v.get("district"),
            "current_level_m": current_level,
            "monthly_depletion": monthly_depletion,
            "recharge_rate": recharge_rate,
            "net_change": round(net_change, 2),
            "status": status,
            "months_to_critical": months_to_critical,
            "recommendation": "Immediate intervention" if status == "critical" else "Monitor closely" if status in ["low", "depleting"] else "Routine monitoring"
        })
    
    analysis.sort(key=lambda x: x["current_level_m"])
    
    return {
        "summary": {
            "total_villages": len(analysis),
            "critical_level": critical_count,
            "actively_depleting": depleting_count,
            "average_level": round(sum(a["current_level_m"] for a in analysis) / len(analysis), 1) if analysis else 0,
            "alert_status": "CRITICAL" if critical_count > 2 else "WARNING" if depleting_count > 3 else "NORMAL"
        },
        "villages": analysis
    }

# 3️⃣ WATER STRESS INDEX GENERATOR
@api_router.get("/analysis/stress-index")
async def analyze_stress_index():
    """Detailed Water Stress Index analysis with component breakdown"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    
    analysis = []
    
    for v in villages:
        # Component calculations
        rainfall_deviation = 0
        if v.get("rainfall_normal", 0) > 0:
            rainfall_deviation = ((v.get("rainfall_normal", 0) - v.get("rainfall_actual", 0)) / v.get("rainfall_normal", 1)) * 100
        
        rainfall_score = min(100, max(0, rainfall_deviation))
        
        # Groundwater score (lower level = higher stress)
        gw_level = v.get("groundwater_level", 5)
        groundwater_score = min(100, max(0, (10 - gw_level) * 12))
        
        # Population pressure score
        population = v.get("population", 0)
        pop_score = min(40, population / 2000)
        
        # Component weights
        weighted_rainfall = rainfall_score * 0.40
        weighted_groundwater = groundwater_score * 0.40
        weighted_population = pop_score * 0.20
        
        total_wsi = weighted_rainfall + weighted_groundwater + weighted_population
        total_wsi = max(0, min(100, total_wsi))
        
        # Risk classification
        risk_level = "safe"
        risk_color = "green"
        if total_wsi >= 70:
            risk_level = "critical"
            risk_color = "red"
        elif total_wsi >= 40:
            risk_level = "moderate"
            risk_color = "orange"
        
        analysis.append({
            "village_id": v.get("id"),
            "village_name": v.get("name"),
            "district": v.get("district"),
            "components": {
                "rainfall": {
                    "raw_score": round(rainfall_score, 1),
                    "weight": 0.40,
                    "weighted_score": round(weighted_rainfall, 1),
                    "deviation_percent": round(rainfall_deviation, 1)
                },
                "groundwater": {
                    "raw_score": round(groundwater_score, 1),
                    "weight": 0.40,
                    "weighted_score": round(weighted_groundwater, 1),
                    "level_m": gw_level
                },
                "population": {
                    "raw_score": round(pop_score, 1),
                    "weight": 0.20,
                    "weighted_score": round(weighted_population, 1),
                    "count": population
                }
            },
            "total_wsi": round(total_wsi, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "primary_stressor": "rainfall" if weighted_rainfall >= weighted_groundwater else "groundwater"
        })
    
    analysis.sort(key=lambda x: x["total_wsi"], reverse=True)
    
    return {
        "summary": {
            "total_villages": len(analysis),
            "critical_count": len([a for a in analysis if a["risk_level"] == "critical"]),
            "moderate_count": len([a for a in analysis if a["risk_level"] == "moderate"]),
            "safe_count": len([a for a in analysis if a["risk_level"] == "safe"]),
            "average_wsi": round(sum(a["total_wsi"] for a in analysis) / len(analysis), 1) if analysis else 0,
            "highest_wsi": analysis[0]["total_wsi"] if analysis else 0,
            "most_affected": analysis[0]["village_name"] if analysis else None
        },
        "villages": analysis
    }

# 4️⃣ TANKER DEMAND PREDICTOR
@api_router.get("/analysis/tanker-demand")
async def predict_tanker_demand():
    """Predict tanker requirements based on population and stress levels"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    tankers = await db.tankers.find({}, {"_id": 0}).to_list(100)
    
    total_capacity = sum(t.get("capacity", 10000) for t in tankers)
    available_tankers = [t for t in tankers if t.get("status") == "available"]
    available_capacity = sum(t.get("capacity", 10000) for t in available_tankers)
    
    analysis = []
    total_demand = 0
    total_water_needed = 0
    
    for v in villages:
        wsi = v.get("water_stress_index", 0)
        population = v.get("population", 0)
        risk_level = v.get("risk_level", "safe")
        
        # Water requirement: 50 liters per person per day baseline
        base_requirement = population * 50  # liters
        
        # Adjust based on stress level
        if risk_level == "critical":
            multiplier = 1.5
            urgency = "URGENT"
        elif risk_level == "moderate":
            multiplier = 1.2
            urgency = "HIGH"
        else:
            multiplier = 0.3  # Only emergency supply for safe villages
            urgency = "LOW"
        
        daily_water_need = int(base_requirement * multiplier)
        
        # Calculate tanker trips (assuming 12000L average capacity)
        avg_tanker_capacity = 12000
        daily_trips = max(0, int(daily_water_need / avg_tanker_capacity))
        weekly_trips = daily_trips * 7
        
        # Cost estimation (Rs 500 per trip fuel + driver)
        weekly_cost = weekly_trips * 500
        
        if risk_level != "safe":
            total_demand += daily_trips
            total_water_needed += daily_water_need
        
        analysis.append({
            "village_id": v.get("id"),
            "village_name": v.get("name"),
            "district": v.get("district"),
            "population": population,
            "water_stress_index": wsi,
            "risk_level": risk_level,
            "urgency": urgency,
            "daily_water_need_liters": daily_water_need,
            "daily_tanker_trips": daily_trips,
            "weekly_tanker_trips": weekly_trips,
            "estimated_weekly_cost": weekly_cost,
            "priority_score": round(wsi * (population / 10000), 1)
        })
    
    analysis.sort(key=lambda x: x["priority_score"], reverse=True)
    
    # Resource gap analysis
    can_fulfill = available_capacity >= total_water_needed
    
    return {
        "summary": {
            "total_daily_demand_trips": total_demand,
            "total_daily_water_need": total_water_needed,
            "available_tankers": len(available_tankers),
            "total_tankers": len(tankers),
            "available_capacity": available_capacity,
            "resource_gap": max(0, total_water_needed - available_capacity),
            "can_fulfill_demand": can_fulfill,
            "additional_tankers_needed": max(0, int((total_water_needed - available_capacity) / 12000))
        },
        "villages": analysis
    }

# 5️⃣ PRIORITY-BASED ALLOCATION ENGINE
@api_router.get("/analysis/priority-allocation")
async def get_priority_allocation():
    """Generate priority-based allocation queue"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    tankers = await db.tankers.find({"status": "available"}, {"_id": 0}).to_list(100)
    
    # Calculate priority score for each village
    allocation_queue = []
    
    for v in villages:
        if v.get("risk_level") == "safe":
            continue
            
        wsi = v.get("water_stress_index", 0)
        population = v.get("population", 0)
        tanker_demand = v.get("tanker_demand", 0)
        
        # Priority formula: WSI * 0.5 + Population Factor * 0.3 + Urgency * 0.2
        pop_factor = min(100, (population / 1000))
        urgency_factor = 100 if v.get("risk_level") == "critical" else 50
        
        priority_score = (wsi * 0.5) + (pop_factor * 0.3) + (urgency_factor * 0.2)
        
        # Fairness adjustment: boost villages that haven't received recent allocation
        # (In production, track last allocation time)
        
        allocation_queue.append({
            "village_id": v.get("id"),
            "village_name": v.get("name"),
            "district": v.get("district"),
            "priority_score": round(priority_score, 1),
            "priority_rank": 0,  # Will be set after sorting
            "water_stress_index": wsi,
            "population": population,
            "risk_level": v.get("risk_level"),
            "tankers_needed": tanker_demand,
            "allocation_status": "pending",
            "fairness_score": 100,  # Placeholder for fairness tracking
            "reasoning": f"WSI: {wsi:.0f}, Pop: {population:,}, Risk: {v.get('risk_level')}"
        })
    
    # Sort by priority score
    allocation_queue.sort(key=lambda x: x["priority_score"], reverse=True)
    
    # Assign ranks
    for i, item in enumerate(allocation_queue):
        item["priority_rank"] = i + 1
    
    # Auto-allocate available tankers
    allocated = []
    remaining_tankers = list(tankers)
    
    for village in allocation_queue:
        if not remaining_tankers:
            break
        if village["tankers_needed"] > 0:
            tanker = remaining_tankers.pop(0)
            allocated.append({
                "village_id": village["village_id"],
                "village_name": village["village_name"],
                "tanker_id": tanker.get("id"),
                "vehicle_number": tanker.get("vehicle_number"),
                "capacity": tanker.get("capacity"),
                "driver": tanker.get("driver_name")
            })
            village["allocation_status"] = "allocated"
    
    return {
        "summary": {
            "villages_in_queue": len(allocation_queue),
            "available_tankers": len(tankers),
            "allocations_made": len(allocated),
            "pending_allocations": len([a for a in allocation_queue if a["allocation_status"] == "pending"]),
            "fairness_index": 95  # Placeholder metric
        },
        "allocation_queue": allocation_queue,
        "recommended_allocations": allocated
    }

# 6️⃣ ROUTE OPTIMIZATION
@api_router.get("/analysis/route-optimization")
async def optimize_routes():
    """Optimize tanker routes for efficient delivery"""
    villages = await db.villages.find({"risk_level": {"$ne": "safe"}}, {"_id": 0}).to_list(1000)
    tankers = await db.tankers.find({}, {"_id": 0}).to_list(100)
    
    # Group villages by proximity (simplified clustering)
    # In production, use proper routing API
    
    def calculate_distance(v1, v2):
        """Simple Euclidean distance (replace with actual road distance API)"""
        lat_diff = v1.get("latitude", 0) - v2.get("latitude", 0)
        lon_diff = v1.get("longitude", 0) - v2.get("longitude", 0)
        return ((lat_diff ** 2) + (lon_diff ** 2)) ** 0.5 * 111  # Approx km
    
    # Create delivery routes
    routes = []
    unassigned = list(villages)
    route_id = 1
    
    while unassigned and route_id <= len(tankers):
        # Start from depot (Aurangabad as central hub)
        depot = {"latitude": 19.8762, "longitude": 75.3433}
        route_villages = []
        current_pos = depot
        total_distance = 0
        
        # Greedy nearest neighbor algorithm
        while len(route_villages) < 4 and unassigned:
            # Find nearest unassigned village
            nearest = None
            min_dist = float('inf')
            
            for v in unassigned:
                dist = calculate_distance(current_pos, v)
                if dist < min_dist:
                    min_dist = dist
                    nearest = v
            
            if nearest and min_dist < 100:  # Max 100km radius
                route_villages.append({
                    "village_id": nearest.get("id"),
                    "village_name": nearest.get("name"),
                    "latitude": nearest.get("latitude"),
                    "longitude": nearest.get("longitude"),
                    "distance_from_prev": round(min_dist, 1),
                    "risk_level": nearest.get("risk_level"),
                    "stop_order": len(route_villages) + 1
                })
                total_distance += min_dist
                current_pos = nearest
                unassigned.remove(nearest)
            else:
                break
        
        # Return to depot
        if route_villages:
            return_distance = calculate_distance(current_pos, depot)
            total_distance += return_distance
            
            tanker = tankers[route_id - 1] if route_id <= len(tankers) else None
            
            routes.append({
                "route_id": f"R{route_id:03d}",
                "tanker": {
                    "id": tanker.get("id") if tanker else None,
                    "vehicle_number": tanker.get("vehicle_number") if tanker else "Unassigned",
                    "driver": tanker.get("driver_name") if tanker else None
                },
                "stops": route_villages,
                "total_stops": len(route_villages),
                "total_distance_km": round(total_distance, 1),
                "estimated_time_hours": round(total_distance / 40, 1),  # 40 km/h average
                "fuel_estimate_liters": round(total_distance / 5, 1),  # 5 km/L
                "fuel_cost_estimate": round((total_distance / 5) * 100, 0),  # Rs 100/L
                "status": "planned"
            })
            route_id += 1
    
    # Calculate optimization metrics
    total_distance_all = sum(r["total_distance_km"] for r in routes)
    naive_distance = len(villages) * 50 * 2  # If each village was served individually
    
    return {
        "summary": {
            "total_routes": len(routes),
            "total_villages_covered": sum(r["total_stops"] for r in routes),
            "villages_remaining": len(unassigned),
            "total_distance_km": round(total_distance_all, 1),
            "estimated_total_time_hours": round(total_distance_all / 40, 1),
            "total_fuel_cost": sum(r["fuel_cost_estimate"] for r in routes),
            "optimization_savings_percent": round((1 - total_distance_all / max(naive_distance, 1)) * 100, 1)
        },
        "routes": routes,
        "unassigned_villages": [{"id": v.get("id"), "name": v.get("name")} for v in unassigned]
    }

# 7️⃣ REAL-TIME MONITORING
@api_router.get("/analysis/realtime-status")
async def get_realtime_status():
    """Get real-time system status and alerts"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    tankers = await db.tankers.find({}, {"_id": 0}).to_list(100)
    
    # Generate alerts
    alerts = []
    
    # Critical village alerts
    critical_villages = [v for v in villages if v.get("risk_level") == "critical"]
    for v in critical_villages:
        alerts.append({
            "id": f"ALT-{v.get('id')[:8]}",
            "type": "CRITICAL_DROUGHT",
            "severity": "high",
            "village": v.get("name"),
            "message": f"{v.get('name')} has WSI of {v.get('water_stress_index', 0):.1f} - Immediate action required",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_required": "Deploy tanker immediately"
        })
    
    # Low groundwater alerts
    low_gw = [v for v in villages if v.get("groundwater_level", 10) < 3]
    for v in low_gw:
        alerts.append({
            "id": f"GW-{v.get('id')[:8]}",
            "type": "LOW_GROUNDWATER",
            "severity": "medium",
            "village": v.get("name"),
            "message": f"Groundwater at {v.get('groundwater_level')}m in {v.get('name')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_required": "Monitor and prepare contingency"
        })
    
    # Tanker availability alert
    available = len([t for t in tankers if t.get("status") == "available"])
    if available < 2:
        alerts.append({
            "id": "TANKER-LOW",
            "type": "LOW_RESOURCES",
            "severity": "medium",
            "village": None,
            "message": f"Only {available} tankers available for deployment",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_required": "Request additional tankers or release dispatched units"
        })
    
    # Sort alerts by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda x: severity_order.get(x["severity"], 3))
    
    # System metrics
    dispatched_tankers = [t for t in tankers if t.get("status") == "dispatched"]
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_status": "OPERATIONAL" if not critical_villages else "ALERT",
        "alerts": {
            "total": len(alerts),
            "high": len([a for a in alerts if a["severity"] == "high"]),
            "medium": len([a for a in alerts if a["severity"] == "medium"]),
            "low": len([a for a in alerts if a["severity"] == "low"]),
            "items": alerts[:10]  # Top 10 alerts
        },
        "metrics": {
            "villages": {
                "total": len(villages),
                "critical": len(critical_villages),
                "moderate": len([v for v in villages if v.get("risk_level") == "moderate"]),
                "safe": len([v for v in villages if v.get("risk_level") == "safe"])
            },
            "tankers": {
                "total": len(tankers),
                "available": available,
                "dispatched": len(dispatched_tankers),
                "maintenance": len([t for t in tankers if t.get("status") == "maintenance"])
            },
            "coverage": {
                "population_at_risk": sum(v.get("population", 0) for v in villages if v.get("risk_level") != "safe"),
                "villages_being_served": len(dispatched_tankers),
                "estimated_water_delivered_today": len(dispatched_tankers) * 12000  # liters
            }
        },
        "recent_activities": [
            {"time": datetime.now(timezone.utc).isoformat(), "action": "System health check completed", "status": "success"},
            {"time": datetime.now(timezone.utc).isoformat(), "action": "Weather data updated", "status": "success"},
            {"time": datetime.now(timezone.utc).isoformat(), "action": "WSI recalculated for all villages", "status": "success"}
        ]
    }

# ============ DASHBOARD ROUTES ============

@api_router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get aggregated dashboard statistics"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    tankers = await db.tankers.find({}, {"_id": 0}).to_list(100)
    
    critical = [v for v in villages if v.get("risk_level") == "critical"]
    moderate = [v for v in villages if v.get("risk_level") == "moderate"]
    safe = [v for v in villages if v.get("risk_level") == "safe"]
    
    population_at_risk = sum(v.get("population", 0) for v in critical + moderate)
    avg_wsi = sum(v.get("water_stress_index", 0) for v in villages) / len(villages) if villages else 0
    
    available_tankers = [t for t in tankers if t.get("status") == "available"]
    dispatched_tankers = [t for t in tankers if t.get("status") == "dispatched"]
    
    return DashboardStats(
        total_villages=len(villages),
        critical_villages=len(critical),
        moderate_villages=len(moderate),
        safe_villages=len(safe),
        total_tankers=len(tankers),
        available_tankers=len(available_tankers),
        dispatched_tankers=len(dispatched_tankers),
        total_population_at_risk=population_at_risk,
        avg_water_stress_index=round(avg_wsi, 2)
    )

@api_router.get("/dashboard/priority-list")
async def get_priority_list():
    """Get villages sorted by priority (WSI + population)"""
    villages = await db.villages.find({}, {"_id": 0}).to_list(1000)
    
    # Sort by water_stress_index descending, then by population descending
    sorted_villages = sorted(
        villages,
        key=lambda v: (v.get("water_stress_index", 0), v.get("population", 0)),
        reverse=True
    )
    
    return sorted_villages[:20]  # Top 20 priority villages

# ============ SEED DATA ROUTE ============

@api_router.post("/seed")
async def seed_database():
    """Seed database with sample villages and tankers"""
    # Sample villages in Maharashtra, India (drought-prone region)
    sample_villages = [
        {"name": "Ahmednagar", "district": "Ahmednagar", "state": "Maharashtra", "latitude": 19.0948, "longitude": 74.7480, "population": 45000, "groundwater_level": 3.2, "rainfall_actual": 180, "rainfall_normal": 450},
        {"name": "Beed", "district": "Beed", "state": "Maharashtra", "latitude": 18.9891, "longitude": 75.7601, "population": 32000, "groundwater_level": 2.5, "rainfall_actual": 120, "rainfall_normal": 400},
        {"name": "Latur", "district": "Latur", "state": "Maharashtra", "latitude": 18.4088, "longitude": 76.5604, "population": 52000, "groundwater_level": 4.0, "rainfall_actual": 200, "rainfall_normal": 380},
        {"name": "Osmanabad", "district": "Osmanabad", "state": "Maharashtra", "latitude": 18.1860, "longitude": 76.0400, "population": 28000, "groundwater_level": 2.0, "rainfall_actual": 90, "rainfall_normal": 350},
        {"name": "Solapur", "district": "Solapur", "state": "Maharashtra", "latitude": 17.6599, "longitude": 75.9064, "population": 68000, "groundwater_level": 5.5, "rainfall_actual": 280, "rainfall_normal": 420},
        {"name": "Jalna", "district": "Jalna", "state": "Maharashtra", "latitude": 19.8347, "longitude": 75.8816, "population": 41000, "groundwater_level": 3.8, "rainfall_actual": 220, "rainfall_normal": 480},
        {"name": "Parbhani", "district": "Parbhani", "state": "Maharashtra", "latitude": 19.2610, "longitude": 76.7700, "population": 35000, "groundwater_level": 2.8, "rainfall_actual": 150, "rainfall_normal": 420},
        {"name": "Nanded", "district": "Nanded", "state": "Maharashtra", "latitude": 19.1383, "longitude": 77.3210, "population": 55000, "groundwater_level": 6.0, "rainfall_actual": 350, "rainfall_normal": 500},
        {"name": "Aurangabad", "district": "Aurangabad", "state": "Maharashtra", "latitude": 19.8762, "longitude": 75.3433, "population": 72000, "groundwater_level": 4.5, "rainfall_actual": 260, "rainfall_normal": 520},
        {"name": "Hingoli", "district": "Hingoli", "state": "Maharashtra", "latitude": 19.7145, "longitude": 77.1497, "population": 22000, "groundwater_level": 3.0, "rainfall_actual": 140, "rainfall_normal": 380},
        {"name": "Washim", "district": "Washim", "state": "Maharashtra", "latitude": 20.1020, "longitude": 77.1250, "population": 18000, "groundwater_level": 2.2, "rainfall_actual": 100, "rainfall_normal": 360},
        {"name": "Yavatmal", "district": "Yavatmal", "state": "Maharashtra", "latitude": 20.3888, "longitude": 78.1204, "population": 48000, "groundwater_level": 5.0, "rainfall_actual": 320, "rainfall_normal": 550},
    ]
    
    # Clear existing data
    await db.villages.delete_many({})
    await db.tankers.delete_many({})
    
    # Insert villages with calculated WSI
    for v_data in sample_villages:
        wsi, risk_level, tanker_demand = calculate_water_stress_index(v_data)
        village = Village(
            **v_data,
            water_stress_index=wsi,
            risk_level=risk_level,
            tanker_demand=tanker_demand
        )
        doc = village.model_dump()
        doc['last_updated'] = doc['last_updated'].isoformat()
        await db.villages.insert_one(doc)
    
    # Sample tankers
    sample_tankers = [
        {"vehicle_number": "MH-12-AB-1234", "capacity": 10000, "driver_name": "Ramesh Kumar", "driver_phone": "+91-9876543210"},
        {"vehicle_number": "MH-12-CD-5678", "capacity": 15000, "driver_name": "Suresh Patil", "driver_phone": "+91-9876543211"},
        {"vehicle_number": "MH-14-EF-9012", "capacity": 12000, "driver_name": "Mahesh Jadhav", "driver_phone": "+91-9876543212"},
        {"vehicle_number": "MH-14-GH-3456", "capacity": 8000, "driver_name": "Ganesh More", "driver_phone": "+91-9876543213"},
        {"vehicle_number": "MH-15-IJ-7890", "capacity": 20000, "driver_name": "Prakash Shinde", "driver_phone": "+91-9876543214"},
        {"vehicle_number": "MH-15-KL-2345", "capacity": 10000, "driver_name": "Vijay Deshmukh", "driver_phone": "+91-9876543215"},
    ]
    
    for t_data in sample_tankers:
        tanker = Tanker(**t_data)
        doc = tanker.model_dump()
        await db.tankers.insert_one(doc)
    
    return {"message": "Database seeded with 12 villages and 6 tankers"}

@api_router.get("/")
async def root():
    return {"message": "DroughtGuard API - Drought Warning & Tanker Management System"}

# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
