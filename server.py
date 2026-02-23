@api_router.post(\"/tankers/{tanker_id}/release\")
async def release_tanker(tanker_id: str):
    \"\"\"Release a tanker back to available\"\"\"
    result = await db.tankers.update_one(
        {\"id\": tanker_id},
        {\"$set\": {
            \"status\": \"available\",
            \"assigned_village_id\": None,
            \"assigned_village_name\": None
        }}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=\"Tanker not found\")
    return {\"message\": \"Tanker released successfully\"}

# ============ WEATHER ROUTES (Open-Meteo - FREE, No API Key) ============

@api_router.get(\"/weather/{village_id}\")
async def get_weather(village_id: str):
    \"\"\"Fetch current weather data using Open-Meteo API (FREE)\"\"\"
    village = await db.villages.find_one({\"id\": village_id}, {\"_id\": 0})
    if not village:
        raise HTTPException(status_code=404, detail=\"Village not found\")
    
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f\"{OPEN_METEO_BASE}/forecast\",
                params={
                    \"latitude\": village[\"latitude\"],
                    \"longitude\": village[\"longitude\"],
                    \"current\": \"temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m\",
                    \"timezone\": \"auto\"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            current = data.get(\"current\", {})
            weather_code = current.get(\"weather_code\", 0)
            
            # Weather code to description mapping
            weather_descriptions = {
                0: \"Clear sky\", 1: \"Mainly clear\", 2: \"Partly cloudy\", 3: \"Overcast\",
                45: \"Foggy\", 48: \"Depositing rime fog\",
                51: \"Light drizzle\", 53: \"Moderate drizzle\", 55: \"Dense drizzle\",
                61: \"Slight rain\", 63: \"Moderate rain\", 65: \"Heavy rain\",
                71: \"Slight snow\", 73: \"Moderate snow\", 75: \"Heavy snow\",
                80: \"Slight rain showers\", 81: \"Moderate rain showers\", 82: \"Violent rain showers\",
                95: \"Thunderstorm\", 96: \"Thunderstorm with slight hail\", 99: \"Thunderstorm with heavy hail\"
            }
            
            return {
                \"village_id\": village_id,
                \"village_name\": village[\"name\"],
                \"temperature\": current.get(\"temperature_2m\", 0),
                \"humidity\": current.get(\"relative_humidity_2m\", 0),
                \"rainfall\": current.get(\"precipitation\", 0),
                \"wind_speed\": current.get(\"wind_speed_10m\", 0),
                \"weather_code\": weather_code,
                \"description\": weather_descriptions.get(weather_code, \"Unknown\"),
                \"source\": \"open-meteo\",
                \"fetched_at\": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f\"Open-Meteo API failed: {e}\")
        raise HTTPException(status_code=502, detail=f\"Weather API error: {str(e)}\")

@api_router.get(\"/weather/{village_id}/historical\")
async def get_historical_weather(village_id: str, days: int = Query(30, ge=7, le=90)):
    \"\"\"Fetch historical weather data using Open-Meteo Historical API (FREE)\"\"\"
    village = await db.villages.find_one({\"id\": village_id}, {\"_id\": 0})
    if not village:
        raise HTTPException(status_code=404, detail=\"Village not found\")
    
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f\"{OPEN_METEO_BASE}/forecast\",
                params={
                    \"latitude\": village[\"latitude\"],
                    \"longitude\": village[\"longitude\"],
                    \"daily\": \"temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum\",
                    \"past_days\": days,
                    \"timezone\": \"auto\"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            daily = data.get(\"daily\", {})
            dates = daily.get(\"time\", [])
            
            history = []
            for i, date in enumerate(dates):
                history.append({
                    \"date\": date,
                    \"temp_max\": daily.get(\"temperature_2m_max\", [None])[i] if i < len(daily.get(\"temperature_2m_max\", [])) else None,
                    \"temp_min\": daily.get(\"temperature_2m_min\", [None])[i] if i < len(daily.get(\"temperature_2m_min\", [])) else None,
                    \"precipitation\": daily.get(\"precipitation_sum\", [0])[i] if i < len(daily.get(\"precipitation_sum\", [])) else 0,
                    \"rain\": daily.get(\"rain_sum\", [0])[i] if i < len(daily.get(\"rain_sum\", [])) else 0
                })
            
            # Calculate summary
            total_rainfall = sum(h.get(\"precipitation\", 0) or 0 for h in history)
            avg_temp = sum((h.get(\"temp_max\", 0) or 0 + h.get(\"temp_min\", 0) or 0) / 2 for h in history) / len(history) if history else 0
            
            return {
                \"village_id\": village_id,
                \"village_name\": village[\"name\"],
                \"period_days\": days,
                \"start_date\": str(start_date),
                \"end_date\": str(end_date),
                \"summary\": {
                    \"total_rainfall_mm\": round(total_rainfall, 1),
                    \"average_temperature\": round(avg_temp, 1),
                    \"rainy_days\": len([h for h in history if (h.get(\"precipitation\") or 0) > 1])
                },
                \"daily_data\": history,
                \"source\": \"open-meteo\"
            }
    except Exception as e:
        logger.error(f\"Open-Meteo Historical API failed: {e}\")
        raise HTTPException(status_code=502, detail=f\"Historical weather API error: {str(e)}\")

# ============ HISTORICAL DATA TRACKING ============

@api_router.post(\"/historical/record\")
async def record_historical_data():
    \"\"\"Record current data snapshot for all villages (call daily via cron)\"\"\"
    villages = await db.villages.find({}, {\"_id\": 0}).to_list(1000)
    today = datetime.now(timezone.utc).strftime(\"%Y-%m-%d\")
    
    records_created = 0
    for village in villages:
        # Check if already recorded today
        existing = await db.historical_records.find_one({
            \"village_id\": village[\"id\"],
            \"date\": today
        })
        
        if not existing:
            record = HistoricalRecord(
                village_id=village[\"id\"],
                village_name=village[\"name\"],
                date=today,
                water_stress_index=village.get(\"water_stress_index\", 0),
                rainfall_actual=village.get(\"rainfall_actual\", 0),
                groundwater_level=village.get(\"groundwater_level\", 0),
                risk_level=village.get(\"risk_level\", \"safe\")
            )
            doc = record.model_dump()
            doc['created_at'] = doc['created_at'].isoformat()
            await db.historical_records.insert_one(doc)
            records_created += 1
    
    return {\"message\": f\"Recorded {records_created} historical snapshots for {today}\"}

@api_router.get(\"/historical/{village_id}\")
async def get_village_historical_data(village_id: str, days: int = Query(30, ge=7, le=365)):
    \"\"\"Get historical WSI and conditions for a village\"\"\"
    village = await db.villages.find_one({\"id\": village_id}, {\"_id\": 0})
    if not village:
        raise HTTPException(status_code=404, detail=\"Village not found\")
    
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(\"%Y-%m-%d\")
    
    records = await db.historical_records.find(
        {\"village_id\": village_id, \"date\": {\"$gte\": start_date}},
        {\"_id\": 0}
    ).sort(\"date\", 1).to_list(365)
    
    if not records:
        return {
            \"village_id\": village_id,
            \"village_name\": village[\"name\"],
            \"message\": \"No historical data available. Run /api/historical/record to start tracking.\",
            \"records\": []
        }
    
    # Calculate trends
    if len(records) >= 2:
        first_wsi = records[0].get(\"water_stress_index\", 0)
        last_wsi = records[-1].get(\"water_stress_index\", 0)
        wsi_trend = \"increasing\" if last_wsi > first_wsi else \"decreasing\" if last_wsi < first_wsi else \"stable\"
        wsi_change = round(last_wsi - first_wsi, 1)
    else:
        wsi_trend = \"insufficient_data\"
        wsi_change = 0
    
    return {
        \"village_id\": village_id,
        \"village_name\": village[\"name\"],
        \"period_days\": days,
        \"total_records\": len(records),
        \"trend_analysis\": {
            \"wsi_trend\": wsi_trend,
            \"wsi_change\": wsi_change,
            \"average_wsi\": round(sum(r.get(\"water_stress_index\", 0) for r in records) / len(records), 1),
            \"max_wsi\": max(r.get(\"water_stress_index\", 0) for r in records),
            \"min_wsi\": min(r.get(\"water_stress_index\", 0) for r in records)
        },
        \"records\": records
    }

@api_router.get(\"/historical/trends/all\")
async def get_all_villages_trends(days: int = Query(30, ge=7, le=365)):
    \"\"\"Get historical trends for all villages\"\"\"
    villages = await db.villages.find({}, {\"_id\": 0}).to_list(1000)
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(\"%Y-%m-%d\")
    
    trends = []
    for village in villages:
        records = await db.historical_records.find(
            {\"village_id\": village[\"id\"], \"date\": {\"$gte\": start_date}},
            {\"_id\": 0}
        ).sort(\"date\", 1).to_list(365)
        
        if len(records) >= 2:
            first_wsi = records[0].get(\"water_stress_index\", 0)
            last_wsi = records[-1].get(\"water_stress_index\", 0)
            wsi_change = round(last_wsi - first_wsi, 1)
            trend = \"worsening\" if wsi_change > 5 else \"improving\" if wsi_change < -5 else \"stable\"
        else:
            wsi_change = 0
            trend = \"no_data\"
        
        trends.append({
            \"village_id\": village[\"id\"],
            \"village_name\": village[\"name\"],
            \"current_wsi\": village.get(\"water_stress_index\", 0),
            \"current_risk\": village.get(\"risk_level\", \"safe\"),
            \"wsi_change\": wsi_change,
            \"trend\": trend,
            \"data_points\": len(records)
        })
    
    # Sort by WSI change (worst first)
    trends.sort(key=lambda x: x[\"wsi_change\"], reverse=True)
    
    return {
        \"period_days\": days,
        \"total_villages\": len(trends),
        \"worsening_count\": len([t for t in trends if t[\"trend\"] == \"worsening\"]),
        \"improving_count\": len([t for t in trends if t[\"trend\"] == \"improving\"]),
        \"stable_count\": len([t for t in trends if t[\"trend\"] == \"stable\"]),
        \"villages\": trends
    }

@api_router.post(\"/historical/seed\")
async def seed_historical_data():
    \"\"\"Seed historical data for demo purposes (last 30 days)\"\"\"
    import random
    
    villages = await db.villages.find({}, {\"_id\": 0}).to_list(1000)
    
    # Clear existing historical data
    await db.historical_records.delete_many({})
    
    records_created = 0
    for village in villages:
        base_wsi = village.get(\"water_stress_index\", 50)
        base_rainfall = village.get(\"rainfall_actual\", 200)
        base_gw = village.get(\"groundwater_level\", 5)
        
        for day_offset in range(30, -1, -1):
            date = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime(\"%Y-%m-%d\")
            
            # Simulate gradual changes
            variation = (30 - day_offset) / 30  # 0 to 1 over 30 days
            wsi = base_wsi - (variation * random.uniform(5, 15))  # WSI was higher in past
            wsi = max(0, min(100, wsi + random.uniform(-3, 3)))
            
            rainfall = base_rainfall * (1 - variation * 0.3) + random.uniform(-20, 20)
            gw_level = base_gw + (variation * random.uniform(-0.5, 0.5))
            
            risk = \"critical\" if wsi >= 70 else \"moderate\" if wsi >= 40 else \"safe\"
            
            record = HistoricalRecord(
                village_id=village[\"id\"],
                village_name=village[\"name\"],
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
    
    return {\"message\": f\"Seeded {records_created} historical records for {len(villages)} villages (30 days)\"}

# ============ ANALYSIS MODULES ============
