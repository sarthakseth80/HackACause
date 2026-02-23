import requests
import sys
import json
from datetime import datetime
from typing import Dict, Any

class DroughtGuardAPITester:
    def __init__(self, base_url="https://drought-command.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = {}

    def run_test(self, name: str, method: str, endpoint: str, expected_status: int, data: Dict[Any, Any] = None) -> tuple:
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=15)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=15)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=15)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=15)

            success = response.status_code == expected_status
            
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, list):
                        print(f"   Response: List with {len(response_data)} items")
                    elif isinstance(response_data, dict):
                        print(f"   Response keys: {list(response_data.keys())}")
                except:
                    print(f"   Response: {response.text[:100]}...")
                self.test_results[name] = {"status": "PASS", "details": f"Status {response.status_code}"}
                return success, response.json() if response.text else {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                self.test_results[name] = {"status": "FAIL", "details": f"Expected {expected_status}, got {response.status_code}"}
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout")
            self.test_results[name] = {"status": "FAIL", "details": "Request timeout"}
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.test_results[name] = {"status": "FAIL", "details": str(e)}
            return False, {}

    def test_api_root(self):
        """Test API root endpoint"""
        return self.run_test("API Root", "GET", "", 200)

    def test_seed_database(self):
        """Test seeding database with sample data"""
        success, response = self.run_test("Seed Database", "POST", "seed", 200)
        if success and "message" in response:
            print(f"   Seed message: {response['message']}")
        return success

    def test_get_villages(self):
        """Test getting all villages"""
        success, response = self.run_test("Get Villages", "GET", "villages", 200)
        if success and isinstance(response, list):
            print(f"   Found {len(response)} villages")
            if len(response) > 0:
                village = response[0]
                required_fields = ['id', 'name', 'district', 'latitude', 'longitude', 'water_stress_index', 'risk_level']
                for field in required_fields:
                    if field not in village:
                        print(f"   ⚠️  Missing field: {field}")
        return success

    def test_get_villages_with_filter(self):
        """Test filtering villages by risk level"""
        success, response = self.run_test("Get Critical Villages", "GET", "villages?risk_level=critical", 200)
        if success and isinstance(response, list):
            print(f"   Found {len(response)} critical villages")
            for village in response:
                if village.get('risk_level') != 'critical':
                    print(f"   ⚠️  Non-critical village in results: {village.get('name')}")
                    break
        return success

    def test_get_tankers(self):
        """Test getting all tankers"""
        success, response = self.run_test("Get Tankers", "GET", "tankers", 200)
        if success and isinstance(response, list):
            print(f"   Found {len(response)} tankers")
            if len(response) > 0:
                tanker = response[0]
                required_fields = ['id', 'vehicle_number', 'capacity', 'status', 'driver_name']
                for field in required_fields:
                    if field not in tanker:
                        print(f"   ⚠️  Missing field: {field}")
        return success

    def test_dashboard_stats(self):
        """Test dashboard statistics endpoint"""
        success, response = self.run_test("Dashboard Stats", "GET", "dashboard/stats", 200)
        if success and isinstance(response, dict):
            required_stats = ['total_villages', 'critical_villages', 'total_tankers', 'available_tankers']
            for stat in required_stats:
                if stat not in response:
                    print(f"   ⚠️  Missing stat: {stat}")
                else:
                    print(f"   {stat}: {response[stat]}")
        return success

    def test_priority_list(self):
        """Test priority villages list"""
        success, response = self.run_test("Priority List", "GET", "dashboard/priority-list", 200)
        if success and isinstance(response, list):
            print(f"   Priority villages: {len(response)}")
            if len(response) > 0:
                # Check if sorted by WSI
                wsi_values = [v.get('water_stress_index', 0) for v in response[:5]]
                print(f"   Top 5 WSI values: {wsi_values}")
        return success

    def test_tanker_allocation_flow(self):
        """Test complete tanker allocation workflow"""
        # First get available tankers and villages
        villages_success, villages = self.run_test("Villages for Allocation", "GET", "villages", 200)
        tankers_success, tankers = self.run_test("Tankers for Allocation", "GET", "tankers", 200)
        
        if not (villages_success and tankers_success):
            print("   ❌ Failed to get initial data for allocation test")
            return False

        # Find an available tanker and a village needing help
        available_tanker = None
        for tanker in tankers:
            if tanker.get('status') == 'available':
                available_tanker = tanker
                break
        
        target_village = None
        for village in villages:
            if village.get('risk_level') in ['critical', 'moderate']:
                target_village = village
                break

        if not available_tanker or not target_village:
            print("   ⚠️  No available tanker or village needing help found")
            return True  # Not a test failure, just no data to test with

        # Test allocation
        allocation_data = {
            "tanker_id": available_tanker['id'],
            "village_id": target_village['id']
        }
        
        alloc_success, alloc_response = self.run_test(
            "Allocate Tanker", 
            "POST", 
            "tankers/allocate", 
            200, 
            allocation_data
        )
        
        if alloc_success:
            print(f"   Allocated {available_tanker['vehicle_number']} to {target_village['name']}")
            
            # Test release
            release_success, release_response = self.run_test(
                "Release Tanker",
                "POST",
                f"tankers/{available_tanker['id']}/release",
                200
            )
            
            if release_success:
                print(f"   Released tanker {available_tanker['vehicle_number']}")
            
            return release_success
        
        return alloc_success

    def test_weather_api(self):
        """Test weather API integration"""
        # First get a village
        villages_success, villages = self.run_test("Villages for Weather", "GET", "villages", 200)
        if not villages_success or len(villages) == 0:
            print("   ❌ No villages available for weather test")
            return False
        
        village = villages[0]
        weather_success, weather_response = self.run_test(
            "Weather Data",
            "GET",
            f"weather/{village['id']}",
            200
        )
        
        if weather_success:
            required_weather_fields = ['temperature', 'humidity', 'description']
            for field in required_weather_fields:
                if field not in weather_response:
                    print(f"   ⚠️  Missing weather field: {field}")
                else:
                    print(f"   {field}: {weather_response[field]}")
        
        return weather_success

    def test_create_village(self):
        """Test creating a new village"""
        test_village = {
            "name": "Test Village",
            "district": "Test District",
            "state": "Maharashtra",
            "latitude": 19.5,
            "longitude": 75.5,
            "population": 1000,
            "groundwater_level": 3.0,
            "rainfall_actual": 150,
            "rainfall_normal": 400
        }
        
        success, response = self.run_test("Create Village", "POST", "villages", 200, test_village)
        
        if success and 'id' in response:
            village_id = response['id']
            print(f"   Created village with ID: {village_id}")
            
            # Test delete
            delete_success, _ = self.run_test(
                "Delete Test Village",
                "DELETE",
                f"villages/{village_id}",
                200
            )
            
            if delete_success:
                print(f"   Cleaned up test village")
            
            return delete_success
        
        return success

    # ====== ANALYSIS MODULES TESTING ======
    
    def test_rainfall_analysis(self):
        """Test Rainfall Deviation Analyzer"""
        success, response = self.run_test("Rainfall Analysis Module", "GET", "analysis/rainfall", 200)
        if success and isinstance(response, dict):
            if 'summary' in response and 'villages' in response:
                summary = response['summary']
                print(f"   📊 Total villages analyzed: {summary.get('total_villages', 0)}")
                print(f"   📊 Villages in deficit: {summary.get('deficit_villages', 0)}")
                print(f"   📊 Drought prediction: {summary.get('drought_prediction', 'Unknown')}")
                print(f"   📊 Village data entries: {len(response.get('villages', []))}")
            else:
                print("   ⚠️  Missing expected response structure (summary/villages)")
        return success

    def test_groundwater_analysis(self):
        """Test Groundwater Trend Analyzer"""
        success, response = self.run_test("Groundwater Analysis Module", "GET", "analysis/groundwater", 200)
        if success and isinstance(response, dict):
            if 'summary' in response and 'villages' in response:
                summary = response['summary']
                print(f"   📊 Villages analyzed: {summary.get('total_villages', 0)}")
                print(f"   📊 Critical level count: {summary.get('critical_level', 0)}")
                print(f"   📊 Alert status: {summary.get('alert_status', 'Unknown')}")
                print(f"   📊 Village trend data: {len(response.get('villages', []))}")
            else:
                print("   ⚠️  Missing expected response structure")
        return success

    def test_stress_index_analysis(self):
        """Test Water Stress Index Generator"""
        success, response = self.run_test("Water Stress Index Module", "GET", "analysis/stress-index", 200)
        if success and isinstance(response, dict):
            if 'summary' in response and 'villages' in response:
                summary = response['summary']
                print(f"   📊 Villages indexed: {summary.get('total_villages', 0)}")
                print(f"   📊 Critical villages: {summary.get('critical_count', 0)}")
                print(f"   📊 Average WSI: {summary.get('average_wsi', 0)}")
                print(f"   📊 Most affected: {summary.get('most_affected', 'None')}")
            else:
                print("   ⚠️  Missing expected response structure")
        return success

    def test_tanker_demand_prediction(self):
        """Test Tanker Demand Predictor"""
        success, response = self.run_test("Tanker Demand Predictor", "GET", "analysis/tanker-demand", 200)
        if success and isinstance(response, dict):
            if 'summary' in response and 'villages' in response:
                summary = response['summary']
                print(f"   📊 Daily demand trips: {summary.get('total_daily_demand_trips', 0)}")
                print(f"   📊 Available tankers: {summary.get('available_tankers', 0)}")
                print(f"   📊 Can fulfill demand: {summary.get('can_fulfill_demand', False)}")
                print(f"   📊 Village demand data: {len(response.get('villages', []))}")
            else:
                print("   ⚠️  Missing expected response structure")
        return success

    def test_priority_allocation(self):
        """Test Priority-Based Allocation Engine"""
        success, response = self.run_test("Priority Allocation Engine", "GET", "analysis/priority-allocation", 200)
        if success and isinstance(response, dict):
            if 'allocation_queue' in response and 'recommended_allocations' in response:
                summary = response.get('summary', {})
                print(f"   📊 Villages in queue: {summary.get('villages_in_queue', 0)}")
                print(f"   📊 Available tankers: {summary.get('available_tankers', 0)}")
                print(f"   📊 Allocations made: {summary.get('allocations_made', 0)}")
                print(f"   📊 Queue entries: {len(response.get('allocation_queue', []))}")
            else:
                print("   ⚠️  Missing expected response structure")
        return success

    def test_route_optimization(self):
        """Test Route Optimization System"""
        success, response = self.run_test("Route Optimization", "GET", "analysis/route-optimization", 200)
        if success and isinstance(response, dict):
            if 'routes' in response and 'summary' in response:
                summary = response['summary']
                print(f"   📊 Total routes: {summary.get('total_routes', 0)}")
                print(f"   📊 Villages covered: {summary.get('total_villages_covered', 0)}")
                print(f"   📊 Total distance: {summary.get('total_distance_km', 0)} km")
                print(f"   📊 Optimization savings: {summary.get('optimization_savings_percent', 0)}%")
            else:
                print("   ⚠️  Missing expected response structure")
        return success

    def test_realtime_monitoring(self):
        """Test Real-Time Monitoring Dashboard"""
        success, response = self.run_test("Real-Time Monitoring", "GET", "analysis/realtime-status", 200)
        if success and isinstance(response, dict):
            if 'system_status' in response and 'alerts' in response and 'metrics' in response:
                alerts = response.get('alerts', {})
                metrics = response.get('metrics', {})
                print(f"   📊 System status: {response.get('system_status', 'Unknown')}")
                print(f"   📊 Total alerts: {alerts.get('total', 0)}")
                print(f"   📊 High priority alerts: {alerts.get('high', 0)}")
                print(f"   📊 Villages monitored: {metrics.get('villages', {}).get('total', 0)}")
            else:
                print("   ⚠️  Missing expected response structure")
        return success

    def run_all_tests(self):
        """Run all API tests"""
        print("=" * 60)
        print("🚀 DROUGHTGUARD API TESTING SUITE")
        print("=" * 60)
        
        # Test sequence - Basic API tests first
        basic_tests = [
            self.test_api_root,
            self.test_seed_database,
            self.test_get_villages,
            self.test_get_villages_with_filter,
            self.test_get_tankers,
            self.test_dashboard_stats,
            self.test_priority_list,
            self.test_weather_api,
            self.test_tanker_allocation_flow,
            self.test_create_village
        ]
        
        # Analysis module tests
        analysis_tests = [
            self.test_rainfall_analysis,
            self.test_groundwater_analysis,
            self.test_stress_index_analysis,
            self.test_tanker_demand_prediction,
            self.test_priority_allocation,
            self.test_route_optimization,
            self.test_realtime_monitoring
        ]
        
        print("\n🔧 Testing Basic API Functionality...")
        for test in basic_tests:
            try:
                test()
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
                self.test_results[test.__name__] = {"status": "ERROR", "details": str(e)}
        
        print("\n📊 Testing Analysis Modules...")
        for test in analysis_tests:
            try:
                test()
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
                self.test_results[test.__name__] = {"status": "ERROR", "details": str(e)}
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        print(f"\n📝 DETAILED RESULTS:")
        passed_tests = []
        failed_tests = []
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_emoji} {test_name}: {result['status']} - {result['details']}")
            if result["status"] == "PASS":
                passed_tests.append(test_name)
            else:
                failed_tests.append({"test": test_name, "error": result["details"]})
        
        # Save results for test report
        with open('/app/backend_test_results.json', 'w') as f:
            import json
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_tests': self.tests_run,
                'passed_tests': self.tests_passed,
                'failed_tests': len(failed_tests),
                'success_rate': round((self.tests_passed/self.tests_run)*100, 1),
                'passed_test_names': passed_tests,
                'failed_test_details': failed_tests
            }, f, indent=2)
        
        return self.tests_passed == self.tests_run

def main():
    """Main test runner"""
    tester = DroughtGuardAPITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {tester.tests_run - tester.tests_passed} tests failed. Check the details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())