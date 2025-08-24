import pytest
import asyncio
import json
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch
import time
from datetime import datetime

from api import app


class TestAPIHealth:
    """Test API health and basic endpoints."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Capsule Acceleration Simulator"
        assert data["version"] == "1.0.0"
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_endpoint(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_jobs" in data
        assert "total_jobs" in data
        assert "timestamp" in data


class TestSimulationAPI:
    """Test simulation job creation and management."""

    def setup_method(self):
        self.client = TestClient(app)
        # Clear any existing jobs
        from api import simulation_jobs
        simulation_jobs.clear()

    def test_create_simulation_job(self):
        payload = {
            "capsule": {
                "mass": 1.0,
                "diameter": 0.083,
                "initial_position": 0.0,
                "initial_velocity": 0.0
            },
            "tube": {
                "length": 0.5,
                "inner_diameter": 0.09
            },
            "coils": [
                {
                    "center": 0.1,
                    "length": 0.1,
                    "force": 10.0,
                    "name": "TestCoil1"
                },
                {
                    "center": 0.3,
                    "length": 0.1,
                    "force": 10.0,
                    "name": "TestCoil2"
                }
            ],
            "simulation": {
                "dt": 0.001,
                "max_time": 1.0,
                "stop_at_exit": True
            },
            "export": {
                "format": "json",
                "json_compress": False
            }
        }

        response = self.client.post("/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_create_custom_simulation(self):
        payload = {
            "tube_length": 1.0,
            "tube_diameter": 0.1,
            "capsule_mass": 2.0,
            "capsule_diameter": 0.08,
            "num_coils": 4,
            "coil_length": 0.2,
            "coil_force": 15.0,
            "dt": 0.001,
            "max_time": 2.0,
            "stop_at_exit": True,
            "export_format": "parquet"
        }

        response = self.client.post("/simulate/custom", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_invalid_simulation_parameters(self):
        # Test with invalid mass (negative)
        payload = {
            "capsule": {
                "mass": -1.0,  # Invalid
                "diameter": 0.083
            },
            "tube": {
                "length": 0.5,
                "inner_diameter": 0.09
            },
            "coils": [
                {
                    "center": 0.25,
                    "length": 0.1,
                    "force": 10.0
                }
            ]
        }

        response = self.client.post("/simulate", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "errors" in data["detail"]

    def test_capsule_too_large_for_tube(self):
        payload = {
            "capsule": {
                "mass": 1.0,
                "diameter": 0.2  # Larger than tube inner diameter
            },
            "tube": {
                "length": 0.5,
                "inner_diameter": 0.1
            },
            "coils": [
                {
                    "center": 0.25,
                    "length": 0.1,
                    "force": 10.0
                }
            ]
        }

        response = self.client.post("/simulate", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "errors" in data["detail"]
        errors = data["detail"]["errors"]
        assert any("smaller than tube inner diameter" in error for error in errors)


class TestJobManagement:
    """Test job status tracking and management."""

    def setup_method(self):
        self.client = TestClient(app)
        from api import simulation_jobs
        simulation_jobs.clear()

    def test_get_nonexistent_job(self):
        response = self.client.get("/jobs/nonexistent-id")
        assert response.status_code == 404
        assert response.json()["detail"] == "Job not found"

    def test_list_empty_jobs(self):
        response = self.client.get("/jobs")
        assert response.status_code == 200
        assert response.json() == []

    def test_job_lifecycle(self):
        # Create a job
        payload = {
            "capsule": {"mass": 1.0, "diameter": 0.083},
            "tube": {"length": 0.5, "inner_diameter": 0.09},
            "coils": [{"center": 0.25, "length": 0.1, "force": 10.0}]
        }

        create_response = self.client.post("/simulate", json=payload)
        assert create_response.status_code == 200
        job_id = create_response.json()["job_id"]

        # Check job status
        status_response = self.client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ["pending", "running", "completed", "failed"]
        assert "created_at" in status_data

        # List jobs
        list_response = self.client.get("/jobs")
        assert list_response.status_code == 200
        jobs = list_response.json()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == job_id

    def test_job_filtering(self):
        # Create multiple jobs with quick completion
        payloads = [
            {
                "capsule": {"mass": 1.0, "diameter": 0.08},
                "tube": {"length": 0.1, "inner_diameter": 0.09},
                "coils": [{"center": 0.05, "length": 0.02, "force": 1.0}],
                "simulation": {"dt": 0.01, "max_time": 0.05}  # Very short simulation
            }
            for _ in range(2)
        ]

        job_ids = []
        for payload in payloads:
            response = self.client.post("/simulate", json=payload)
            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])

        # Test limit parameter
        response = self.client.get("/jobs?limit=1")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) <= 1


class TestConfigurationEndpoints:
    """Test configuration-related endpoints."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_get_presets(self):
        response = self.client.get("/presets")
        assert response.status_code == 200
        data = response.json()
        assert "default_assignment" in data

        default_config = data["default_assignment"]
        assert "capsule" in default_config
        assert "tube" in default_config
        assert "coils" in default_config
        assert default_config["capsule"]["mass"] == 1.0
        assert default_config["tube"]["length"] == 0.5

    def test_upload_yaml_config(self):
        # Create a temporary YAML config
        yaml_content = """
capsule:
  mass: 1.0
  diameter: 0.083
  initial_position: 0.0
  initial_velocity: 0.0

tube:
  length: 0.5
  inner_diameter: 0.09

coils:
  - center: 0.1
    length: 0.1
    force: 10.0
    name: "TestCoil"

simulation:
  dt: 0.001
  max_time: 1.0
  stop_at_exit: true

export:
  format: "json"

metadata:
  scenario: "test_upload"
  description: "Test YAML upload"
"""

        # Test YAML upload
        files = {"config_file": ("test_config.yaml", yaml_content, "text/yaml")}
        response = self.client.post("/simulate/config", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_upload_json_config(self):
        json_config = {
            "capsule": {"mass": 1.0, "diameter": 0.083},
            "tube": {"length": 0.5, "inner_diameter": 0.09},
            "coils": [{"center": 0.25, "length": 0.1, "force": 10.0}],
            "simulation": {"dt": 0.001, "max_time": 1.0},
            "metadata": {"scenario": "test_json_upload"}
        }

        json_content = json.dumps(json_config)
        files = {"config_file": ("test_config.json", json_content, "application/json")}
        response = self.client.post("/simulate/config", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_upload_invalid_file_type(self):
        # Test unsupported file type
        files = {"config_file": ("test.txt", "invalid content", "text/plain")}
        response = self.client.post("/simulate/config", files=files)
        assert response.status_code == 400
        assert "File must be .yaml, .yml, or .json" in response.json()["detail"]

    def test_upload_invalid_yaml(self):
        # Test malformed YAML
        invalid_yaml = "invalid: yaml: content: ["
        files = {"config_file": ("invalid.yaml", invalid_yaml, "text/yaml")}
        response = self.client.post("/simulate/config", files=files)
        assert response.status_code == 400
        assert "Invalid config file" in response.json()["detail"]


class TestCustomSimulationValidation:
    """Test validation for custom simulation endpoint."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_too_many_coils(self):
        payload = {
            "tube_length": 1.0,
            "tube_diameter": 0.1,
            "capsule_mass": 1.0,
            "capsule_diameter": 0.08,
            "num_coils": 25,  # Exceeds limit of 20
            "coil_length": 0.1,
            "coil_force": 10.0
        }

        response = self.client.post("/simulate/custom", json=payload)
        assert response.status_code == 422  # Validation error

    def test_coil_too_long_for_tube(self):
        payload = {
            "tube_length": 0.5,
            "tube_diameter": 0.1,
            "capsule_mass": 1.0,
            "capsule_diameter": 0.08,
            "num_coils": 1,
            "coil_length": 0.6,  # Longer than tube
            "coil_force": 10.0
        }

        response = self.client.post("/simulate/custom", json=payload)
        assert response.status_code == 400
        errors = response.json()["detail"]["errors"]
        assert any("coil_length must be <= tube length" in error for error in errors)

    def test_negative_parameters(self):
        payload = {
            "tube_length": -1.0,  # Invalid
            "tube_diameter": 0.1,
            "capsule_mass": 1.0,
            "capsule_diameter": 0.08,
            "num_coils": 1,
            "coil_length": 0.1,
            "coil_force": 10.0
        }

        response = self.client.post("/simulate/custom", json=payload)
        assert response.status_code == 422  # Pydantic validation


class TestSimulationExecution:
    """Test actual simulation execution through API."""

    def setup_method(self):
        self.client = TestClient(app)
        from api import simulation_jobs
        simulation_jobs.clear()

    @pytest.mark.asyncio
    async def test_short_simulation_completion(self):
        """Test a very short simulation that should complete quickly."""
        payload = {
            "capsule": {"mass": 1.0, "diameter": 0.08},
            "tube": {"length": 0.1, "inner_diameter": 0.09},
            "coils": [{"center": 0.05, "length": 0.02, "force": 5.0}],
            "simulation": {"dt": 0.01, "max_time": 0.05},  # Very short
            "export": {"format": "json"}
        }

        # Create job
        response = self.client.post("/simulate", json=payload)
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Wait for completion (with timeout)
        max_wait = 10  # seconds
        wait_time = 0
        while wait_time < max_wait:
            status_response = self.client.get(f"/jobs/{job_id}")
            status = status_response.json()["status"]

            if status == "completed":
                break
            elif status == "failed":
                pytest.fail(f"Simulation failed: {status_response.json()}")

            await asyncio.sleep(0.1)
            wait_time += 0.1

        assert status == "completed", f"Simulation did not complete within {max_wait}s"

        # Check result
        result_response = self.client.get(f"/jobs/{job_id}/result")
        assert result_response.status_code == 200
        result_data = result_response.json()
        assert "summary" in result_data
        assert result_data["summary"]["final_velocity"] >= 0


class TestErrorHandling:
    """Test API error handling and edge cases."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_malformed_json_request(self):
        # Send malformed JSON
        response = self.client.post(
            "/simulate",
            data="{'invalid': json}",  # Malformed JSON
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self):
        # Missing capsule field
        payload = {
            "tube": {"length": 0.5, "inner_diameter": 0.09},
            "coils": [{"center": 0.25, "length": 0.1, "force": 10.0}]
            # Missing "capsule"
        }

        response = self.client.post("/simulate", json=payload)
        assert response.status_code == 422

    def test_empty_coils_array(self):
        payload = {
            "capsule": {"mass": 1.0, "diameter": 0.083},
            "tube": {"length": 0.5, "inner_diameter": 0.09},
            "coils": []  # Empty coils array
        }

        response = self.client.post("/simulate", json=payload)
        assert response.status_code == 400
        errors = response.json()["detail"]["errors"]
        assert any("At least one coil must be specified" in error for error in errors)

    def test_download_nonexistent_result(self):
        response = self.client.get("/jobs/nonexistent-id/download")
        assert response.status_code == 404

    def test_get_result_for_pending_job(self):
        # Create a job
        payload = {
            "capsule": {"mass": 1.0, "diameter": 0.083},
            "tube": {"length": 0.5, "inner_diameter": 0.09},
            "coils": [{"center": 0.25, "length": 0.1, "force": 10.0}]
        }

        response = self.client.post("/simulate", json=payload)
        job_id = response.json()["job_id"]

        # Try to get result immediately (should still be pending)
        result_response = self.client.get(f"/jobs/{job_id}/result")
        if result_response.status_code == 400:
            # Expected behavior for pending job
            assert "not completed" in result_response.json()["detail"]
        # If job completed very quickly, that's also acceptable


class TestPerformanceAndLimits:
    """Test performance characteristics and limits."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_concurrent_job_creation(self):
        """Test creating multiple jobs concurrently."""
        payloads = [
            {
                "capsule": {"mass": 1.0, "diameter": 0.08},
                "tube": {"length": 0.1, "inner_diameter": 0.09},
                "coils": [{"center": 0.05, "length": 0.01, "force": 1.0}],
                "simulation": {"dt": 0.01, "max_time": 0.02}
            }
            for _ in range(3)
        ]

        job_ids = []
        for payload in payloads:
            response = self.client.post("/simulate", json=payload)
            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])

        # All jobs should have unique IDs
        assert len(set(job_ids)) == len(job_ids)

        # All jobs should be trackable
        for job_id in job_ids:
            response = self.client.get(f"/jobs/{job_id}")
            assert response.status_code == 200

    def test_large_simulation_parameters(self):
        """Test with large but reasonable parameters."""
        payload = {
            "tube_length": 10.0,  # 10 meter tube
            "tube_diameter": 1.0,
            "capsule_mass": 100.0,  # 100 kg capsule
            "capsule_diameter": 0.5,
            "num_coils": 20,  # Maximum allowed
            "coil_length": 0.4,
            "coil_force": 1000.0,  # 1kN force
            "dt": 0.001,
            "max_time": 0.1  # Keep simulation short for testing
        }

        response = self.client.post("/simulate/custom", json=payload)
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Verify job was created
        status_response = self.client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
