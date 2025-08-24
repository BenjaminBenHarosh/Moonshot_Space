from __future__ import annotations
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import uuid
import logging
from pathlib import Path
import json
from datetime import datetime

from simulator import Capsule, Tube, Coil, Simulator
from exporter import export_timeseries
from config import SimulationSpec, CapsuleConfig, TubeConfig, CoilConfig, SimulationConfig, ExportConfig, \
    validate_simulation_spec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Capsule Acceleration Simulator API",
    description="1D electromagnetic capsule acceleration simulation service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (replace with Redis/DB in production)
simulation_jobs: Dict[str, Dict[str, Any]] = {}


# Pydantic models for API
class CapsuleRequest(BaseModel):
    mass: float = Field(..., gt=0, description="Capsule mass in kg")
    diameter: float = Field(..., gt=0, description="Capsule diameter in m")
    initial_position: float = Field(0.0, ge=0, description="Initial position in m")
    initial_velocity: float = Field(0.0, ge=0, description="Initial velocity in m/s")


class TubeRequest(BaseModel):
    length: float = Field(..., gt=0, description="Tube length in m")
    inner_diameter: float = Field(..., gt=0, description="Tube inner diameter in m")


class CoilRequest(BaseModel):
    center: float = Field(..., ge=0, description="Coil center position in m")
    length: float = Field(..., gt=0, description="Coil active length in m")
    force: float = Field(..., ge=0, description="Coil force in N")
    name: Optional[str] = Field(None, description="Optional coil name")


class SimulationRequest(BaseModel):
    dt: float = Field(1e-3, gt=0, le=0.01, description="Time step in seconds")
    max_time: float = Field(10.0, gt=0, description="Maximum simulation time in s")
    stop_at_exit: bool = Field(True, description="Stop simulation when capsule exits tube")


class ExportRequest(BaseModel):
    format: str = Field("parquet", pattern="^(json|json\\.gz|parquet|hdf5)$", description="Export format")
    json_compress: bool = Field(True, description="Compress JSON output")
    parquet_compression: str = Field("snappy", description="Parquet compression algorithm")


class SimulationJobRequest(BaseModel):
    capsule: CapsuleRequest
    tube: TubeRequest
    coils: List[CoilRequest] = Field(..., min_items=1, description="List of coils")
    simulation: SimulationRequest = Field(default_factory=SimulationRequest)
    export: ExportRequest = Field(default_factory=ExportRequest)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(0.0, ge=0, le=1.0)
    error_message: Optional[str] = None
    result_path: Optional[str] = None


class SimulationSummary(BaseModel):
    final_time: float
    exit_time: Optional[float]
    final_position: float
    final_velocity: float
    final_kinetic_energy: float
    num_events: int


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "service": "Capsule Acceleration Simulator",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "active_jobs": len([j for j in simulation_jobs.values() if j["status"] in ["pending", "running"]]),
        "total_jobs": len(simulation_jobs),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/simulate", response_model=Dict[str, str], tags=["Simulation"])
async def create_simulation_job(
        job_request: SimulationJobRequest,
        background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Create a new simulation job."""

    # Convert to internal config format
    spec = SimulationSpec(
        capsule=CapsuleConfig(**job_request.capsule.dict()),
        tube=TubeConfig(**job_request.tube.dict()),
        coils=[CoilConfig(**coil.dict()) for coil in job_request.coils],
        simulation=SimulationConfig(**job_request.simulation.dict()),
        export=ExportConfig(**job_request.export.dict()),
        metadata=job_request.metadata
    )

    # Validate specification
    errors = validate_simulation_spec(spec)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Create job
    job_id = str(uuid.uuid4())
    simulation_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(),
        "started_at": None,
        "completed_at": None,
        "progress": 0.0,
        "error_message": None,
        "result_path": None,
        "spec": spec,
        "summary": None
    }

    # Start background task
    background_tasks.add_task(run_simulation, job_id, spec)

    logger.info(f"Created simulation job {job_id}")
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job_status(job_id: str) -> JobStatus:
    """Get status of a simulation job."""
    if job_id not in simulation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = simulation_jobs[job_id]
    return JobStatus(**job)


@app.get("/jobs", tags=["Jobs"])
async def list_jobs(limit: int = 50, status: Optional[str] = None) -> List[JobStatus]:
    """List simulation jobs."""
    jobs = list(simulation_jobs.values())

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    # Sort by creation time, most recent first
    jobs.sort(key=lambda x: x["created_at"], reverse=True)

    return [JobStatus(**job) for job in jobs[:limit]]


@app.get("/jobs/{job_id}/result", tags=["Results"])
async def get_job_result(job_id: str) -> Dict[str, Any]:
    """Get simulation result summary."""
    if job_id not in simulation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = simulation_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}, not completed")

    return {
        "job_id": job_id,
        "summary": job["summary"],
        "result_path": job["result_path"],
        "metadata": job["spec"].metadata
    }


@app.get("/jobs/{job_id}/download", tags=["Results"])
async def download_result_file(job_id: str):
    """Download simulation result file."""
    if job_id not in simulation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = simulation_jobs[job_id]
    if job["status"] != "completed" or not job["result_path"]:
        raise HTTPException(status_code=400, detail="Result file not available")

    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    return FileResponse(
        path=result_path,
        filename=f"simulation_{job_id}_{result_path.suffix}",
        media_type="application/octet-stream"
    )


@app.post("/simulate/config", tags=["Configuration"])
async def simulate_from_config(
        background_tasks: BackgroundTasks,
        config_file: UploadFile = File(...)
) -> Dict[str, str]:
    """Create simulation job from uploaded YAML/JSON config file."""

    # Read uploaded file
    content = await config_file.read()

    try:
        if config_file.filename.endswith('.yaml') or config_file.filename.endswith('.yml'):
            import yaml
            data = yaml.safe_load(content.decode())
        elif config_file.filename.endswith('.json'):
            data = json.loads(content.decode())
        else:
            raise HTTPException(status_code=400, detail="File must be .yaml, .yml, or .json")

        spec = SimulationSpec.from_dict(data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config file: {str(e)}")

    # Validate and create job (similar to regular simulate endpoint)
    errors = validate_simulation_spec(spec)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    job_id = str(uuid.uuid4())
    simulation_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(),
        "started_at": None,
        "completed_at": None,
        "progress": 0.0,
        "error_message": None,
        "result_path": None,
        "spec": spec,
        "summary": None
    }

    background_tasks.add_task(run_simulation, job_id, spec)

    logger.info(f"Created simulation job {job_id} from config file {config_file.filename}")
    return {"job_id": job_id, "status": "pending"}


class CustomSimulationRequest(BaseModel):
    tube_length: float = Field(..., gt=0, description="Tube length in meters")
    tube_diameter: float = Field(..., gt=0, description="Tube inner diameter in meters")
    capsule_mass: float = Field(..., gt=0, description="Capsule mass in kg")
    capsule_diameter: float = Field(..., gt=0, description="Capsule diameter in meters")
    num_coils: int = Field(..., gt=0, le=20, description="Number of coils (max 20)")
    coil_length: float = Field(..., gt=0, description="Length of each coil in meters")
    coil_force: float = Field(..., gt=0, description="Force per coil in Newtons")
    dt: float = Field(1e-3, gt=0, le=0.01, description="Time step in seconds")
    max_time: float = Field(10.0, gt=0, description="Maximum simulation time")
    stop_at_exit: bool = Field(True, description="Stop when capsule exits tube")
    export_format: str = Field("parquet", pattern="^(json|parquet|hdf5)$", description="Export format")


@app.post("/simulate/custom", response_model=Dict[str, str], tags=["Simulation"])
async def create_custom_simulation(
        request: CustomSimulationRequest,
        background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Create a custom simulation with simplified parameters."""

    # Create configuration from simple parameters
    spec = SimulationSpec(
        capsule=CapsuleConfig(
            mass=request.capsule_mass,
            diameter=request.capsule_diameter,
            initial_position=0.0,
            initial_velocity=0.0
        ),
        tube=TubeConfig(
            length=request.tube_length,
            inner_diameter=request.tube_diameter
        ),
        coils=[],  # Will be generated automatically
        simulation=SimulationConfig(
            dt=request.dt,
            max_time=request.max_time,
            stop_at_exit=request.stop_at_exit
        ),
        export=ExportConfig(
            format=request.export_format,
            base_path="outputs/custom_simulation"
        ),
        metadata={
            "scenario": "custom_api",
            "description": f"Custom simulation: {request.num_coils} coils, {request.coil_force}N each"
        }
    )

    # Generate coils using evenly_spaced_coils_inside
    from simulator import evenly_spaced_coils_inside, Tube
    tube_obj = Tube(length=request.tube_length, inner_diameter=request.tube_diameter)
    coils = evenly_spaced_coils_inside(tube_obj, request.num_coils, request.coil_length, request.coil_force)

    # Convert to config format
    spec.coils = [
        CoilConfig(
            center=coil.center,
            length=coil.length,
            force=coil.force,
            name=coil.name
        )
        for coil in coils
    ]

    # Validate specification
    errors = validate_simulation_spec(spec)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Create job
    job_id = str(uuid.uuid4())
    simulation_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(),
        "started_at": None,
        "completed_at": None,
        "progress": 0.0,
        "error_message": None,
        "result_path": None,
        "spec": spec,
        "summary": None
    }

    # Start background task
    background_tasks.add_task(run_simulation, job_id, spec)

    logger.info(f"Created custom simulation job {job_id}")
    return {"job_id": job_id, "status": "pending"}


@app.get("/presets", tags=["Configuration"])
async def get_preset_configurations() -> Dict[str, Dict[str, Any]]:
    """Get predefined simulation configurations."""
    from config import DEFAULT_ASSIGNMENT_CONFIG

    presets = {
        "default_assignment": DEFAULT_ASSIGNMENT_CONFIG.to_dict()
    }

    return presets


# Background task to run simulation
async def run_simulation(job_id: str, spec: SimulationSpec):
    """Run simulation in background."""
    try:
        # Update job status
        simulation_jobs[job_id]["status"] = "running"
        simulation_jobs[job_id]["started_at"] = datetime.now()
        simulation_jobs[job_id]["progress"] = 0.1

        logger.info(f"Starting simulation job {job_id}")

        # Convert config to simulator objects
        capsule = Capsule(
            mass=spec.capsule.mass,
            diameter=spec.capsule.diameter,
            x0=spec.capsule.initial_position,
            v0=spec.capsule.initial_velocity
        )

        tube = Tube(
            length=spec.tube.length,
            inner_diameter=spec.tube.inner_diameter
        )

        coils = [
            Coil(
                center=coil.center,
                length=coil.length,
                force=coil.force,
                name=coil.name
            )
            for coil in spec.coils
        ]

        simulation_jobs[job_id]["progress"] = 0.2

        # Create and run simulator
        simulator = Simulator(
            capsule=capsule,
            tube=tube,
            coils=coils,
            dt=spec.simulation.dt,
            max_time=spec.simulation.max_time,
            stop_at_exit=spec.simulation.stop_at_exit
        )

        simulation_jobs[job_id]["progress"] = 0.3

        # Run simulation
        result = simulator.run()

        simulation_jobs[job_id]["progress"] = 0.8

        # Export results
        output_path = export_timeseries(
            result,
            base_path=f"{spec.export.base_path}_{job_id}",
            fmt=spec.export.format,
            json_compress=spec.export.json_compress,
            parquet_compression=spec.export.parquet_compression
        )

        simulation_jobs[job_id]["progress"] = 0.9

        # Create summary
        summary = SimulationSummary(
            final_time=float(result.time[-1]) if len(result.time) > 0 else 0.0,
            exit_time=result.metadata.get('exit_time_s'),
            final_position=result.metadata.get('final_position_m', 0.0),
            final_velocity=result.metadata.get('final_velocity_mps', 0.0),
            final_kinetic_energy=0.5 * spec.capsule.mass * (result.metadata.get('final_velocity_mps', 0.0) ** 2),
            num_events=len(result.events)
        )

        # Update job as completed
        simulation_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.now(),
            "progress": 1.0,
            "result_path": output_path,
            "summary": summary.dict()
        })

        logger.info(f"Completed simulation job {job_id}")

    except Exception as e:
        logger.error(f"Simulation job {job_id} failed: {str(e)}")
        simulation_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.now(),
            "error_message": str(e)
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)