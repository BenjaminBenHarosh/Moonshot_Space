# Capsule Acceleration Simulator

A 1D electromagnetic capsule acceleration simulation service with REST API, built for the Moonshot Space assignment.

---

## Features

- **Physical Simulation**: 1D capsule acceleration through electromagnetic coils
- **REST API**: FastAPI service with Swagger documentation
- **Multiple Export Formats**: JSON, Parquet, HDF5 with compression options
- **Configuration Management**: YAML/JSON config files with validation
- **Containerized**: Docker and docker-compose ready
- **Production Ready**: Health checks, logging, error handling

---

## Assignment Requirements

**Core Requirements**:
- [x] 1D capsule acceleration simulation
- [x] Electromagnetic coil force model
- [x] OOP design with clean architecture
- [x] Binary/structured output (Parquet, HDF5, JSON)
- [x] Position, velocity, acceleration vs time
- [x] Coil engagement logging
- [x] Configurable time resolution and integration

**Implementation Requirements**:
- [x] Python with clean, modular codebase
- [x] Public API with Swagger documentation
- [x] Dockerfile and docker-compose
- [x] Makefile for build/test operations
- [x] Sample run configuration

**Bonus Features**:
- [x] Time series data export
- [x] Professional API architecture
- [x] Background job processing
- [x] Multiple export formats

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone <your-repo-url>
cd Moonshot_Space

# Build and run with Docker Compose
make docker-run

# API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Install dependencies
make install

# Run API server
make run

# Or run example scenarios
make demo
```

---

## Sample Scenario & Results

**Default Assignment Scenario**:
- Capsule: 1 kg, 83 mm diameter
- Tube: 0.5 m length, 90 mm inner diameter  
- 6 coils, 0.3 m length each, 10 N force
- **Result**: ~6.01 m/s exit velocity in 0.23 seconds

Config file: [`configs/sample_config.yaml`](configs/sample_config.txt)  
Sample output: [`outputs/default_scenario.parquet`](outputs/)  

---

## API Usage

### Start a Simulation

```bash
curl -X POST "http://localhost:8000/simulate"   -H "Content-Type: application/json"   -d '{
    "capsule": {"mass": 1.0, "diameter": 0.083},
    "tube": {"length": 0.5, "inner_diameter": 0.09},
    "coils": [
      {"center": 0.075, "length": 0.3, "force": 10.0, "name": "C1"},
      {"center": 0.15, "length": 0.3, "force": 10.0, "name": "C2"}
    ],
    "simulation": {"dt": 0.001, "max_time": 2.0, "stop_at_exit": true}
  }'
```

### List Jobs

```bash
curl "http://localhost:8000/jobs"
```

### Check Job Status

```bash
curl "http://localhost:8000/jobs/{job_id}"
```

### Download Results

```bash
curl "http://localhost:8000/results/{job_id}" -o results.parquet
```

### Using Configuration Files

```bash
curl -X POST "http://localhost:8000/simulate/config"   -F "config_file=@configs/sample_config.yaml"
```

---

## Project Structure

```
Moonshot_Space/
├── simulator.py           # Core physics simulation
├── exporter.py            # Data export utilities
├── config.py              # Configuration management
├── api.py                 # FastAPI service
├── configs/
│   └── sample_config.yaml # Default scenario config
├── outputs/               # Simulation results (sample parquet committed)
├── tests/                 # Unit tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Physics Model

### Force Model
- **Span-based acceleration**: Constant force while capsule is within coil span [start, end]
- **Force superposition**: Multiple coil forces add linearly when spans overlap
- **No force outside coils**: Constant velocity motion in free space

### Integration
- **Semi-implicit Euler**: Velocity updated before position for numerical stability
- **Configurable time step**: Default 1ms, adjustable for accuracy vs performance

### Energy Conservation
- Kinetic energy: KE = ½mv²
- Work done: W = F·d for each coil span
- Conservation verified after tube exit (ballistic phase)

---

## Output Data

### Time Series
- Position vs time (m)
- Velocity vs time (m/s)  
- Acceleration vs time (m/s²)

### Event Logging
- Coil engagement/disengagement
- Tube exit events
- Force application history

### Export Formats
- **Parquet** (recommended, snappy-compressed)
- **HDF5** 
- **JSON** (optionally gzipped)

---

## Development

### Run Tests
```bash
make test
```

### Code Quality
```bash
make lint      # Check code style
make format    # Auto-format
```

### Build Image
```bash
make docker-build
```

---

## Deployment

### Production Deployment
```bash
# With PostgreSQL + Nginx (see docker-compose.yml)
make deploy
```

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (optional)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING)
- `ENVIRONMENT`: Environment name (development, production)

---

## API Documentation

Interactive docs available at:
- **Swagger UI**: http://localhost:8000/docs  
- **ReDoc**: http://localhost:8000/redoc  

---

## Author

Benjamin David Ben Harosh – Moonshot Space Assignment
