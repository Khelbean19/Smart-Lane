# SmartLane

Smart traffic light system with vehicle classification for single-lane, bi-directional roads.
Team: Vision Vanguard (BSIT 3-3) | SDG 11 – Target 11.2

**Stack:** React.js (Vite) · FastAPI · PostgreSQL + SQLAlchemy · OpenCV + YOLOv8 · WebSocket + MJPEG · SMS API

## Setup
### Database
#### Generate or Write a Database Password
It is important to secure even the local development database with a password. Since this project's
database is set up in a Docker container, a password file inside that container must be accessible
for the database to read it later when run.

A randomly-generated password string can be written to a file inside the container using the
following command. Make sure to run this in the WSL Shell:
```bash
(umask 077; openssl rand -base64 24 > postgres-password.txt)
```

If a specific password is desired, simply run the following command to write it to a file:
```bash
echo "yourPasswordHere123!" > postgres-password.txt
```

Either way, when the postgres container is run later, it will reference the saved file and extract
the password content in it. To verify the actual password, run `cat postgres-password.txt`.

#### Running the Database
The postgres database runs on Docker with port 5432. Run the database using the following commands:
```bash
docker run --name postgres -d \
    -p 127.0.0.1:5432:5432 \
    -v postgres-data:/var/lib/postgresql \
    -v "$PWD/postgres-password.txt:/run/secrets/postgres-password:ro" \
    -e POSTGRES_PASSWORD_FILE=/run/secrets/postgres-password \
    postgres:18
```

If the port 5432 is already used, simply change the second argument in the `-p` flag with the
desired port number that is available. For example: `-p 127.0.0.1:5433:5432` which uses port 5433
to expose the postgres database in the container.

#### Connecting to pgAdmin
This is applicable if pgAdmin 4 is installed as an application to the computer. Use the following
connection settings when adding a new server connection:
- Host name/address: `localhost` or `127.0.0.1` (However, it is recommended to directly use
`127.0.0.1` for the address as it provides faster connection to the postgres server)
- Username: `postgres`
- Password: (Whatever password has been set in the `postgres-password.txt` file)


### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
cp ../.env.example ../.env   # then fill in the values
uvicorn app.main:app --reload
```

### Frontend
Folder structure only for now. The frontend dev will set up the React (Vite) app inside `frontend/`.

### Tests
```bash
cd backend
pytest
```

## Git workflow
- `main` = stable, `develop` = integration, `feature/<name>` = work branches
- Merge via pull requests. Never commit `.env` or model weights.
