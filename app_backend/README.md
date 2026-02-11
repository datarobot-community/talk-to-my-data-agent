# Talk to My Data: FastAPI App app_backend

## Tech Stack

- FastAPI
- React 18 with TypeScript served as static files

## Local Development

Start both the backend FastAPI developer server and frontend React with the root `dev` task.

```sh
cd <repository root>
task dev
```

These can be started separately if desired with `task app_frontend:dev` and `task app_backend dev`.
The frontend will be available on port 5173 and the FastAPI backend on port 8080.

If in a DataRobot codespace, check the "Ports" configuration to check that port forwarding is enabled
for those ports.