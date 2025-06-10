from fastapi import FastAPI
from dotenv import load_dotenv

import routes  # local import

load_dotenv()

app = FastAPI(title="Career Builder API")
app.include_router(routes.router)

@app.get("/")
def root():
    return {"message": "server working"}
