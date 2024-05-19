from pydantic import BaseModel


class HealthIndexes(BaseModel):
    pulse: float
    spo2: float
    temperature: float
    timeInMillis: int
    status: int
