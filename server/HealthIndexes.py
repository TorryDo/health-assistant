from pydantic import BaseModel


class HealthIndexes(BaseModel):
    pulse: float
    spo2: float
    temperature: float
    timeInMillis: int
    status: int

    def to_dict(self):
        return {
            'pulse': self.pulse,
            'spo2': self.spo2,
            'temperature': self.temperature,
            'timeInMillis': self.timeInMillis,
            'status': self.status
        }
