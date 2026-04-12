from pydantic import BaseModel

class HouseInput(BaseModel):
    house_size: int
    bedrooms: int
    car_space: int