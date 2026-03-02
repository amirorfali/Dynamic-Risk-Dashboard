from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "development"
    debug: bool = True
    log_level: str = "info"

    data_dir: str = "./data"
    cache_dir: str = "./data/cache"

    model_config = ConfigDict(env_file=".env")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str) and value.lower() == "release":
            return False
        return value


settings = Settings()
