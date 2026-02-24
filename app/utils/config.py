from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "development"
    debug: bool = True
    log_level: str = "info"

    data_dir: str = "./data"
    cache_dir: str = "./data/cache"

    class Config:
        env_file = ".env"


settings = Settings()