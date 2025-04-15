from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class GithubConf(BaseSettings):
    api_url: str = Field(default="")
    auth_token: str = Field(default="")

    class Config:
        env_prefix = "github_"
        env_file = Path(__file__).parents[2].joinpath(".env")
        extra = "ignore"
