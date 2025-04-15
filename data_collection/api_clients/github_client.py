from github import Github, Auth
from github.Repository import Repository

from data_collection.api_clients.confs import GithubConf


class GithubClient:
    config: GithubConf
    auth: Auth.Token

    def __init__(self, config: GithubConf = GithubConf()):
        self.config = config
        self.auth = Auth.Token(self.config.auth_token)

    def get_git_repo(self, repo_identifier: str) -> Repository:
        """
        repo_identifier is expected to be in the form: owner/repository as seen in git urls:
        https://github.com/Microsoft/TypeScript.git --> microsoft/typescript
        """
        github = Github(auth=self.auth, per_page=60)
        repo = github.get_repo(repo_identifier)
        return repo
