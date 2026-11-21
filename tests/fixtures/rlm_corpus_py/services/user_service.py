from services.auth_service import AuthService


class UserRepository:
    def fetch_user(self, user_id: str) -> dict:
        return {"id": user_id, "name": "Taylor"}


class UserService:
    def __init__(self, user_repository: UserRepository, auth_service: AuthService) -> None:
        self.user_repository = user_repository
        self.auth_service = auth_service

    def load_user_profile(self, user_id: str) -> dict:
        token = self.auth_service.issue_token(user_id)
        profile = self.user_repository.fetch_user(user_id)
        profile["token"] = token
        return profile
