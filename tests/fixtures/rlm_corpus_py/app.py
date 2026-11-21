from services.auth_service import AuthService, TokenStore
from services.user_service import UserRepository, UserService


class DashboardController:
    def __init__(self) -> None:
        token_store = TokenStore()
        user_repository = UserRepository()
        auth_service = AuthService(token_store)
        self.user_service = UserService(user_repository, auth_service)

    def load_dashboard(self, user_id: str) -> dict:
        return self.user_service.load_user_profile(user_id)
