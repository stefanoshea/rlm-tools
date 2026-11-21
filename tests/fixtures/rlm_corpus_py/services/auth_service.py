class TokenStore:
    def create(self, user_id: str) -> str:
        return f"token-{user_id}"


class AuthService:
    def __init__(self, token_store: TokenStore) -> None:
        self.token_store = token_store

    def issue_token(self, user_id: str) -> str:
        return self.token_store.create(user_id)
