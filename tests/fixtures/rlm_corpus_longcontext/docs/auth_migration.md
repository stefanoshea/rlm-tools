# Auth Migration: legacy_auth to token_service

## Background

The original authentication system lives in `app/auth/legacy_auth.py`.
It was written early in the project and handles v1 API tokens using a
symmetric HMAC algorithm defined in `legacy_auth.TOKEN_ALGORITHM`.

The new system, `app/auth/token_service.py`, was introduced to support
refresh-token rotation, scope-based permissions, and asymmetric key
signing.

## Current State

Both modules coexist in the codebase:

- **`legacy_auth.py`** handles v1 API tokens. These tokens are still
  accepted by the middleware for backwards compatibility. The algorithm
  constant `legacy_auth.TOKEN_ALGORITHM` is set to `"HS256"` and is
  referenced by several integration partners.

- **`token_service.py`** issues new tokens using the algorithm
  configured in `settings.yaml` (`auth.token_algorithm`, currently
  `"HS512"`). All new signups and logins go through this module.

## Migration Plan

1. **Phase 1 (complete):** New endpoints (`/auth/login`,
   `/auth/signup`, `/auth/refresh`) use `token_service` exclusively.

2. **Phase 2 (in progress):** Middleware accepts both v1 and v2 tokens.
   The `AuthMiddleware` checks the token header to determine which
   validator to invoke. If the `alg` field is `HS256`, it falls back
   to `legacy_auth.validate_token()`.

3. **Phase 3 (planned):** Force-expire all v1 tokens and remove the
   `legacy_auth` module. Target date: 2024-Q4.

## Key Differences

| Aspect            | legacy_auth          | token_service        |
|-------------------|----------------------|----------------------|
| Algorithm         | HS256                | HS512                |
| Refresh tokens    | Not supported        | Supported            |
| Token rotation    | N/A                  | Automatic            |
| Scope claims      | No                   | Yes                  |

## Notes

Until Phase 3 is complete, `legacy_auth.TOKEN_ALGORITHM` remains the
source of truth for v1 token validation. Do not remove it prematurely.
