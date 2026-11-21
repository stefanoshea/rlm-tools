# ShipStation Architecture

## Overview

ShipStation is an order management platform built as a modular Python
application. The system is organised into four horizontal layers, each
with clear responsibilities and dependency rules.

## Layers

### 1. API Layer (`app/api/`)

The HTTP surface area. Route registration lives in `routes.py`, which
wires URL patterns to handler functions in `order_handlers.py`,
`catalog_handlers.py`, and `auth_handlers.py`. Middleware is composed
via `middleware_chain.py` using `MIDDLEWARE_ORDER`.

Handlers are thin: they deserialise the request, call a service, and
serialise the response. No business logic lives here.

### 2. Service Layer (`app/orders/`, `app/catalog/`, `app/auth/`)

Domain logic and orchestration. Key services:

- **OrderService** (`app/orders/service.py`) -- order lifecycle from
  creation through fulfillment and cancellation.
- **SearchService** (`app/catalog/service.py`) -- full-text product
  search backed by the search engine adapter.
- **TokenService** (`app/auth/token_service.py`) -- JWT issuance,
  validation, and refresh-token rotation. Replaces the older
  `legacy_auth` module which is no longer active.

### 3. Adapter Layer (`app/adapters/`)

Thin wrappers around external systems: database, Redis, HTTP clients,
and the search engine. Services never talk to infrastructure directly;
they go through adapters.

### 4. Worker Layer (`app/workers/`)

Background processing. The `TaskQueue` provides a simple FIFO queue.
Specialised workers include `ExportWorker` (delegates to the export
pipeline) and `NotificationWorker` (delegates to the notification
dispatcher).

## Cross-cutting Concerns

| Concern          | Module                        |
|------------------|-------------------------------|
| Configuration    | `app/settings.py`             |
| Feature flags    | `app/feature_flags.py`        |
| Metrics          | `app/metrics.py`              |
| Exception tree   | `app/exceptions.py`           |
| Logging          | `app/logging_config.py`       |
| Retry policies   | `app/retry_config.py`         |
| Audit trail      | `app/admin/audit_log.py`      |

## Design Decisions

1. **Feature-flag gated rollouts.** Major subsystem changes (e.g. the
   export pipeline v2) are gated behind `feature_flags` so they can be
   enabled per-environment without a deploy.

2. **Adapter pattern for external services.** Payment processing goes
   through `stripe_adapter.py` (current) or `paypal_adapter.py`
   (legacy). The service layer is decoupled from the provider.

3. **Reconciliation as a first-class concern.** The `reconcile/`
   package runs nightly to verify ledger consistency and emits metrics
   for alerting.

4. **Cron-scheduled exports.** The `ExportScheduler` in
   `app/export/scheduling.py` manages recurring data exports without
   requiring an external scheduler.

## Deployment

The application boots via `app/bootstrap.py`, which loads settings,
registers routes, and builds the middleware chain.
