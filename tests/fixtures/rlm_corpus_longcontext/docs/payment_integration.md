# Payment Integration

## Current Provider: Stripe

All new payment processing flows through `app/adapters/stripe_adapter.py`.
The adapter wraps the Stripe Python SDK and handles:

- **Charge creation** -- called by `OrderService` at checkout.
- **Refund processing** -- triggered when an admin cancels an order via
  `cancel_order_handler`.
- **Webhook verification** -- incoming Stripe events are verified using
  the `STRIPE_WEBHOOK_SECRET` from `app/settings.py`. The verification
  flow is:
  1. Read the raw request body (must be bytes, not parsed JSON).
  2. Extract the `Stripe-Signature` header.
  3. Call `stripe.Webhook.construct_event(body, sig, secret)`.
  4. Route the verified event to `app/payments/webhooks.py`.

## Legacy Provider: PayPal (deprecated)

The PayPal integration in `app/adapters/paypal_adapter.py` is **legacy**
and no longer processes live transactions. It remains in the codebase
for two reasons:

1. Historical order records reference PayPal transaction IDs, and the
   adapter is used in read-only mode for support lookups.
2. The reconciliation service (`app/reconcile/service.py`) still
   cross-references PayPal settlement files for orders placed before
   the Stripe migration.

No new code should import `paypal_adapter` for write operations.

## Migration Timeline

| Date       | Milestone                                  |
|------------|--------------------------------------------|
| 2024-01-15 | Stripe adapter merged                      |
| 2024-02-01 | Feature flag `USE_STRIPE` enabled (10%)    |
| 2024-03-01 | Full rollout, PayPal writes disabled       |
| 2024-06-01 | PayPal read-only mode formalised           |

## Testing

Payment tests live in `tests/test_payments.py` and mock the Stripe SDK
at the adapter boundary. See `tests/conftest.py` for shared fixtures.
