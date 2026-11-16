-- A session cannot reach a later funnel step without the earlier ones. If this fails, either the
-- event stream is out of order or int_sessions is assembling flags incorrectly — and every funnel
-- conversion rate above it is wrong.
select session_id, reached_product_view, reached_add_to_cart, reached_checkout_start, reached_purchase
from {{ ref('int_sessions') }}
where (reached_purchase and not reached_checkout_start)
   or (reached_checkout_start and not reached_add_to_cart)
   or (reached_add_to_cart and not reached_product_view)
