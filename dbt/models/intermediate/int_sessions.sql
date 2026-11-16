-- One row per session: when it happened, how far down the funnel it got, what it earned.
--
-- The funnel flags come from conditional aggregation over the session's events rather than from a
-- self-join per step. One pass, and adding a step is one more line instead of another join.
with events as (
    select * from {{ ref('stg_events') }}
),

per_session as (
    select
        session_id,
        user_id,
        min(event_ts) as session_start_ts,
        max(event_ts) as session_end_ts,
        count(*)      as event_count,
        max(event_name = 'product_view')   as reached_product_view,
        max(event_name = 'add_to_cart')    as reached_add_to_cart,
        max(event_name = 'checkout_start') as reached_checkout_start,
        max(event_name = 'purchase')       as reached_purchase,
        max(funnel_step)                   as deepest_funnel_step,
        sum(revenue)                       as revenue,
        count(*) filter (where event_name = 'purchase') as purchases
    from events
    group by session_id, user_id
),

ordered as (
    select
        *,
        -- Where this session sits in the user's own history, and how long since the last one.
        -- Window functions rather than a self-join: one scan, and the ordering is explicit.
        row_number() over (partition by user_id order by session_start_ts) as session_index,
        lag(session_start_ts) over (partition by user_id order by session_start_ts)
            as previous_session_ts,
        count(*) over (partition by user_id) as user_total_sessions
    from per_session
)

select
    session_id,
    user_id,
    session_start_ts,
    session_end_ts,
    cast(session_start_ts as date)         as session_date,
    date_trunc('week', session_start_ts)   as session_week,
    session_index,
    user_total_sessions,
    previous_session_ts,
    date_diff('hour', previous_session_ts, session_start_ts) as hours_since_previous_session,
    session_index = 1                      as is_first_session,
    event_count,
    deepest_funnel_step,
    reached_product_view,
    reached_add_to_cart,
    reached_checkout_start,
    reached_purchase,
    purchases,
    revenue
from ordered
