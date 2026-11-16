-- One row per user per period, where "period" is before or during the experiment.
--
-- The split is what makes a pre-period covariate available to CUPED. The boundary is the earliest
-- assignment timestamp, taken from the data rather than hardcoded, so a reseed with different dates
-- cannot silently put pre-period sessions inside the experiment.
with boundary as (
    select min(assigned_at) as experiment_start from {{ ref('stg_assignment') }}
),

sessions as (
    select
        s.*,
        case when s.session_start_ts < b.experiment_start then 'pre' else 'during' end as period
    from {{ ref('int_sessions') }} s
    cross join boundary b
)

select
    user_id,
    period,
    count(*)                                      as sessions,
    sum(purchases)                                as purchases,
    sum(revenue)                                  as revenue,
    count(*) filter (where reached_add_to_cart)   as sessions_with_cart,
    count(*) filter (where reached_purchase)      as sessions_with_purchase,
    min(session_start_ts)                         as first_session_ts,
    max(session_start_ts)                         as last_session_ts
from sessions
group by user_id, period
