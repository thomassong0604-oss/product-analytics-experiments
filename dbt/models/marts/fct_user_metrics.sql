-- One row per user: the unit of analysis, with the pre-period covariate beside the outcome.
--
-- This is the table the experiment analysis reads, and its grain is the grain randomisation used.
-- Aggregating to the randomised unit here — rather than handing sessions to the estimator — is what
-- keeps the standard error honest downstream.
with pivoted as (
    select
        user_id,
        sum(sessions)              filter (where period = 'during') as sessions,
        sum(purchases)             filter (where period = 'during') as purchases,
        sum(revenue)               filter (where period = 'during') as revenue,
        sum(sessions_with_cart)    filter (where period = 'during') as sessions_with_cart,
        sum(sessions)              filter (where period = 'pre')    as sessions_pre,
        sum(purchases)             filter (where period = 'pre')    as purchases_pre,
        sum(revenue)               filter (where period = 'pre')    as revenue_pre
    from {{ ref('int_user_period') }}
    group by user_id
)

select
    u.user_id,
    a.experiment,
    a.variant,
    a.assigned_at,
    u.signup_date,
    u.cohort_week,
    u.device,
    u.channel,
    u.country,
    coalesce(p.sessions, 0)           as sessions,
    coalesce(p.purchases, 0)          as purchases,
    coalesce(p.revenue, 0.0)          as revenue,
    coalesce(p.sessions_with_cart, 0) as sessions_with_cart,
    coalesce(p.sessions_pre, 0)       as sessions_pre,
    coalesce(p.purchases_pre, 0)      as purchases_pre,
    coalesce(p.revenue_pre, 0.0)      as revenue_pre,
    coalesce(p.purchases, 0) > 0      as converted,
    case when coalesce(p.sessions, 0) > 0
         then coalesce(p.purchases, 0)::double / p.sessions end as purchases_per_session
from {{ ref('stg_users') }} u
inner join {{ ref('stg_assignment') }} a using (user_id)
left join pivoted p using (user_id)
